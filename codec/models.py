from __future__ import print_function
import os
import io
import sys
import time
import math
import random
import numpy as np
import subprocess as sp
import shlex
import cv2

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Function
from torchvision import transforms
sys.path.append('..')
from codec.deepcod import DeepCOD
from compressai.layers import GDN,ResidualBlock
from codec.entropy_models import EntropyBottleneck2
from datasets.clip import *

# compress I frames with an image compression alg, e.g., DeepCOD, bpg, CA, none
# compress P frames wth RLVC
# no size estimation is performed on the first/last P frame
# loss can be psnr,ms-ssim

# Goal: convert images/videos that are compressed by RLVC
# GOP_size = args.f_P + args.b_P + 1
# Output: compressed images, predicted bits, actual bits
class LearnedVideoCodecs(nn.Module):
    def __init__(self, name, channels=64):
        super(LearnedVideoCodecs, self).__init__()
        self.name = name # 'MRLVC-BASE', 'MRLVC-RPM', 'MRLVC-RHP',RLVC,DVC
        device = torch.device('cuda')
        self.optical_flow = OpticalFlowNet()
        self.MC_network = MCNet()
        self.image_coder_name = 'deepcod' # or BPG or none
        self._image_coder = DeepCOD() if self.image_coder_name == 'deepcod' else None
        self.mv_codec = ComprNet(device, 'mv', self.name, in_channels=2, channels=channels, kernel1=3, padding1=1, kernel2=4, padding2=1)
        self.res_codec = ComprNet(device, 'res', self.name, in_channels=3, channels=channels, kernel1=5, padding1=2, kernel2=6, padding2=2)
        self.channels = channels
        
        # split on multi-gpus
        self.split()

    def split(self):
        if self._image_coder is not None:
            self._image_coder.cuda(0)
        self.optical_flow.cuda(0)
        self.mv_codec.cuda(0)
        self.MC_network.cuda(1)
        self.res_codec.cuda(1)

    def forward(self, Y0_com, Y1_raw, hidden_states, RPM_flag, use_psnr=True):
        # Y0_com: compressed previous frame
        # Y1_raw: uncompressed current frame
        gamma_0, gamma_1, gamma_2, gamma_3 = 1,1,.01,.01
        batch_size, _, Height, Width = Y1_raw.shape
        if Y0_com is None:
            Y1_com, bpp_est, img_loss, aux_loss, flow_loss, bpp_act, metrics = I_compression(Y1_raw,self.image_coder_name,self._image_coder,use_psnr)
            return Y1_com, hidden_states, gamma_0*bpp_est, gamma_1*img_loss, gamma_2*aux_loss, gamma_3*flow_loss, bpp_act, metrics
        # otherwise, it's P frame
        # hidden states
        rae_mv_hidden, rae_res_hidden, rpm_mv_hidden, rpm_res_hidden = hidden_states
        # estimate optical flow
        mv_tensor, l0, l1, l2, l3, l4 = self.optical_flow(Y0_com, Y1_raw, batch_size, Height, Width)
        # compress optical flow
        mv_hat,rae_mv_hidden,rpm_mv_hidden,mv_act,mv_est,mv_aux = self.mv_codec(mv_tensor, rae_mv_hidden, rpm_mv_hidden, RPM_flag)
        # motion compensation
        loc = get_grid_locations(batch_size, Height, Width).type(Y0_com.type())
        Y1_warp = F.grid_sample(Y0_com, loc + mv_hat.permute(0,2,3,1), align_corners=True)
        warp_loss = calc_loss(Y1_raw, Y1_warp.to(Y1_raw.device), use_psnr)
        MC_input = torch.cat((mv_hat, Y0_com, Y1_warp), axis=1)
        Y1_MC = self.MC_network(MC_input.cuda(1))
        mc_loss = calc_loss(Y1_raw, Y1_MC.to(Y1_raw.device), use_psnr)
        # compress residual
        res_tensor = Y1_raw.cuda(1) - Y1_MC
        res_hat,rae_res_hidden,rpm_res_hidden,res_act,res_est,res_aux = self.res_codec(res_tensor, rae_res_hidden, rpm_res_hidden, RPM_flag)
        # reconstruction
        Y1_com = torch.clip(res_hat + Y1_MC, min=0, max=1)
        ##### compute bits
        # estimated bits
        bpp_est = (mv_est + res_est.cuda(0))/(Height * Width * batch_size)
        # actual bits
        bpp_act = (mv_act + res_act.to(mv_act.device))/(Height * Width * batch_size)
        # auxilary loss
        aux_loss = (mv_aux + res_aux.to(mv_aux.device))/2
        # calculate metrics/loss
        metrics = calc_metrics(Y1_raw, Y1_com.to(Y1_raw.device), use_psnr)
        rec_loss = calc_loss(Y1_raw, Y1_com.to(Y1_raw.device), use_psnr)
        img_loss = (rec_loss + warp_loss + mc_loss)/3
        flow_loss = (l0+l1+l2+l3+l4)/5*1024
        # hidden states
        hidden_states = (rae_mv_hidden.detach(), rae_res_hidden.detach(), rpm_mv_hidden, rpm_res_hidden)
        return Y1_com.cuda(0), hidden_states, gamma_0*bpp_est, gamma_1*img_loss, gamma_2*aux_loss, gamma_3*flow_loss, bpp_act, metrics
        
    def update_cache(self, base_path, imgpath, train, shape, dataset, transform, \
                    frame_idx, GOP, clip_duration, sampling_rate, cache, startNewClip):
        if startNewClip:
            # read raw video clip
            clip = read_video_clip(base_path, imgpath, train, clip_duration, sampling_rate, shape, dataset)
            if transform is not None:
                clip = [transform(img).cuda() for img in clip]
            # create cache
            cache['clip'] = clip
            cache['bpp_est'] = {}
            cache['img_loss'] = {}
            cache['flow_loss'] = {}
            cache['aux'] = {}
            cache['bpp_act'] = {}
            cache['metrics'] = {}
            # compress from the first frame of the first clip to the current frame
            Iframe_idx = (frame_idx - (clip_duration-1) * sampling_rate - 1)//GOP*GOP
            Iframe_idx = max(0,Iframe_idx)
            for i in range(Iframe_idx,frame_idx):
                self._process_single_frame(i, GOP, cache, i==Iframe_idx)
        else:
            self._process_single_frame(frame_idx-1, GOP, cache, False)
            
    def _process_single_frame(self, i, GOP, cache, isNew):
        # frame shape
        _,h,w = cache['clip'][0].shape
        # frames to be processed
        Y0_com = cache['clip'][i-1].unsqueeze(0) if i>0 else None
        Y1_raw = cache['clip'][i].unsqueeze(0)
        # hidden variables
        if isNew:
            rae_mv_hidden, rae_res_hidden = init_hidden(h,w,self.channels)
            rpm_mv_hidden, rpm_res_hidden = self.mv_codec.entropy_bottleneck.init_state(), self.res_codec.entropy_bottleneck.init_state()
            hidden = (rae_mv_hidden, rae_res_hidden, rpm_mv_hidden, rpm_res_hidden)
        else:
            hidden = cache['hidden']
        RPM_flag = False
        if i%GOP == 0:
            Y0_com = None
        elif i%GOP >= 2:
            RPM_flag = True
        Y1_com,hidden,bpp_est,img_loss,aux_loss,flow_loss,bpp_act,metrics = self(Y0_com, Y1_raw, hidden, RPM_flag)
        cache['hidden'] = hidden
        cache['clip'][i] = Y1_com.detach().squeeze(0)
        cache['img_loss'][i] = img_loss
        cache['flow_loss'][i] = flow_loss
        cache['aux'][i] = aux_loss
        cache['bpp_est'][i] = bpp_est
        cache['metrics'][i] = metrics
        cache['bpp_act'][i] = bpp_act.cpu()
        cache['max_idx'] = i
    
    def loss(self, app_loss, pix_loss, bpp_loss, aux_loss, flow_loss):
        if self.name[:5] == 'MRLVC':
            return app_loss + pix_loss + bpp_loss + aux_loss + flow_loss
        elif self.name == 'RLVC' or self.name == 'DVC':
            return pix_loss + bpp_loss + aux_loss + flow_loss
        else:
            print('Loss not implemented')
            exit(1)
        
    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name.endswith("._offset") or name.endswith("._quantized_cdf") or name.endswith("._cdf_length"):
                 continue
            own_state[name].copy_(param)
        
class StandardVideoCodecs(nn.Module):
    def __init__(self, name):
        super(StandardVideoCodecs, self).__init__()
        self.name = name # x264, x265?
        self.placeholder = torch.nn.Parameter(torch.zeros(1))
        
    def update_cache(self, base_path, imgpath, train, shape, dataset, transform, \
                    frame_idx, GOP, clip_duration, sampling_rate, cache, startNewClip):
        if startNewClip:
            # read raw video clip
            raw_clip = read_video_clip(base_path, imgpath, train, clip_duration, sampling_rate, shape, dataset)
            imgByteArr = io.BytesIO()
            width,height = shape
            fps = 25
            output_filename = 'tmp/videostreams/output.mp4'
            if self.name == 'x265':
                libname = 'libx265'
            elif self.name == 'x264':
                libname = 'libx264'
            else:
                print('Codec not supported')
                exit(1)
            # bgr24, rgb24, rgb?
            process = sp.Popen(shlex.split(f'/usr/bin/ffmpeg -y -s {width}x{height} -pixel_format bgr24 -f rawvideo -r {fps} -i pipe: -vcodec {libname} -pix_fmt yuv420p -crf 24 {output_filename}'), stdin=sp.PIPE)
            for img in raw_clip:
                process.stdin.write(np.array(img).tobytes())
            # Close and flush stdin
            process.stdin.close()
            # Wait for sub-process to finish
            process.wait()
            # Terminate the sub-process
            process.terminate()
            # check video size
            video_size = os.path.getsize(output_filename)*8
            # Use OpenCV to read video
            clip = []
            cap = cv2.VideoCapture(output_filename)
            # Check if camera opened successfully
            if (cap.isOpened()== False):
                print("Error opening video stream or file")
            # Read until video is completed
            while(cap.isOpened()):
                # Capture frame-by-frame
                ret, img = cap.read()
                if ret != True:break
                clip.append(transform(img).cuda())
            # When everything done, release the video capture object
            cap.release()
            assert len(clip) == len(raw_clip), 'Clip size mismatch'
            # create cache
            cache['clip'] = clip
            cache['bpp_est'] = {}
            cache['img_loss'] = {}
            cache['bpp_act'] = {}
            cache['metrics'] = {}
            bpp = video_size*1.0/len(clip)/(height*width)
            for i in range(len(clip)):
                Y1_raw,Y1_com = transform(raw_clip[i]).cuda(),clip[i]
                cache['img_loss'][i] = torch.FloatTensor([0]).squeeze(0).cuda(0)
                cache['bpp_est'][i] = torch.FloatTensor([0]).cuda(0)
                cache['metrics'][i] = PSNR(Y1_raw, Y1_com)
                cache['bpp_act'][i] = torch.FloatTensor([bpp])
        cache['max_idx'] = frame_idx-1
    
    def loss(self, app_loss, pix_loss, bpp_loss, aux_loss):
        return app_loss + pix_loss + bpp_loss + aux_loss
        
def calc_metrics(Y1_raw, Y1_com, use_psnr):
    if use_psnr:
        metrics = PSNR(Y1_raw, Y1_com)
    else:
        metrics = MSSSIM(Y1_raw, Y1_com)
    return metrics
    
def calc_loss(Y1_raw, Y1_com, use_psnr):
    if use_psnr:
        loss = torch.mean(torch.pow(Y1_raw - Y1_com, 2))*1024
    else:
        metrics = MSSSIM(Y1_raw, Y1_com)
        loss = 32*(1-metrics)
    return loss
        
def I_compression(Y1_raw, image_coder_name, _image_coder, use_psnr):
    # we can compress with bpg,deepcod ...
    if image_coder_name == 'deepcod':
        Y1_com,bits_act,bits_est,aux_loss = _image_coder(Y1_raw)
        # calculate bpp
        batch_size, _, Height, Width = Y1_com.shape
        bpp_est = bits_est/(Height * Width * batch_size)
        bpp_act = bits_act/(Height * Width * batch_size)
        # calculate metrics/loss
        metrics = calc_metrics(Y1_raw, Y1_com, use_psnr)
        loss = calc_loss(Y1_raw, Y1_com, use_psnr)
        flow_loss = torch.FloatTensor([0]).squeeze(0).cuda(0)
    elif image_coder_name == 'bpg':
        prename = "tmp/frames/prebpg"
        binname = "tmp/frames/bpg"
        postname = "tmp/frames/postbpg"
        raw_img = transforms.ToPILImage()(Y1_raw.squeeze(0))
        raw_img.save(prename + '.jpg')
        pre_bits = os.path.getsize(prename + '.jpg')*8
        os.system('bpgenc -f 444 -m 9 ' + prename + '.jpg -o ' + binname + '.bin -q 22')
        os.system('bpgdec ' + binname + '.bin -o ' + postname + '.jpg')
        post_bits = os.path.getsize(binname + '.bin')*8/(Height * Width * batch_size)
        bpp_act = torch.FloatTensor([post_bits])
        bpg_img = Image.open(postname + '.jpg').convert('RGB')
        Y1_com = transforms.ToTensor()(bpg_img).cuda().unsqueeze(0)
        metrics = calc_metrics(Y1_raw, Y1_com, use_psnr)
        loss = calc_loss(Y1_raw, Y1_com, use_psnr)
        bpp_est, aux_loss, flow_loss = torch.FloatTensor([0]).cuda(0), torch.FloatTensor([0]).squeeze(0).cuda(0), torch.FloatTensor([0]).squeeze(0).cuda(0)
    else:
        print('This image compression not implemented.')
        exit(0)
    return Y1_com, bpp_est, loss, aux_loss, flow_loss, bpp_act, metrics

def init_hidden(h,w,channels):
    rae_hidden = torch.zeros(1,channels*8,h//4,w//4).cuda()
    return torch.split(rae_hidden,channels*4,dim=1)
    
def PSNR(Y1_raw, Y1_com):
    train_mse = torch.mean(torch.pow(Y1_raw - Y1_com, 2))
    log10 = torch.log(torch.FloatTensor([10])).cuda()
    quality = 10.0*torch.log(1/train_mse)/log10
    return quality

def MSSSIM(Y1_raw, Y1_com):
    # pip install pytorch-msssim
    import pytorch_msssim
    return pytorch_msssim.ms_ssim(Y1_raw, Y1_com)

# pyramid flow estimation
class OpticalFlowNet(nn.Module):
    def __init__(self):
        super(OpticalFlowNet, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=0)
        self.loss = LossNet()

    def forward(self, im1_4, im2_4, batch, h, w):
        im1_3 = self.pool(im1_4)
        im1_2 = self.pool(im1_3)
        im1_1 = self.pool(im1_2)
        im1_0 = self.pool(im1_1)

        im2_3 = self.pool(im2_4)
        im2_2 = self.pool(im2_3)
        im2_1 = self.pool(im2_2)
        im2_0 = self.pool(im2_1)

        flow_zero = torch.zeros(batch, 2, h//16, w//16).cuda()

        loss_0, flow_0 = self.loss(flow_zero, im1_0, im2_0, upsample=False)
        loss_1, flow_1 = self.loss(flow_0, im1_1, im2_1, upsample=True)
        loss_2, flow_2 = self.loss(flow_1, im1_2, im2_2, upsample=True)
        loss_3, flow_3 = self.loss(flow_2, im1_3, im2_3, upsample=True)
        loss_4, flow_4 = self.loss(flow_3, im1_4, im2_4, upsample=True)

        return flow_4, loss_0, loss_1, loss_2, loss_3, loss_4

class LossNet(nn.Module):
    def __init__(self):
        super(LossNet, self).__init__()
        self.convnet = FlowCNN()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, flow, im1, im2, upsample=True):
        if upsample:
            flow = self.upsample(flow)
        batch_size, _, H, W = flow.shape
        loc = get_grid_locations(batch_size, H, W).type(im1.type())
        flow = flow.type(im1.type())
        im1_warped = F.grid_sample(im1, loc + flow.permute(0,2,3,1), align_corners=True)
        res = self.convnet(im1_warped, im2, flow)
        flow_fine = res + flow # N,2,H,W

        im1_warped_fine = F.grid_sample(im1, loc + flow_fine.permute(0,2,3,1), align_corners=True)
        loss_layer = torch.mean(torch.pow(im1_warped_fine-im2,2))

        return loss_layer, flow_fine

class FlowCNN(nn.Module):
    def __init__(self):
        super(FlowCNN, self).__init__()
        self.conv1 = nn.Conv2d(8, 32, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=7, stride=1, padding=3)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(16, 2, kernel_size=7, stride=1, padding=3)

    def forward(self, im1_warp, im2, flow):
        x = torch.cat((im1_warp, im2, flow),axis=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        return x

class ConvLSTM(nn.Module):
    def __init__(self, channels=128, forget_bias=1.0, activation=F.relu):
        super(ConvLSTM, self).__init__()
        self.conv = nn.Conv2d(2*channels, 4*channels, kernel_size=3, stride=1, padding=1)
        self._forget_bias = forget_bias
        self._activation = activation
        self._channels = channels

    def forward(self, x, state):
        c, h = torch.split(state,self._channels,dim=1)
        x = torch.cat((x, h), dim=1)
        y = self.conv(x)
        j, i, f, o = torch.split(y, self._channels, dim=1)
        f = torch.sigmoid(f + self._forget_bias)
        i = torch.sigmoid(i)
        c = c * f + i * self._activation(j)
        o = torch.sigmoid(o)
        h = o * self._activation(c)

        return h, torch.cat((c, h),dim=1)

class ComprNet(nn.Module):
    def __init__(self, device, data_name, codec_name, in_channels=2, channels=128, kernel1=3, padding1=1, kernel2=4, padding2=1):
        super(ComprNet, self).__init__()
        self.enc_conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        self.enc_conv4 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.gdn1 = GDN(channels)
        self.gdn2 = GDN(channels)
        self.gdn3 = GDN(channels)
        self.dec_conv1 = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)
        self.dec_conv4 = nn.ConvTranspose2d(channels, in_channels, kernel_size=4, stride=2, padding=1)
        self.igdn1 = GDN(channels, inverse=True)
        self.igdn2 = GDN(channels, inverse=True)
        self.igdn3 = GDN(channels, inverse=True)
        self.bottleneck_type = 'BASE'
        if codec_name in ['MRLVC-RPM', 'RLVC']:
            self.bottleneck_type = 'RPM'
        elif codec_name in ['MRLVC-RHP']:
            self.bottleneck_type = 'RHP'
        self.entropy_bottleneck = EntropyBottleneck2(channels,data_name,self.bottleneck_type)
        self.channels = channels
        self.use_RAE = (codec_name in ['MRLVC-BASE', 'MRLVC-RPM', 'MRLVC-RHP', 'RLVC'])
        if self.use_RAE:
            self.enc_lstm = ConvLSTM(channels)
            self.dec_lstm = ConvLSTM(channels)
        
    def forward(self, x, hidden, rpm_hidden, RPM_flag):
        state_enc, state_dec = torch.split(hidden.to(x.device),self.channels*2,dim=1)
        # compress
        x = self.gdn1(self.enc_conv1(x))
        x = self.gdn2(self.enc_conv2(x))
        if self.use_RAE:
            x, state_enc = self.enc_lstm(x, state_enc)
        x = self.gdn3(self.enc_conv3(x))
        latent = self.enc_conv4(x) # latent optical flow

        # quantization + entropy coding
        latent_hat, likelihoods, rpm_hidden = self.entropy_bottleneck(latent, rpm_hidden, RPM_flag, training=self.training)
        
        # calculate bpp (estimated)
        log2 = torch.log(torch.FloatTensor([2])).to(likelihoods.device)
        bits_est = torch.sum(torch.log(likelihoods)) / (-log2)
        
        # calculate bpp (actual)
        if self.bottleneck_type == 'RPM':
            bits_act = bits_est
        else:
            bits_act = self.entropy_bottleneck.get_actual_bits(latent, RPM_flag)

        # decompress
        x = self.igdn1(self.dec_conv1(latent_hat))
        x = self.igdn2(self.dec_conv2(x))
        if self.use_RAE:
            x, state_dec = self.enc_lstm(x, state_dec)
        x = self.igdn3(self.dec_conv3(x))
        hat = self.dec_conv4(x) # compressed optical flow (less accurate)
        
        # auxilary loss
        aux_loss = self.entropy_bottleneck.loss(RPM_flag)/self.channels
        
        if self.use_RAE:
            hidden = torch.cat((state_enc, state_dec),dim=1)
            
        #print("max: %.3f, min %.3f, act %.3f, est %.3f" % (torch.max(latent),torch.min(latent),bits_act,bits_est),latent.shape)
        return hat, hidden, rpm_hidden, bits_act, bits_est, aux_loss

class MCNet(nn.Module):
    def __init__(self):
        super(MCNet, self).__init__()
        self.l1 = nn.Conv2d(8, 64, kernel_size=3, stride=1, padding=1)
        self.l2 = ResidualBlock(64,64)
        self.l3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.l4 = ResidualBlock(64,64)
        self.l5 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.l6 = ResidualBlock(64,64)
        self.l7 = ResidualBlock(64,64)
        self.l8 = nn.Upsample(scale_factor=2, mode='nearest')
        self.l9 = ResidualBlock(64,64)
        self.l10 = nn.Upsample(scale_factor=2, mode='nearest')
        self.l11 = ResidualBlock(64,64)
        self.l12 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.l13 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        m1 = self.l1(x)
        m2 = self.l2(m1)
        m3 = self.l3(m2)
        m4 = self.l4(m3)
        m5 = self.l5(m4)
        m6 = self.l6(m5)
        m7 = self.l7(m6)
        m8 = self.l8(m7) + m4
        m9 = self.l9(m8)
        m10 = self.l10(m9) + m2
        m11 = self.l11(m10)
        m12 = F.relu(self.l12(m11))
        m13 = self.l13(m12)
        return m13

def get_grid_locations(b, h, w):
    new_h = torch.linspace(-1,1,h).view(-1,1).repeat(1,w)
    new_w = torch.linspace(-1,1,w).repeat(h,1)
    grid  = torch.cat((new_w.unsqueeze(2),new_h.unsqueeze(2)),dim=2)
    grid  = grid.unsqueeze(0)
    grid = grid.repeat(b,1,1,1)
    return grid

if __name__ == '__main__':
    # Adam, lr=1e-4,1e-6
    Y0_com = torch.randn(1,3,224,224).cuda(0)
    Y1_raw = torch.randn(1,3,224,224).cuda(0)
    Y2_raw = torch.randn(1,3,224,224).cuda(0)
    # init hidden states
    h = w = 224
    rae_hidden = torch.zeros(1,128*8,h//4,w//4).cuda(0)
    rpm_hidden = torch.zeros(1,128*4,h//16,w//16).cuda(0)
    model_codec = MRLVC(image_coder='deepcod')
    model_codec.split()
    latent = None
    # Y1_com, rae_hidden, rpm_hidden, latent = \
    #     model_codec(Y0_com, Y1_raw, rae_hidden, rpm_hidden, latent, False, False)
    while True:
        Y1_com, _, _, latent = \
            model_codec(Y0_com, Y1_raw, rae_hidden, rpm_hidden, None, False, False)
        print(Y0_com.shape)
    # encode I frames with image compression
    # encode I+1(P) frames and I-1(P) frames with the bottleneck
    # we can test with Y0_com set to Y0_raw,
    # later we can use DeepCOD to compress Y0_raw