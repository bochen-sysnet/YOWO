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
from compressai.entropy_models import EntropyBottleneck
sys.path.append('..')
import codec.arithmeticcoding as arithmeticcoding
from codec.deepcod import DeepCOD
from compressai.layers import GDN,ResidualBlock
from datasets.clip import *

# compress I frames with an image compression alg, e.g., DeepCOD, bpg, CA, none
# compress P frames wth RLVC
# no size estimation is performed on the first/last P frame
# loss can be psnr,ms-ssim

# Goal: convert images/videos that are compressed by RLVC
# GOP_size = args.f_P + args.b_P + 1
# Output: compressed images, predicted bits, actual bits
class LearnedVideoCodecs(nn.Module):
    def __init__(self, name):
        super(LearnedVideoCodecs, self).__init__()
        self.name = name # MRLVC,RLVC,DVC
        device = torch.device('cuda')
        self.optical_flow = OpticalFlowNet()
        self.MC_network = MCNet()
        self.image_coder_name = 'deepcod' # or BPG or none
        self._image_coder = DeepCOD() if self.image_coder_name == 'deepcod' else None
        use_RNN = (self.name == 'MRLVC' or self.name == 'RLVC')
        # non rnn ignores other hiddens
        self.mv_codec = ComprNet(device, use_RNN=use_RNN, in_channels=2, channels=128, kernel1=3, padding1=1, kernel2=4, padding2=1)
        self.res_codec = ComprNet(device, use_RNN=use_RNN, in_channels=3, channels=128, kernel1=5, padding1=2, kernel2=6, padding2=2)
        if self.name == 'MRLVC':
            self.RPM_mv = RecProbModel()
            self.RPM_res = RecProbModel()
        
        # split on multi-gpus
        self.split()

    def split(self):
        if self._image_coder is not None:
            self._image_coder.cuda(0)
        self.optical_flow.cuda(0)
        self.mv_codec.cuda(0)
        self.MC_network.cuda(1)
        self.res_codec.cuda(1)
        if self.name == 'MRLVC':
            self.RPM_mv.cuda(1)
            self.RPM_res.cuda(1)

    def forward(self, Y0_com, Y1_raw, rae_hidden, rpm_hidden, prior_latent, \
                RPM_flag, use_psnr=True):
        # Y0_com: compressed previous frame
        # Y1_raw: uncompressed current frame
        # RPM flag: whether the first P frame (0: yes, it is the first P frame)
        # exp1: encode all frames with I_flag on
        # exp2: encode all frames without RPM_flag on
        # exp3: encode I frames with I_flag, first P frames with RPM_flag off, 
        # other P frames with RPM_flag on
        # If is I frame, return image compression result of Y1_raw
        batch_size, _, Height, Width = Y1_raw.shape
        if Y0_com is None:
            # we can compress with bpg,deepcod ...
            if self.image_coder_name == 'deepcod':
                Y1_com,bits_act,bits_est,aux_loss = self._image_coder(Y1_raw)
                # calculate bpp
                batch_size, _, Height, Width = Y1_com.shape
                bpp_est = bits_est/(Height * Width * batch_size)
                bpp_act = bits_act/(Height * Width * batch_size)
                # calculate metrics/loss
                if use_psnr:
                    metrics = PSNR(Y1_raw, Y1_com)
                    loss = torch.mean(torch.pow(Y1_raw - Y1_com, 2))*1024
                else:
                    metrics = MSSSIM(Y1_raw, Y1_com)
                    loss = 32*(1-metrics)
            elif self.image_coder_name == 'bpg':
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
                if use_psnr:
                    metrics = PSNR(Y1_raw, Y1_com)
                else:
                    metrics = MSSSIM(Y1_raw, Y1_com)
                bpp_est, loss, aux_loss = torch.FloatTensor([0]).cuda(0), torch.FloatTensor([0]).squeeze(0).cuda(0), torch.FloatTensor([0]).squeeze(0).cuda(0)
            else:
                Y1_com = Y1_raw
                bpp_act = torch.FloatTensor([24])
                metrics = torch.FloatTensor([0])
            return Y1_com, rae_hidden, rpm_hidden, prior_latent, bpp_est, loss, aux_loss, bpp_act, metrics
        # otherwise, it's P frame
        # hidden states
        mv_hidden, res_hidden = torch.split(rae_hidden,128*4,dim=1)
        hidden_rpm_mv, hidden_rpm_res = torch.split(rpm_hidden,128*2,dim=1)
        # estimate optical flow
        mv_tensor, _, _, _, _, _ = self.optical_flow(Y0_com, Y1_raw, batch_size, Height, Width)
        # compress optical flow
        mv_hat,mv_latent_hat,mv_hidden,mv_bits,mv_bpp,mv_aux = self.mv_codec(mv_tensor, mv_hidden, RPM_flag)
        # motion compensation
        loc = get_grid_locations(batch_size, Height, Width).type(Y0_com.type())
        Y1_warp = F.grid_sample(Y0_com, loc + mv_hat.permute(0,2,3,1), align_corners=True)
        MC_input = torch.cat((mv_hat, Y0_com, Y1_warp), axis=1)
        Y1_MC = self.MC_network(MC_input.cuda(1))
        # compress residual
        res = Y1_raw.cuda(1) - Y1_MC
        res_hat,res_latent_hat,res_hidden,res_bits,res_bpp,res_aux = self.res_codec(res, res_hidden.cuda(1), RPM_flag)
        # reconstruction
        Y1_com = torch.clip(res_hat + Y1_MC, min=0, max=1)
        ##### compute bits
        # estimated bits
        bpp_est = (mv_bpp + res_bpp.cuda(0))/(Height * Width * batch_size)
        # actual bits
        bpp_act = (mv_bits + res_bits)/(Height * Width * batch_size)
        # during training the bits calculated using entropy bottleneck will
        # replace the bits that used to do entropy encoding
        if self.name == 'MRLVC' and RPM_flag:
            # latent presentations
            prior_mv_latent, prior_res_latent = torch.split(prior_latent.cuda(1),128,dim=1)
            # RPM 
            prob_latent_mv, hidden_rpm_mv = self.RPM_mv(prior_mv_latent.cuda(1), hidden_rpm_mv.cuda(1))
            prob_latent_res, hidden_rpm_res = self.RPM_res(prior_res_latent.cuda(1), hidden_rpm_res.cuda(1))
            # estimate bpp
            bits_est_mv, sigma_mv, mu_mv = bits_estimation(mv_latent_hat, prob_latent_mv.cuda(0))
            bits_est_res, sigma_res, mu_res = bits_estimation(res_latent_hat.cuda(0), prob_latent_res.cuda(0))
            bpp_est = (bits_est_mv + bits_est_res)/(Height * Width * batch_size)
            bpp_est = bpp_est.unsqueeze(0)
            # actual bits
            # if not self.training:
            #    bits_act_mv = entropy_coding('mv', 'tmp/bitstreams', mv_latent_hat.detach().cpu().numpy(), sigma_mv.detach().cpu().numpy(), mu_mv.detach().cpu().numpy())
            #    bits_act_res = entropy_coding('res', 'tmp/bitstreams', res_latent_hat.detach().cpu().numpy(), sigma_res.detach().cpu().numpy(), mu_res.detach().cpu().numpy())
            #    bpp_act = (bits_act_mv + bits_act_res)/(Height * Width * batch_size)
            #    bpp_act = torch.FloatTensor([bpp_act])
            # hidden
            rpm_hidden = torch.cat((hidden_rpm_mv.cuda(0), hidden_rpm_res.cuda(0)),dim=1)
        # latent
        prior_latent = torch.cat((mv_latent_hat, res_latent_hat.cuda(0)),dim=1)
            
        # hidden states
        if self.name != 'DVC':
            rae_hidden = torch.cat((mv_hidden, res_hidden.cuda(0)),dim=1)
        # calculate metrics/loss
        if use_psnr:
            metrics = PSNR(Y1_raw, Y1_com.to(Y1_raw.device))
            loss = torch.mean(torch.pow(Y1_raw - Y1_com.to(Y1_raw.device), 2))*1024
        else:
            metrics = MSSSIM(Y1_raw, Y1_com.to(Y1_raw.device))
            loss = 32*(1-metrics)
        # auxilary loss
        aux_loss = mv_aux + res_aux
        return Y1_com.cuda(0), rae_hidden, rpm_hidden, prior_latent, bpp_est, loss, aux_loss, bpp_act, metrics
        
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
            cache['loss'] = {}
            cache['aux'] = {}
            cache['bpp_act'] = {}
            cache['metrics'] = {}
            # compress from the first frame of the first clip to the current frame
            Iframe_idx = (frame_idx - (clip_duration-1) * sampling_rate - 1)//GOP*GOP
            Iframe_idx = max(0,Iframe_idx)
            for i in range(Iframe_idx,frame_idx):
                self._process_single_frame(i, GOP, cache)
        else:
            self._process_single_frame(frame_idx-1, GOP, cache)
            
    def _process_single_frame(self, i, GOP, cache):
        # frame shape
        _,h,w = cache['clip'][0].shape
        # frames to be processed
        Y0_com = cache['clip'][i-1].unsqueeze(0) if i>0 else None
        Y1_raw = cache['clip'][i].unsqueeze(0)
        # hidden variables
        RPM_flag = False
        rae_hidden, rpm_hidden = init_hidden(h,w)
        latent = torch.zeros(1,8,4,4).cuda()
        if i%GOP == 0:
            Y0_com = None
        elif i%GOP > 1:
            rae_hidden, rpm_hidden, latent = cache['rae_hidden'], cache['rpm_hidden'], cache['latent']
            RPM_flag = True
        Y1_com, rae_hidden,rpm_hidden,latent,bpp_est,img_loss, aux_loss, bpp_act, metrics = self(Y0_com, Y1_raw, rae_hidden, rpm_hidden, latent, RPM_flag)
        cache['rae_hidden'] = rae_hidden.detach()
        cache['rpm_hidden'] = rpm_hidden.detach()
        cache['latent'] = latent.detach()
        cache['clip'][i] = Y1_com.detach().squeeze(0)
        cache['loss'][i] = img_loss
        cache['aux'][i] = aux_loss
        cache['bpp_est'][i] = bpp_est
        cache['metrics'][i] = metrics
        cache['bpp_act'][i] = bpp_act
        cache['max_idx'] = i
    
    def loss(self, app_loss, pix_loss, bpp_loss):
        if self.name == 'MRLVC':
            return app_loss + pix_loss + bpp_loss
        elif self.name == 'RLVC' or self.name == 'DVC':
            return pix_loss + bpp_loss
        else:
            print('Loss not implemented')
            exit(1)
        
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
            cache['loss'] = {}
            cache['bpp_act'] = {}
            cache['metrics'] = {}
            bpp = video_size*1.0/len(clip)/(height*width)
            for i in range(len(clip)):
                Y1_raw,Y1_com = transform(raw_clip[i]).cuda(),clip[i]
                cache['loss'][i] = torch.FloatTensor([0]).squeeze(0).cuda(0)
                cache['bpp_est'][i] = torch.FloatTensor([0]).cuda(0)
                cache['metrics'][i] = PSNR(Y1_raw, Y1_com)
                cache['bpp_act'][i] = torch.FloatTensor([bpp])
        cache['max_idx'] = frame_idx-1
    
    def loss(self, app_loss, pix_loss, bpp_loss):
        return app_loss + pix_loss + bpp_loss

def init_hidden(h,w):
    rae_hidden = torch.zeros(1,128*8,h//4,w//4).cuda()
    rpm_hidden = torch.zeros(1,128*4,h//16,w//16).cuda()
    return rae_hidden, rpm_hidden
    
def PSNR(Y1_raw, Y1_com):
    train_mse = torch.mean(torch.pow(Y1_raw - Y1_com, 2))
    log10 = torch.log(torch.FloatTensor([10])).cuda()
    quality = 10.0*torch.log(1/train_mse)/log10
    return quality

def MSSSIM(Y1_raw, Y1_com):
    # pip install pytorch-msssim
    import pytorch_msssim
    return pytorch_msssim.ms_ssim(Y1_raw, Y1_com)

# conditional probability
# predict y_t based on parameters computed from y_t-1
class RecProbModel(nn.Module):
    def __init__(self, channels=128, act=torch.tanh):
        super(RecProbModel, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.lstm = ConvLSTM(channels)
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(channels, 2*channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, hidden):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x, hidden = self.lstm(x, hidden)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        return x, hidden

def bits_estimation(x_target, sigma_mu, num_filters=128, tiny=1e-10):

    sigma, mu = torch.split(sigma_mu, num_filters, dim=1)

    half = torch.FloatTensor([0.5]).cuda()

    upper = x_target + half
    lower = x_target - half

    sig = torch.maximum(sigma, torch.FloatTensor([-7.0]).cuda())
    upper_l = torch.sigmoid((upper - mu) * (torch.exp(-sig) + tiny))
    lower_l = torch.sigmoid((lower - mu) * (torch.exp(-sig) + tiny))
    p_element = upper_l - lower_l
    p_element = torch.clip(p_element, min=tiny, max=1 - tiny)

    ent = -torch.log(p_element) / torch.log(torch.FloatTensor([2])).cuda()
    bits = torch.sum(ent)

    return bits, sigma, mu

def entropy_coding(lat, path_bin, latent, sigma, mu):

    if lat == 'mv':
        bias = 50
    else:
        bias = 100

    bin_name = lat + '.bin'
    bitout = arithmeticcoding.BitOutputStream(open(path_bin + bin_name, "wb"))
    enc = arithmeticcoding.ArithmeticEncoder(32, bitout)

    for h in range(latent.shape[1]):
        for w in range(latent.shape[2]):
            for ch in range(latent.shape[3]):
                mu_val = mu[0, h, w, ch] + bias
                sigma_val = sigma[0, h, w, ch]
                symbol = latent[0, h, w, ch] + bias

                freq = arithmeticcoding.logFrequencyTable_exp(mu_val, sigma_val, np.int(bias * 2 + 1))
                enc.write(freq, symbol)

    enc.finish()
    bitout.close()

    bits_value = os.path.getsize(path_bin + bin_name) * 8

    return bits_value

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
    def __init__(self, device, use_RNN=True, in_channels=2, channels=128, kernel1=3, padding1=1, kernel2=4, padding2=1):
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
        self.entropy_bottleneck = EntropyBottleneck(128)
        self.entropy_bottleneck.update()
        self.use_RNN = use_RNN
        if use_RNN:
            self.enc_lstm = ConvLSTM(channels)
            self.dec_lstm = ConvLSTM(channels)
        
    def forward(self, x, hidden, RPM_flag):
        state_enc, state_dec = torch.split(hidden,128*2,dim=1)
        # compress
        x = self.gdn1(self.enc_conv1(x))
        x = self.gdn2(self.enc_conv2(x))
        if self.use_RNN:
            x, state_enc = self.enc_lstm(x, state_enc)
        x = self.gdn3(self.enc_conv3(x))
        latent = self.enc_conv4(x) # latent optical flow

        # quantization + entropy coding
        string = self.entropy_bottleneck.compress(latent)
        bits_act = torch.FloatTensor([len(b''.join(string))*8])
        latent_decom, likelihoods = self.entropy_bottleneck(latent, training=self.training)
        latent_hat = torch.round(latent) if RPM_flag else latent_decom

        # decompress
        x = self.igdn1(self.dec_conv1(latent_hat))
        x = self.igdn2(self.dec_conv2(x))
        if self.use_RNN:
            x, state_dec = self.enc_lstm(x, state_dec)
        x = self.igdn3(self.dec_conv3(x))
        hat = self.dec_conv4(x) # compressed optical flow (less accurate)
        
        # calculate bpp (estimated)
        log2 = torch.log(torch.FloatTensor([2])).to(likelihoods.device)
        bits_est = torch.sum(torch.log(likelihoods)) / (-log2)
        
        # auxilary loss
        aux_loss = self.entropy_bottleneck.loss()
        
        if self.use_RNN:
            hidden = torch.cat((state_enc, state_dec),dim=1)
        return hat, latent_hat, hidden, bits_act, bits_est, aux_loss

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
    y_range = torch.linspace(-1, 1, h)
    x_range = torch.linspace(-1, 1, w)
    y_grid, x_grid = torch.meshgrid(y_range, x_range)
    loc = torch.cat((y_grid.unsqueeze(-1), x_grid.unsqueeze(-1)), -1)
    loc = loc.unsqueeze(0)
    loc = loc.repeat(b,1,1,1)
    return loc

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