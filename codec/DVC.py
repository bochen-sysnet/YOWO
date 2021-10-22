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
from compressai.layers import GDN,ResidualBlock

class DVC(nn.Module):
    def __init__(self, channels=128):
        super(DVC, self).__init__()
        device = torch.device('cuda')
        self.optical_flow = OpticalFlowNet()
        self.MC_network = MCNet()
        self.mv_codec = ComprNet(device, 'mv', in_channels=2, channels=channels, kernel1=3, padding1=1, kernel2=4, padding2=1)
        self.res_codec = ComprNet(device, 'res', in_channels=3, channels=channels, kernel1=5, padding1=2, kernel2=6, padding2=2)
        self.channels = channels
        self.image_coder_name = 'bpg'

    def forward(self, Y0_com, Y1_raw, use_psnr=True):
        # Y0_com: compressed previous frame
        # Y1_raw: uncompressed current frame
        batch_size, _, Height, Width = Y1_raw.shape
        # deal with I frame
        if self.name == 'RAW':
            bpp_est = bpp_act = metrics = torch.FloatTensor([0]).cuda(0)
            aux_loss = flow_loss = img_loss = torch.FloatTensor([0]).squeeze(0).cuda(0)
            return Y1_raw, bpp_est, img_loss, aux_loss, flow_loss, bpp_act, metrics
        if Y0_com is None:
            Y1_com, bpp_est, img_loss, aux_loss, flow_loss, bpp_act, metrics = I_compression(Y1_raw,self.image_coder_name,use_psnr)
            return Y1_com, bpp_est, img_loss, aux_loss, flow_loss, bpp_act, metrics
        # otherwise, it's P frame
        # estimate optical flow
        mv_tensor, l0, l1, l2, l3, l4 = self.optical_flow(Y0_com, Y1_raw, batch_size, Height, Width)
        # compress optical flow
        mv_hat,mv_act,mv_est,mv_aux = self.mv_codec(mv_tensor)
        # motion compensation
        loc = get_grid_locations(batch_size, Height, Width).type(Y0_com.type())
        Y1_warp = F.grid_sample(Y0_com, loc + mv_hat.permute(0,2,3,1), align_corners=True)
        warp_loss = calc_loss(Y1_raw, Y1_warp.to(Y1_raw.device), use_psnr)
        MC_input = torch.cat((mv_hat, Y0_com, Y1_warp), axis=1)
        Y1_MC = self.MC_network(MC_input.cuda(1))
        mc_loss = calc_loss(Y1_raw, Y1_MC.to(Y1_raw.device), use_psnr)
        # compress residual
        res_tensor = Y1_raw.cuda(1) - Y1_MC
        res_hat,res_act,res_est,res_aux = self.res_codec(res_tensor)
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
        return Y1_com.cuda(0), bpp_est, img_loss, aux_loss, flow_loss, bpp_act, metrics
        
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
        
def I_compression(Y1_raw, image_coder_name, use_psnr):
    if image_coder_name == 'bpg':
        prename = "../tmp/frames/prebpg"
        binname = "../tmp/frames/bpg"
        postname = "../tmp/frames/postbpg"
        raw_img = transforms.ToPILImage()(Y1_raw.squeeze(0))
        raw_img.save(prename + '.jpg')
        pre_bits = os.path.getsize(prename + '.jpg')*8
        os.system('bpgenc -f 444 -m 9 ' + prename + '.jpg -o ' + binname + '.bin -q 22')
        os.system('bpgdec ' + binname + '.bin -o ' + postname + '.jpg')
        post_bits = os.path.getsize(binname + '.bin')*8/(Height * Width * batch_size)
        bpp_act = torch.FloatTensor([post_bits]).squeeze(0)
        bpg_img = Image.open(postname + '.jpg').convert('RGB')
        Y1_com = transforms.ToTensor()(bpg_img).cuda().unsqueeze(0)
        metrics = calc_metrics(Y1_raw, Y1_com, use_psnr)
        bpp_est = loss = aux_loss = flow_loss = torch.FloatTensor([0]).squeeze(0).cuda(0)
    else:
        print('This image compression not implemented.')
        exit(0)
    return Y1_com, bpp_est, loss, aux_loss, flow_loss, bpp_act, metrics
    
def PSNR(Y1_raw, Y1_com):
    train_mse = torch.mean(torch.pow(Y1_raw - Y1_com, 2))
    log10 = torch.log(torch.FloatTensor([10])).squeeze(0).cuda()
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
    def __init__(self, device, data_name, in_channels=2, channels=128, kernel1=3, padding1=1, kernel2=4, padding2=1):
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
        from compressai.entropy_models import EntropyBottleneck
        self.entropy_bottleneck = EntropyBottleneck(channels)
        self.channels = channels
        
    def forward(self, x):
        # compress
        x = self.gdn1(self.enc_conv1(x))
        x = self.gdn2(self.enc_conv2(x))
        x = self.gdn3(self.enc_conv3(x))
        latent = self.enc_conv4(x) # latent optical flow

        # quantization + entropy coding
        self.entropy_bottleneck.update(force=True)
        latent_hat, likelihoods = self.entropy_bottleneck(latent, training=self.training)
        
        # calculate bpp (estimated)
        log2 = torch.log(torch.FloatTensor([2])).squeeze(0).to(likelihoods.device)
        bits_est = torch.sum(torch.log(likelihoods)) / (-log2)
        
        # calculate bpp (actual)
        string = self.compress(latent_hat)
        bits_act = torch.FloatTensor([len(b''.join(string))*8]).squeeze(0)

        # decompress
        x = self.igdn1(self.dec_conv1(latent_hat))
        x = self.igdn2(self.dec_conv2(x))
        x = self.igdn3(self.dec_conv3(x))
        hat = self.dec_conv4(x)
        
        # auxilary loss
        aux_loss = self.entropy_bottleneck.loss()/self.channels
        return hat, bits_act, bits_est, aux_loss

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
    Y0_raw = torch.randn(1,3,224,224).cuda(0)
    Y1_raw = torch.randn(1,3,224,224).cuda(0)
    net = DVC()
    # compress the I frame
    Y0_com, bpp_est, img_loss, aux_loss, flow_loss, bpp_act, metrics = net(None, Y0_raw)
    # compress the P frame
    Y1_com, bpp_est, img_loss, aux_loss, flow_loss, bpp_act, metrics = net(Y0_com, Y1_raw)
    print(bpp_est, img_loss, aux_loss, flow_loss, bpp_act, metrics)