from __future__ import print_function
import os
import sys
import time
import math
import random

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Function
from compressai import EntropyBottleneck

# Goal: convert images/videos that are compressed by DVC
# Output: compressed images, predicted bits, actual bits
class DVC(nn.Module):
    def __init__(self):
        super(DVC, self).__init__()
        self.optical_flow = OpticalFlowNet()
        self.mv_codec = MV_CODEC_NET()
        self.MC_network = MCNet()
        self.res_codec = RES_CODEC_NET()

    def forward(self, Y0_com, Y1_raw):
        # Y0_com: compressed previous frame
        # Y1_raw: uncompressed current frame
        batch_size, _, Height, Width = Y0_com.shape
        # estimate optical flow
        flow_tensor, _, _, _, _, _ = self.optical_flow(Y0_com, Y1_raw, batch_size, Height, Width)
        # compress optical flow
        flow_hat,flow_likelihood,flow_bytes = self.mv_codec(flow_tensor)
        # motion compensation
        loc = get_grid_locations(batch_size, H, W)
        Y1_warp = F.grid_sample(Y0_com, loc + flow_hat.permute(0,2,3,1))
        MC_input = torch.cat((flow_hat, Y0_com, Y1_warp), axis=1)
        Y1_MC = self.MC_network(MC_input)
        # compress residual
        Res = Y1_raw - Y1_MC
        Res_hat,Res_likelihood,Res_bytes = self.res_codec(Res)
        # reconstruction
        Y1_com = torch.clip(Res_hat + Y1_MC, min=0, max=1)
        # calculate bpp
        bpp_MV = torch.sum(torch.log(flow_likelihood)) / (-torch.log(2) * Height * Width * batch_size)
        bpp_Res = torch.sum(torch.log(Res_likelihood)) / (-torch.log(2) * Height * Width * batch_size)

        return Y1_com,bpp_MV+bpp_Res,(flow_bytes+Res_bytes)*8

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

        flow_zero = torch.zeros(batch, 2, h//16, w//16)

        loss_0, flow_0 = self.loss(flow_zero, im1_0, im2_0, 0)
        loss_1, flow_1 = self.loss(flow_0, im1_1, im2_1, 1)
        loss_2, flow_2 = self.loss(flow_1, im1_2, im2_2, 2)
        loss_3, flow_3 = self.loss(flow_2, im1_3, im2_3, 3)
        loss_4, flow_4 = self.loss(flow_3, im1_4, im2_4, 4)

    return flow_4, loss_0, loss_1, loss_2, loss_3, loss_4

class LossNet(nn.Module):
    def __init__(self):
        super(LossNet, self).__init__()
        self.convnet = FlowCNN()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, flow_course, im1, im2):
        flow = self.upsample(flow_course)
        batch_size, _, H, W = flow.shape
        loc = get_grid_locations(batch_size, H, W)
        im1_warped = F.grid_sample(im1, loc + flow.permute(0,2,3,1))
        res = self.convnet(im1_warped, im2, flow)
        flow_fine = res + flow # N,2,H,W

        im1_warped_fine = F.grid_sample(im1, loc + flow_fine.permute(0,2,3,1))
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
        x = F.relu(self.conv5(x))
        return x

class MV_CODEC_NET(nn.Module):
    def __init__(self, channels=128):
        super(MV_CODEC_NET, self).__init__()
        device = torch.device('cuda')
        self.enc_conv1 = nn.Conv2d(2, channels, kernel_size=3, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        self.enc_conv4 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.gdn = GDN(channels, device)
        self.dec_conv1 = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)
        self.dec_conv4 = nn.ConvTranspose2d(channels, 2, kernel_size=4, stride=2, padding=1)
        self.igdn = GDN(channels, device, inverse=True)
        self.entropy_bottleneck = EntropyBottleneck(3)

    def forward(self, x):
        x = self.gdn(self.enc_conv1(x))
        x = self.gdn(self.enc_conv2(x))
        x = self.gdn(self.enc_conv3(x))
        latent = self.enc_conv4(x) # latent optical flow

        # quantization + entropy coding
        string = self.entropy_bottleneck.compress(latent)
        latent_hat, likelihoods = self.entropy_bottleneck(latent, training=self.training)

        # decompress
        x = self.igdn(self.dec_conv1(latent_hat))
        x = self.igdn(self.dec_conv2(x))
        x = self.igdn(self.dec_conv3(x))
        hat = self.dec_conv4(x) # compressed optical flow (less accurate)
        return hat, likelihoods, len(string.view(-1))

class MCNet(nn.Module):
    def __init__(self):
        super(MCNet, self).__init__()
        self.l1 = nn.Conv2d(8, 64, kernel_size=3, stride=1, padding=1)
        self.l2 = ResBlock(64,64)
        self.l3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.l4 = ResBlock(64,64)
        self.l5 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.l6 = ResBlock(64,64)
        self.l7 = ResBlock(64,64)
        self.l8 = nn.Upsample(scale_factor=2, mode='nearest')
        self.l9 = ResBlock(64,64)
        self.l10 = nn.Upsample(scale_factor=2, mode='nearest')
        self.l11 = ResBlock(64,64)
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

class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        skip = self.conv_skip(x)
        # batch norm?
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x + skip

class RES_CODEC_NET(nn.Module):
    def __init__(self):
        super(RES_CODEC_NET, self).__init__()
        device = torch.device('cuda')
        self.enc_conv1 = nn.Conv2d(3, channels, kernel_size=5, stride=2, padding=2)
        self.enc_conv2 = nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2)
        self.enc_conv3 = nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2)
        self.enc_conv4 = nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2, bias=False)
        self.gdn = GDN(channels, device)
        self.dec_conv1 = nn.ConvTranspose2d(channels, channels, kernel_size=5, stride=2, padding=2)
        self.dec_conv2 = nn.ConvTranspose2d(channels, channels, kernel_size=5, stride=2, padding=2)
        self.dec_conv3 = nn.ConvTranspose2d(channels, channels, kernel_size=5, stride=2, padding=2)
        self.dec_conv4 = nn.ConvTranspose2d(channels, 2, kernel_size=5, stride=2, padding=2)
        self.igdn = GDN(channels, device, inverse=True)
        self.entropy_bottleneck = EntropyBottleneck(3)

    def forward(self, x):
        # compress
        x = self.gdn(self.enc_conv1(x))
        x = self.gdn(self.enc_conv2(x))
        x = self.gdn(self.enc_conv3(x))
        latent = self.enc_conv4(x) # latent optical flow

        # quantization + entropy coding
        string = self.entropy_bottleneck.compress(latent)
        latent_hat, likelihoods = self.entropy_bottleneck(latent, training=self.training)

        # decompress
        x = self.igdn(self.dec_conv1(latent_hat))
        x = self.igdn(self.dec_conv2(x))
        x = self.igdn(self.dec_conv3(x))
        hat = self.dec_conv4(x) # compressed optical flow (less accurate)
        return hat, likelihoods, len(string.view(-1))

class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones(inputs.size())*bound
        b = b.to(inputs.device)
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)
  
    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]^2))
    """
  
    def __init__(self,
                 ch,
                 device,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=.1,
                 reparam_offset=2**-18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = torch.FloatTensor([reparam_offset])

        self.build(ch, torch.device(device))
  
    def build(self, ch, device):
        self.pedestal = self.reparam_offset**2
        self.beta_bound = (self.beta_min + self.reparam_offset**2)**.5
        self.gamma_bound = self.reparam_offset
  
        # Create beta param
        beta = torch.sqrt(torch.ones(ch)+self.pedestal)
        self.beta = nn.Parameter(beta.to(device))

        # Create gamma param
        eye = torch.eye(ch)
        g = self.gamma_init*eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)

        self.gamma = nn.Parameter(gamma.to(device))
        self.pedestal = self.pedestal.to(device)

    def forward(self, inputs):
        # Assert internal parameters to same device as input
        self.beta = self.beta.to(inputs.device)
        self.gamma = self.gamma.to(inputs.device)
        self.pedestal = self.pedestal.to(inputs.device)

        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size() 
            inputs = inputs.view(bs, ch, d*w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal 

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma**2 - self.pedestal
        gamma  = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs**2, gamma, beta)
        norm_ = torch.sqrt(norm_)
  
        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs

def get_grid_locations(b, h, w):
    y_range = torch.linspace(-1, 1, h)
    x_range = np.linspace(-1, 1, w)
    y_grid, x_grid = np.meshgrid(y_range, x_range)
    loc = torch.cat((y_grid.unsqueeze(-1), x_grid.unsqueeze(-1)), -1)
    loc = loc.unsqueeze(0)
    loc = loc.repeat(b,1,1,1)
    return loc