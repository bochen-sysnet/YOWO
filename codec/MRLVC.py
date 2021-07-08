from __future__ import print_function
import os
import sys
import time
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Function
from compressai.entropy_models import EntropyBottleneck
sys.path.append('..')
import codec.arithmeticcoding as arithmeticcoding
from codec.deepcod import DeepCOD

# compress I frames with an image compression alg, e.g., DeepCOD, bpg, CA, none
# compress P frames wth RLVC
# no size estimation is performed on the first/last P frame
# loss can be psnr,ms-ssim

# Goal: convert images/videos that are compressed by RLVC
# GOP_size = args.f_P + args.b_P + 1
# Output: compressed images, predicted bits, actual bits
class MRLVC(nn.Module):
    def __init__(self, image_coder='deepcod'):
        super(MRLVC, self).__init__()
        device = torch.device('cuda')
        self.optical_flow = OpticalFlowNet()
        self.mv_codec = MV_CODEC_NET(device)
        self.MC_network = MCNet()
        self.res_codec = RES_CODEC_NET(device)
        self.RPM_mv = RecProbModel()
        self.RPM_res = RecProbModel()
        self._image_coder = DeepCOD().cuda() if image_coder == 'deepcod' else None

    def forward(self, Y0_com, Y1_raw, prior_latent, hidden, RPM_flag, I_flag, use_psnr=True): 
        # Y0_com: compressed previous frame
        # Y1_raw: uncompressed current frame
        # RPM flag: whether the first P frame (0: yes, it is the first P frame)
        # exp1: encode all frames with I_flag on
        # exp2: encode all frames without RPM_flag on
        # exp3: encode I frames with I_flag, first P frames with RPM_flag off, 
        # other P frames with RPM_flag on
        # If is I frame, return image compression result of Y1_raw
        if I_flag:
            # we can compress with bpg,deepcod ...
            if self._image_coder is not None:
                Y1_com,bits_act,bits_est = self._image_coder(Y1_raw)
                # calculate bpp
                batch_size, _, Height, Width = Y1_com.shape
                bpp_est = bits_est/(Height * Width * batch_size)
                bpp_act = bits_act/(Height * Width * batch_size)
                # calculate metrics/loss
                if use_psnr:
                    metrics = PSNR(Y1_raw, Y1_com)
                    loss = 1024*torch.mean(torch.pow(Y1_raw - Y1_com, 2)) + bpp_est
                else:
                    metrics = MSSSIM(Y1_raw, Y1_com)
                    loss = 32*(1-metrics) + bpp_est
                return Y1_com, loss, bpp_est, bpp_act, metrics
            else:
                # no compression
                return Y1_raw, 0, 0, 0, 0
        # otherwise, it's P frame
        batch_size, _, Height, Width = Y0_com.shape
        # hidden states
        mv_hidden, res_hidden, hidden_rpm_mv, hidden_rpm_res = hidden
        # estimate optical flow
        mv_tensor, _, _, _, _, _ = self.optical_flow(Y0_com, Y1_raw, batch_size, Height, Width)
        # compress optical flow
        mv_hat,mv_latent_hat,mv_hidden,mv_bits,mv_bpp = self.mv_codec(mv_tensor, mv_hidden, RPM_flag)
        # motion compensation
        loc = get_grid_locations(batch_size, Height, Width)
        Y1_warp = F.grid_sample(Y0_com, loc + mv_hat.permute(0,2,3,1))
        MC_input = torch.cat((mv_hat, Y0_com, Y1_warp), axis=1)
        Y1_MC = self.MC_network(MC_input)
        # compress residual
        res = Y1_raw - Y1_MC
        res_hat,res_latent_hat,res_hidden,res_bits,res_bpp = self.res_codec(res, res_hidden, RPM_flag)
        # reconstruction
        Y1_com = torch.clip(res_hat + Y1_MC, min=0, max=1)
        if RPM_flag:
            # latent presentations
            prior_mv_latent, prior_res_latent = prior_latent
            # RPM 
            prob_latent_mv, hidden_rpm_mv = self.RPM_mv(prior_mv_latent, hidden_rpm_mv)
            prob_latent_res, hidden_rpm_res = self.RPM_res(prior_res_latent, hidden_rpm_res)
            # estimate bpp
            bits_est_mv, sigma_mv, mu_mv = bits_estimation(mv_latent_hat, prob_latent_mv)
            bits_est_res, sigma_res, mu_res = bits_estimation(res_latent_hat, prob_latent_res)
            bpp_est = (bits_est_mv + bits_est_res)/(Height * Width * batch_size)
            # actual bits
            bits_act_mv = entropy_coding('mv', 'tmp', mv_latent_hat.detach().numpy(), sigma_mv.detach().numpy(), mu_mv.detach().numpy())
            bits_act_res = entropy_coding('res', 'tmp', res_latent_hat.detach().numpy(), sigma_res.detach().numpy(), mu_res.detach().numpy())
            bpp_act = (bits_act_mv + bits_act_res)/(Height * Width * batch_size)
        else:
            bpp_est = (mv_bpp + res_bpp)/(Height * Width * batch_size)
            bpp_act = (mv_bits + res_bits)/(Height * Width * batch_size)
        # hidden states
        hidden = (mv_hidden, res_hidden, hidden_rpm_mv, hidden_rpm_res)
        # latent
        prior_latent = (mv_latent_hat, res_latent_hat)
        # calculate metrics/loss
        if use_psnr:
            metrics = PSNR(Y1_raw, Y1_com)
            loss = 1024*torch.mean(torch.pow(Y1_raw - Y1_com, 2)) + bpp_est
        else:
            metrics = MSSSIM(Y1_raw, Y1_com)
            loss = 32*(1-metrics) + bpp_est
        return Y1_com, loss, hidden, prior_latent, bpp_est, bpp_act, metrics

def PSNR(Y1_raw, Y1_com):
    train_mse = torch.mean(torch.pow(Y1_raw - Y1_com, 2))
    log10 = torch.log(torch.FloatTensor([10])).cuda()
    quality = 10.0*torch.log(1.0/train_mse)/log10
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
        c_state, h_state = hidden
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x, (c_state, h_state) = self.lstm(x, (c_state, h_state))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        return x, (c_state, h_state)

def bits_estimation(x_target, sigma_mu, num_filters=128, tiny=1e-10):

    sigma, mu = torch.split(sigma_mu, num_filters, dim=1)

    half = torch.FloatTensor([0.5])

    upper = x_target + half
    lower = x_target - half

    sig = torch.maximum(sigma, torch.FloatTensor([-7.0]))
    upper_l = torch.sigmoid((upper - mu) * (torch.exp(-sig) + tiny))
    lower_l = torch.sigmoid((lower - mu) * (torch.exp(-sig) + tiny))
    p_element = upper_l - lower_l
    p_element = torch.clip(p_element, min=tiny, max=1 - tiny)

    ent = -torch.log(p_element) / torch.log(torch.FloatTensor([2]))
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

        flow_zero = torch.zeros(batch, 2, h//16, w//16)

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
        c, h = state
        x = torch.cat((x, h), axis=1)
        y = self.conv(x)
        j, i, f, o = torch.split(y, self._channels, dim=1)
        f = torch.sigmoid(f + self._forget_bias)
        i = torch.sigmoid(i)
        c = c * f + i * self._activation(j)
        o = torch.sigmoid(o)
        h = o * self._activation(c)

        return h, (c, h)


class MV_CODEC_NET(nn.Module):
    def __init__(self, device, channels=128):
        super(MV_CODEC_NET, self).__init__()
        self.enc_conv1 = nn.Conv2d(2, channels, kernel_size=3, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        self.enc_conv4 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.gdn = GDN(channels, device)
        self.enc_lstm = ConvLSTM(channels)
        self.dec_conv1 = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)
        self.dec_conv4 = nn.ConvTranspose2d(channels, 2, kernel_size=4, stride=2, padding=1)
        self.igdn = GDN(channels, device, inverse=True)
        self.dec_lstm = ConvLSTM(channels)
        self.entropy_bottleneck = EntropyBottleneck(128)
        self.entropy_bottleneck.update()

    def forward(self, x, hidden, RPM_flag):
        c_state_enc, h_state_enc, c_state_dec, h_state_dec = hidden
        x = self.gdn(self.enc_conv1(x))
        x = self.gdn(self.enc_conv2(x))
        x, (c_state_enc, h_state_enc) = self.enc_lstm(x, (c_state_enc, h_state_enc))
        x = self.gdn(self.enc_conv3(x))
        latent = self.enc_conv4(x) # latent optical flow

        # quantization + entropy coding
        # _,C,H,W = latent.shape
        string = self.entropy_bottleneck.compress(latent)
        latent_decom, likelihoods = self.entropy_bottleneck(latent, training=self.training)
        # latent_decom = self.entropy_bottleneck.decompress(string, (C, H, W))
        latent_hat = torch.round(latent) if RPM_flag else latent_decom

        # decompress
        x = self.igdn(self.dec_conv1(latent_hat))
        x = self.igdn(self.dec_conv2(x))
        x, (c_state_dec, h_state_dec) = self.enc_lstm(x, (c_state_dec, h_state_dec))
        x = self.igdn(self.dec_conv3(x))
        hat = self.dec_conv4(x) # compressed optical flow (less accurate)

        # calculate bpp (estimated)
        log2 = torch.log(torch.FloatTensor([2])).cuda()
        bits_est = torch.sum(torch.log(likelihoods)) / (-log2)

        hidden = (c_state_enc, h_state_enc, c_state_dec, h_state_dec)
        return hat, latent_hat, hidden, len(b''.join(string))*8, bits_est

class RES_CODEC_NET(nn.Module):
    def __init__(self, device, channels=128):
        super(RES_CODEC_NET, self).__init__()
        self.enc_conv1 = nn.Conv2d(3, channels, kernel_size=5, stride=2, padding=2)
        self.enc_conv2 = nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2)
        self.enc_conv3 = nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2)
        self.enc_conv4 = nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2, bias=False)
        self.gdn = GDN(channels, device)
        self.dec_conv1 = nn.ConvTranspose2d(channels, channels, kernel_size=6, stride=2, padding=2)
        self.dec_conv2 = nn.ConvTranspose2d(channels, channels, kernel_size=6, stride=2, padding=2)
        self.dec_conv3 = nn.ConvTranspose2d(channels, channels, kernel_size=6, stride=2, padding=2)
        self.dec_conv4 = nn.ConvTranspose2d(channels, 3, kernel_size=6, stride=2, padding=2)
        self.igdn = GDN(channels, device, inverse=True)
        self.entropy_bottleneck = EntropyBottleneck(128)
        self.entropy_bottleneck.update()

    def forward(self, x, hidden, RPM_flag):
        c_state_enc, h_state_enc, c_state_dec, h_state_dec = hidden
        # compress
        x = self.gdn(self.enc_conv1(x))
        x = self.gdn(self.enc_conv2(x))
        x = self.gdn(self.enc_conv3(x))
        latent = self.enc_conv4(x) # latent optical flow

        # quantization + entropy coding
        _,C,H,W = latent.shape
        string = self.entropy_bottleneck.compress(latent)
        latent_decom, likelihoods = self.entropy_bottleneck(latent, training=self.training)
        # latent_decom = self.entropy_bottleneck.decompress(string, (C, H, W))
        latent_hat = torch.round(latent) if RPM_flag else latent_decom

        # decompress
        x = self.igdn(self.dec_conv1(latent_hat))
        x = self.igdn(self.dec_conv2(x))
        x = self.igdn(self.dec_conv3(x))
        hat = self.dec_conv4(x) # compressed optical flow (less accurate)

        # calculate bpp (estimated)
        log2 = torch.log(torch.FloatTensor([2])).cuda()
        bits_est = torch.sum(torch.log(likelihoods)) / (-log2)

        hidden = (c_state_enc, h_state_enc, c_state_dec, h_state_dec)
        return hat, latent_hat, hidden, len(b''.join(string))*8, bits_est

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
    x_range = torch.linspace(-1, 1, w)
    y_grid, x_grid = torch.meshgrid(y_range, x_range)
    loc = torch.cat((y_grid.unsqueeze(-1), x_grid.unsqueeze(-1)), -1)
    loc = loc.unsqueeze(0)
    loc = loc.repeat(b,1,1,1)
    return loc

if __name__ == '__main__':
    # Adam, lr=1e-4,1e-6
    Y0_com = torch.randn(1,3,416,240)
    Y1_raw = torch.randn(1,3,416,240)
    Y2_raw = torch.randn(1,3,416,240)
    f_p = b_p = 6
    assert(f_p>=2 and b_p>=2)
    GOP_size = f_p + b_p + 1
    total_frame = 100
    GOP_num = int(np.floor((total_frame - 1)/GOP_size))
    # init hidden states
    mv_hidden = torch.split(torch.zeros(4,128,104,60),1)
    res_hidden = torch.split(torch.zeros(4,128,104,60),1) 
    hidden_rpm_mv = torch.split(torch.zeros(2,128,26,15),1) 
    hidden_rpm_res = torch.split(torch.zeros(2,128,26,15),1)
    hidden = (mv_hidden, res_hidden, hidden_rpm_mv, hidden_rpm_res)
    model = MRLVC()
    latent = None
    Y1_com, hidden, latent, bpp_est, bpp_act, metrics, loss = \
        model(Y0_com, Y1_raw, latent, hidden, False)
    Y2_com, hidden, latent, bpp_est, bpp_act, metrics, loss = \
        model(Y1_com, Y2_raw, latent, hidden, True)
    # encode I frames with image compression
    # encode I+1(P) frames and I-1(P) frames with the bottleneck
    # we can test with Y0_com set to Y0_raw,
    # later we can use DeepCOD to compress Y0_raw