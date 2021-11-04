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
from compressai.layers import GDN,ResidualBlock
from codec.entropy_models import RecProbModel,JointAutoregressiveHierarchicalPriors
from datasets.clip import *

# DVC,RLVC,MLVC
class LearnedVideoCodecs(nn.Module):
    def __init__(self, name, channels=128):
        super(LearnedVideoCodecs, self).__init__()
        self.name = name 
        device = torch.device('cuda')
        self.optical_flow = OpticalFlowNet()
        self.MC_network = MCNet()
        if name in ['MLVC','RLVC','DVC']:
            self.image_coder_name = 'bpg' 
        elif 'RAW' in name:
            self.image_coder_name = 'raw'
        else:
            print('I frame compression not implemented:',name)
            exit(1)
        print('I-frame compression:',self.image_coder_name)
        if self.image_coder_name == 'deepcod':
            self._image_coder = DeepCOD()
        else:
            self._image_coder = None
        self.mv_codec = ComprNet(device, self.name, in_channels=2, channels=channels, kernel1=3, padding1=1, kernel2=4, padding2=1)
        self.res_codec = ComprNet(device, self.name, in_channels=3, channels=channels, kernel1=5, padding1=2, kernel2=6, padding2=2)
        self.channels = channels
        self.gamma_img, self.gamma_bpp, self.gamma_flow, self.gamma_aux, self.gamma_app, self.gamma_rec, self.gamma_warp, self.gamma_mc = 1,1,1,1,1,1,1,1
        self.r = 1024 # PSNR:[256,512,1024,2048] MSSSIM:[8,16,32,64]
        self.epoch = -1
        
        # split on multi-gpus
        self.split()

    def split(self):
        if self._image_coder is not None:
            self._image_coder.cuda(0)
        self.optical_flow.cuda(0)
        self.mv_codec.cuda(0)
        self.MC_network.cuda(1)
        self.res_codec.cuda(1)
        
    def update_training(self, epoch):
        # warmup with all gamma set to 1
        # optimize for bpp,img loss and focus only reconstruction loss
        # optimize bpp and app loss only
        
        # setup training weights
        if epoch <= -1:
            self.gamma_img, self.gamma_bpp, self.gamma_flow, self.gamma_aux, self.gamma_app, self.gamma_rec, self.gamma_warp, self.gamma_mc = 1,1,1,1,1,1,1,1
        else:
            self.gamma_img, self.gamma_bpp, self.gamma_flow, self.gamma_aux, self.gamma_app, self.gamma_rec, self.gamma_warp, self.gamma_mc = 1,1,0,.1,0,1,0,0
            
        # set up GOP
        # epoch >=1 means pretraining on I-frame compression
        GOP = 10 if epoch >= -1 else 1
        
        # whether to compute action detection
        doAD = True if self.gamma_app > 0 else False
        
        self.epoch = epoch
        
        return GOP, doAD

    def forward(self, Y0_com, Y1_raw, hidden_states, RPM_flag, use_psnr=True):
        # Y0_com: compressed previous frame, [1,c,h,w]
        # Y1_raw: uncompressed current frame
        batch_size, _, Height, Width = Y1_raw.shape
        if self.name == 'RAW':
            bpp_est = bpp_act = metrics = torch.FloatTensor([0]).cuda(0)
            aux_loss = flow_loss = img_loss = torch.FloatTensor([0]).squeeze(0).cuda(0)
            return Y1_raw, hidden_states, bpp_est, img_loss, aux_loss, flow_loss, bpp_act, metrics
        if Y0_com is None:
            Y1_com, bpp_est, img_loss, aux_loss, flow_loss, bpp_act, metrics = I_compression(Y1_raw,self.image_coder_name,self._image_coder, self.r,use_psnr)
            return Y1_com, hidden_states, bpp_est, img_loss, aux_loss, flow_loss, bpp_act, metrics
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
        warp_loss = calc_loss(Y1_raw, Y1_warp.to(Y1_raw.device), self.r, use_psnr)
        MC_input = torch.cat((mv_hat, Y0_com, Y1_warp), axis=1)
        Y1_MC = self.MC_network(MC_input.cuda(1))
        mc_loss = calc_loss(Y1_raw, Y1_MC.to(Y1_raw.device), self.r, use_psnr)
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
        rec_loss = calc_loss(Y1_raw, Y1_com.to(Y1_raw.device), self.r, use_psnr)
        img_loss = (self.gamma_rec*rec_loss + self.gamma_warp*warp_loss + self.gamma_mc*mc_loss)/(self.gamma_rec+self.gamma_warp+self.gamma_mc) 
        flow_loss = (l0+l1+l2+l3+l4)/5*1024
        # hidden states
        hidden_states = (rae_mv_hidden.detach(), rae_res_hidden.detach(), rpm_mv_hidden, rpm_res_hidden)
        return Y1_com.cuda(0), hidden_states, bpp_est, img_loss, aux_loss, flow_loss, bpp_act, metrics
        
    def update_cache(self, frame_idx, clip_duration, sampling_rate, cache, startNewClip, shape):
        # process the involving GOP
        # how to deal with backward P frames?
        # if process in order, some frames need later frames to compress
        if startNewClip:
            # create cache
            cache['bpp_est'] = {}
            cache['img_loss'] = {}
            cache['flow_loss'] = {}
            cache['aux'] = {}
            cache['bpp_act'] = {}
            cache['metrics'] = {}
            cache['hidden'] = None
            cache['max_processed_idx'] = -1
            # the first frame to be compressed in a video
            start_idx = (frame_idx - (clip_duration-1) * sampling_rate - 1)
            start_idx = max(0,start_idx)
            # search for the I frame in this GOP
            # then compress all frames based on that
            # GOP = 6+6+1 = 13
            # I frames: 0, 13, ...
            for i in range(start_idx,frame_idx):
                self._compress_GOP(i, cache)
        else:
            self._compress_GOP(frame_idx-1, cache)
            
    def _compress_GOP(self, i, cache, fP=6, bP=6):
        cache['max_idx'] = i
        if i<=cache['max_processed_idx']:
            return
        GOP = fP + bP + 1
        if i%GOP <= fP:
            # e.g.: i=4,left=0,right=6,mid=0
            mid = i//GOP*GOP
            left = max(mid,0)
            right = min(mid+6,len(cache['clip'])-1)
        else:
            # e.g.: i=8,left=7,right=19,mid=13
            # in this case the last frame is always I frame
            possible_I = (i//GOP+1)*GOP
            mid = min(possible_I,len(cache['clip'])-1)
            left = max(possible_I-6,0)
            right = min(mid+6,len(cache['clip'])-1)
        cache['max_processed_idx'] = right
        # process backward frames
        if mid > left:
            for i in range(mid,left-1,-1):
                prev = i+1 if i<mid else -1
                self._process_single_frame(i, prev, cache, i==mid-1, i<=mid-2)
            mid2 = mid+1
        else:
            mid2 = mid
        # process forward frames
        for i in range(mid2,right+1):
            prev = i-1 if i>mid else -1
            self._process_single_frame(i, prev, cache, i==mid+1, i>=mid+2)
        
            
    def _process_single_frame(self, i, prev, cache, P_flag, RPM_flag):
        # frame shape
        _,h,w = cache['clip'][0].shape
        # frames to be processed
        Y0_com = cache['clip'][prev].unsqueeze(0) if prev>=0 else None
        Y1_raw = cache['clip'][i].unsqueeze(0)
        # hidden variables
        if P_flag:
            rae_mv_hidden, rae_res_hidden = init_hidden(h,w,self.channels)
            rpm_mv_hidden, rpm_res_hidden = self.mv_codec.entropy_bottleneck.init_state(), self.res_codec.entropy_bottleneck.init_state()
            hidden = (rae_mv_hidden, rae_res_hidden, rpm_mv_hidden, rpm_res_hidden)
        else:
            hidden = cache['hidden']
        Y1_com,hidden,bpp_est,img_loss,aux_loss,flow_loss,bpp_act,metrics = self(Y0_com, Y1_raw, hidden, RPM_flag)
        cache['hidden'] = hidden
        cache['clip'][i] = Y1_com.detach().squeeze(0)
        cache['img_loss'][i] = img_loss
        cache['flow_loss'][i] = flow_loss
        cache['aux'][i] = aux_loss
        cache['bpp_est'][i] = bpp_est
        cache['metrics'][i] = metrics
        cache['bpp_act'][i] = bpp_act.cpu()
        # we can record PSNR wrt the distance to I-frame to show error propagation
    
    def loss(self, app_loss, pix_loss, bpp_loss, aux_loss, flow_loss):
        if self.name in ['MLVC','RAW']:
            return self.gamma_app*app_loss + self.gamma_img*pix_loss + self.gamma_bpp*bpp_loss + self.gamma_aux*aux_loss + self.gamma_flow*flow_loss
        elif self.name == 'RLVC' or self.name == 'DVC':
            return self.gamma_img*pix_loss + self.gamma_bpp*bpp_loss + self.gamma_aux*aux_loss + self.gamma_flow*flow_loss
        else:
            print('Loss not implemented')
            exit(1)
        
    def load_state_dict_whatever(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name.endswith("._offset") or name.endswith("._quantized_cdf") or name.endswith("._cdf_length"):
                 continue
            if name in own_state:
                own_state[name].copy_(param)
                
    def load_state_dict_all(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name.endswith("._offset") or name.endswith("._quantized_cdf") or name.endswith("._cdf_length") or name.endswith(".scale_table"):
                 continue
            own_state[name].copy_(param)
            
# DCVC?

class DCVC(nn.Module):
    def __init__(self, name, channels=64, channels2=96):
        super(DCVC, self).__init__()
        device = torch.device('cuda')
        self.ctx_encoder = nn.Sequential(nn.Conv2d(channels+3, channels, kernel_size=5, stride=2, padding=2),
                                        GDN(channels),
                                        ResidualBlock(channels,channels),
                                        nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2),
                                        GDN(channels),
                                        ResidualBlock(channels,channels),
                                        nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2),
                                        GDN(channels),
                                        nn.Conv2d(channels, channels2, kernel_size=5, stride=2, padding=2)
                                        )
        self.ctx_decoder = nn.ModuleList([nn.ConvTranspose2d(channels2, channels, kernel_size=3, stride=2, padding=1),
                                        GDN(channels, inverse=True),
                                        nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=2, padding=1),
                                        GDN(channels, inverse=True),
                                        ResidualBlock(channels,channels),
                                        nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=2, padding=1),
                                        GDN(channels, inverse=True),
                                        ResidualBlock(channels,channels),
                                        nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=2, padding=1),
                                        nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1),
                                        ResidualBlock(channels,channels),
                                        ResidualBlock(channels,channels),
                                        nn.Conv2d(channels, 3, kernel_size=3, stride=1, padding=1)
                                        ])
        self.feature_extract = nn.Sequential(nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1),
                                        ResidualBlock(channels,channels)
                                        )
        self.ctx_refine = nn.Sequential(ResidualBlock(channels,channels),
                                        nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
                                        )
        self.tmp_prior_encoder = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2),
                                        GDN(channels),
                                        nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2),
                                        GDN(channels),
                                        nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2),
                                        GDN(channels),
                                        nn.Conv2d(channels, channels2, kernel_size=5, stride=2, padding=2)
                                        )
        self.optical_flow = OpticalFlowNet()
        self.mv_codec = ComprNet(device, name, in_channels=2, channels=channels, kernel1=3, padding1=1, kernel2=4, padding2=1)
        self.entropy_bottleneck = JointAutoregressiveHierarchicalPriors(channels2)
        self.gamma_img, self.gamma_bpp, self.gamma_flow, self.gamma_aux, self.gamma_app, self.gamma_rec, self.gamma_warp = 1,1,1,1,1,1,1
        self.r = 1024
        self.split()

    def split(self):
        self.optical_flow.cuda(0)
        self.mv_codec.cuda(0)
        self.feature_extract.cuda(0)
        self.ctx_refine.cuda(0)
        self.tmp_prior_encoder.cuda(1)
        self.ctx_encoder.cuda(1)
        self.entropy_bottleneck.cuda(1)
        self.ctx_decoder.cuda(1)
    
    def forward(self, x, x_hat_prev, hidden_states, RPM_flag, use_psnr=True):
        # I-frame compression
        if x_hat_prev is None:
            x_hat, bpp_est, img_loss, aux_loss, flow_loss, bpp_act, metrics = I_compression(x,'bpg',None,self.r,use_psnr)
            return x_hat, hidden_states, bpp_est, img_loss, aux_loss, flow_loss, bpp_act, metrics
        # size
        bs,c,h,w = x.size()
        
        # hidden states
        rae_mv_hidden, rpm_mv_hidden = hidden_states
                
        # motion estimation
        mv, l0, l1, l2, l3, l4 = self.optical_flow(x, x_hat_prev, bs, h, w)
        
        # compress optical flow
        mv_hat,rae_mv_hidden,rpm_mv_hidden,mv_act,mv_est,mv_aux = self.mv_codec(mv, rae_mv_hidden, rpm_mv_hidden, RPM_flag)
        
        # feature extraction
        x_rhat_prev = self.feature_extract(x_hat_prev)
        
        # motion compensation
        loc = get_grid_locations(bs, h, w).type(x.type())
        x_warp = F.grid_sample(x_rhat_prev, loc + mv_hat.permute(0,2,3,1), align_corners=True)
        x_tilde = F.grid_sample(x_hat_prev, loc + mv_hat.permute(0,2,3,1), align_corners=True)
        warp_loss = calc_loss(x, x_tilde.to(x.device), self.r, use_psnr)
        
        # context refinement
        context = self.ctx_refine(x_warp)
        
        # temporal prior
        prior = self.tmp_prior_encoder(context.cuda(1))
        
        # contextual encoder
        y = self.ctx_encoder(torch.cat((x, context), axis=1).cuda(1))
        
        # entropy model
        self.entropy_bottleneck.update()
        y_hat, likelihoods = self.entropy_bottleneck(y, prior, training=self.training)
        y_est = self.entropy_bottleneck.get_estimate_bits(likelihoods)
        y_act = self.entropy_bottleneck.get_actual_bits(y)
        y_aux = self.entropy_bottleneck.loss()
        
        # contextual decoder
        x_hat = y_hat
        for i,m in enumerate(self.ctx_decoder):
            if i in [0,2,5,8]:
                if i==0:
                    sz = torch.Size([bs,c,h//8,w//8])
                elif i==2:
                    sz = torch.Size([bs,c,h//4,w//4])
                elif i==5:
                    sz = torch.Size([bs,c,h//2,w//2])
                else:
                    sz = torch.Size([bs,c,h,w])
                x_hat = m(x_hat,output_size=sz)
            elif i==9:
                x_hat = m(torch.cat((x_hat, context), axis=1))
            else:
                x_hat = m(x_hat)
        
        # estimated bits
        bpp_est = (mv_est + y_est.cuda(0))/(h * w * bs)
        # actual bits
        bpp_act = (mv_act + y_act.cuda(0))/(h * w * bs)
        # auxilary loss
        aux_loss = (mv_aux + y_aux.cuda(0))/2
        # calculate metrics/loss
        metrics = calc_metrics(x, x_hat.cuda(0), use_psnr)
        rec_loss = calc_loss(x, x_hat.cuda(0), use_psnr)
        img_loss = (self.gamma_rec*rec_loss + self.gamma_warp*warp_loss)/(self.gamma_rec+self.gamma_warp) 
        # flow loss
        flow_loss = (l0+l1+l2+l3+l4)/5*1024
        # hidden states
        hidden_states = (rae_mv_hidden.detach(), rpm_mv_hidden)
        return x_hat, hidden_states, bpp_est, img_loss, aux_loss, flow_loss, bpp_act, metrics
    
    def update_cache(self, frame_idx, clip_duration, sampling_rate, cache, startNewClip, shape):
        # process the involving GOP
        # how to deal with backward P frames?
        # if process in order, some frames need later frames to compress
        if startNewClip:
            # create cache
            cache['bpp_est'] = {}
            cache['img_loss'] = {}
            cache['flow_loss'] = {}
            cache['aux'] = {}
            cache['bpp_act'] = {}
            cache['metrics'] = {}
            cache['hidden'] = None
            cache['max_processed_idx'] = -1
            # the first frame to be compressed in a video
            start_idx = (frame_idx - (clip_duration-1) * sampling_rate - 1)
            start_idx = max(0,start_idx)
            # search for the I frame in this GOP
            # then compress all frames based on that
            # GOP = 6+6+1 = 13
            # I frames: 0, 13, ...
            for i in range(start_idx,frame_idx):
                self._compress_GOP(i, cache)
        else:
            self._compress_GOP(frame_idx-1, cache)
            
    def _compress_GOP(self, i, cache, fP=6, bP=6):
        cache['max_idx'] = i
        if i<=cache['max_processed_idx']:
            return
        GOP = fP + bP + 1
        if i%GOP <= fP:
            # e.g.: i=4,left=0,right=6,mid=0
            mid = i//GOP*GOP
            left = max(mid,0)
            right = min(mid+6,len(cache['clip'])-1)
        else:
            # e.g.: i=8,left=7,right=19,mid=13
            # in this case the last frame is always I frame
            possible_I = (i//GOP+1)*GOP
            mid = min(possible_I,len(cache['clip'])-1)
            left = max(possible_I-6,0)
            right = min(mid+6,len(cache['clip'])-1)
        cache['max_processed_idx'] = right
        # process backward frames
        if mid > left:
            for i in range(mid,left-1,-1):
                prev = i+1 if i<mid else -1
                self._process_single_frame(i, prev, cache, i==mid-1, i<=mid-2)
            mid2 = mid+1
        else:
            mid2 = mid
        # process forward frames
        for i in range(mid2,right+1):
            prev = i-1 if i>mid else -1
            self._process_single_frame(i, prev, cache, i==mid+1, i>=mid+2)
        
    def _process_single_frame(self, i, prev, cache, P_flag, RPM_flag):
        # frame shape
        _,h,w = cache['clip'][0].shape
        # frames to be processed
        Y0_com = cache['clip'][prev].unsqueeze(0) if prev>=0 else None
        Y1_raw = cache['clip'][i].unsqueeze(0)
        # hidden variables
        if P_flag:
            rae_mv_hidden, rae_res_hidden = init_hidden(h,w,self.channels)
            rpm_mv_hidden, rpm_res_hidden = self.mv_codec.entropy_bottleneck.init_state(), self.res_codec.entropy_bottleneck.init_state()
            hidden = (rae_mv_hidden, rae_res_hidden, rpm_mv_hidden, rpm_res_hidden)
        else:
            hidden = cache['hidden']
        Y1_com,hidden,bpp_est,img_loss,aux_loss,flow_loss,bpp_act,metrics = self(Y0_com, Y1_raw, hidden, RPM_flag)
        cache['hidden'] = hidden
        cache['clip'][i] = Y1_com.detach().squeeze(0)
        cache['img_loss'][i] = img_loss
        cache['flow_loss'][i] = flow_loss
        cache['aux'][i] = aux_loss
        cache['bpp_est'][i] = bpp_est
        cache['metrics'][i] = metrics
        cache['bpp_act'][i] = bpp_act.cpu()
        # we can record PSNR wrt the distance to I-frame to show error propagation
        
    def loss(self, app_loss, pix_loss, bpp_loss, aux_loss, flow_loss):
        return self.gamma_app*app_loss + self.gamma_img*pix_loss + self.gamma_bpp*bpp_loss + self.gamma_aux*aux_loss + self.gamma_flow*flow_loss
        
    def update_training(self, epoch):
        # warmup with all gamma set to 1
        # optimize for bpp,img loss and focus only reconstruction loss
        # optimize bpp and app loss only
        
        # setup training weights
        if epoch <= -1:
            self.gamma_img, self.gamma_bpp, self.gamma_flow, self.gamma_aux, self.gamma_app, self.gamma_rec, self.gamma_warp = 1,1,1,1,0,1,1
        else:
            self.gamma_img, self.gamma_bpp, self.gamma_flow, self.gamma_aux, self.gamma_app, self.gamma_rec, self.gamma_warp = 1,1,0,.1,0,1,0
            
        # set up GOP
        # epoch >=1 means pretraining on I-frame compression
        GOP = 10 if epoch >= -1 else 1
        
        # whether to compute action detection
        doAD = True if self.gamma_app > 0 else False
        
        self.epoch = epoch
        
        return GOP, doAD
        
    def load_state_dict_whatever(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name.endswith("._offset") or name.endswith("._quantized_cdf") or name.endswith("._cdf_length"):
                 continue
            if name in own_state:
                own_state[name].copy_(param)
                
    def load_state_dict_all(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name.endswith("._offset") or name.endswith("._quantized_cdf") or name.endswith("._cdf_length") or name.endswith(".scale_table"):
                 continue
            own_state[name].copy_(param)
        
class StandardVideoCodecs(nn.Module):
    def __init__(self, name):
        super(StandardVideoCodecs, self).__init__()
        self.name = name # x264, x265?
        self.placeholder = torch.nn.Parameter(torch.zeros(1))
        
    def update_cache(self, frame_idx, clip_duration, sampling_rate, cache, startNewClip, shape):
        if startNewClip:
            imgByteArr = io.BytesIO()
            width,height = shape
            fps = 25
            Q = 27#15,19,23,27
            GOP = 13
            output_filename = 'tmp/videostreams/output.mp4'
            if self.name == 'x265':
                cmd = f'/usr/bin/ffmpeg -y -s {width}x{height} -pixel_format bgr24 -f rawvideo -r {fps} -i pipe: -vcodec libx265 -pix_fmt yuv420p -preset veryfast -tune zerolatency -x265-params "crf={Q}:keyint={GOP}:verbose=1" {output_filename}'
            elif self.name == 'x264':
                cmd = f'/usr/bin/ffmpeg -y -s {width}x{height} -pixel_format bgr24 -f rawvideo -r {fps} -i pipe: -vcodec libx264 -pix_fmt yuv420p -preset veryfast -tune zerolatency -crf {Q} -g {GOP} -bf 2 -b_strategy 0 -sc_threshold 0 -loglevel debug {output_filename}'
            else:
                print('Codec not supported')
                exit(1)
            # bgr24, rgb24, rgb?
            #process = sp.Popen(shlex.split(f'/usr/bin/ffmpeg -y -s {width}x{height} -pixel_format bgr24 -f rawvideo -r {fps} -i pipe: -vcodec {libname} -pix_fmt yuv420p -crf 24 {output_filename}'), stdin=sp.PIPE)
            process = sp.Popen(shlex.split(cmd), stdin=sp.PIPE)
            raw_clip = cache['clip']
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
                clip.append(transforms.ToTensor()(img).cuda())
            # When everything done, release the video capture object
            cap.release()
            assert len(clip) == len(raw_clip), 'Clip size mismatch'
            # create cache
            cache['bpp_est'] = {}
            cache['img_loss'] = {}
            cache['bpp_act'] = {}
            cache['metrics'] = {}
            cache['flow_loss'] = {}
            cache['aux'] = {}
            bpp = video_size*1.0/len(clip)/(height*width)
            for i in range(len(clip)):
                Y1_raw = transforms.ToTensor()(raw_clip[i]).cuda()
                Y1_com = clip[i]
                cache['img_loss'][i] = torch.FloatTensor([0]).squeeze(0).cuda(0)
                cache['bpp_est'][i] = torch.FloatTensor([0]).cuda(0)
                cache['metrics'][i] = PSNR(Y1_raw, Y1_com)
                cache['bpp_act'][i] = torch.FloatTensor([bpp])
                cache['flow_loss'][i] = torch.FloatTensor([0]).cuda(0)
                cache['aux'][i] = torch.FloatTensor([0]).cuda(0)
            cache['clip'] = clip
        cache['max_idx'] = frame_idx-1
    
    def loss(self, app_loss, pix_loss, bpp_loss, aux_loss, flow_loss):
        return app_loss + pix_loss + bpp_loss + aux_loss + flow_loss
        
def calc_metrics(Y1_raw, Y1_com, use_psnr):
    if use_psnr:
        metrics = PSNR(Y1_raw, Y1_com)
    else:
        metrics = MSSSIM(Y1_raw, Y1_com)
    return metrics
    
def calc_loss(Y1_raw, Y1_com, r, use_psnr):
    if use_psnr:
        loss = torch.mean(torch.pow(Y1_raw - Y1_com, 2))*r
    else:
        metrics = MSSSIM(Y1_raw, Y1_com)
        loss = r*(1-metrics)
    return loss
        
def I_compression(Y1_raw, image_coder_name, _image_coder, r, use_psnr):
    # we can compress with bpg,deepcod ...
    batch_size, _, Height, Width = Y1_raw.shape
    if image_coder_name in ['deepcod']:
        Y1_com,bits_act,bits_est,aux_loss = _image_coder(Y1_raw)
        # calculate bpp
        bpp_est = bits_est/(Height * Width * batch_size)
        bpp_act = bits_act/(Height * Width * batch_size)
        # calculate metrics/loss
        metrics = calc_metrics(Y1_raw, Y1_com, use_psnr)
        loss = calc_loss(Y1_raw, Y1_com, r, use_psnr)
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
        bpp_act = torch.FloatTensor([post_bits]).squeeze(0)
        bpg_img = Image.open(postname + '.jpg').convert('RGB')
        Y1_com = transforms.ToTensor()(bpg_img).cuda().unsqueeze(0)
        metrics = calc_metrics(Y1_raw, Y1_com, use_psnr)
        #loss = calc_loss(Y1_raw, Y1_com, use_psnr)
        bpp_est = loss = aux_loss = flow_loss = torch.FloatTensor([0]).squeeze(0).cuda(0)
    else:
        print('This image compression not implemented.')
        exit(0)
    return Y1_com, bpp_est, loss, aux_loss, flow_loss, bpp_act, metrics

def init_hidden(h,w,channels):
    rae_hidden = torch.zeros(1,channels*8,h//4,w//4).cuda()
    return torch.split(rae_hidden,channels*4,dim=1)
    
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
        # im1_4,im2_4:[1,c,h,w]
        # flow_4:[1,2,h,w]
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
        
def init_state(self):
    return self.model_states
    
def get_actual_bits(self, x):
    string = self.compress(x)
    bits_act = torch.FloatTensor([len(b''.join(string))*8]).squeeze(0)
    return bits_act
        
def get_estimate_bits(self, likelihoods):
    log2 = torch.log(torch.FloatTensor([2])).squeeze(0).to(likelihoods.device)
    bits_est = torch.sum(torch.log(likelihoods)) / (-log2)
    return bits_est

class ComprNet(nn.Module):
    def __init__(self, device, codec_name, in_channels=2, channels=128, kernel1=3, padding1=1, kernel2=4, padding2=1):
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
        if codec_name in ['MLVC', 'RLVC', 'SLVC','DCVC']:
            self.entropy_bottleneck = RecProbModel(channels)
            self.entropy_type = 'rec'
        elif codec_name in ['DVC','SLVC2']:
            from compressai.entropy_models import EntropyBottleneck
            EntropyBottleneck.model_states = []
            EntropyBottleneck.init_state = init_state
            EntropyBottleneck.get_actual_bits = get_actual_bits
            EntropyBottleneck.get_estimate_bits = get_estimate_bits
            self.entropy_bottleneck = EntropyBottleneck(channels)
            self.entropy_type = 'non-rec'
        else:
            print('Bottleneck not implemented for:',codec_name)
            exit(1)
        print('Bottleneck:',self.entropy_type)
        self.channels = channels
        self.encoder_type = 'rec' if codec_name in ['MLVC', 'RLVC'] else 'non-rec'
        if self.encoder_type == 'rec':
            self.enc_lstm = ConvLSTM(channels)
            self.dec_lstm = ConvLSTM(channels)
            
        # might need residual struct to avoid PE vanishing?
        
    def forward(self, x, hidden, rpm_hidden, RPM_flag):
        state_enc, state_dec = torch.split(hidden.to(x.device),self.channels*2,dim=1)
        # compress
        x = self.gdn1(self.enc_conv1(x))
        x = self.gdn2(self.enc_conv2(x))
        if self.encoder_type == 'rec':
            x, state_enc = self.enc_lstm(x, state_enc)
        x = self.gdn3(self.enc_conv3(x))
        latent = self.enc_conv4(x) # latent optical flow

        # update CDF
        self.entropy_bottleneck.update(force=True)
        
        # quantization + entropy coding
        if self.entropy_type == 'non-rec':
            latent_hat, likelihoods = self.entropy_bottleneck(latent, training=self.training)
        else:
            latent_hat, likelihoods, rpm_hidden = self.entropy_bottleneck(latent, rpm_hidden, RPM_flag, training=self.training)
        
        # calculate bpp (estimated)
        bits_est = self.entropy_bottleneck.get_estimate_bits(likelihoods)
        
        # calculate bpp (actual)
        if self.entropy_type == 'non-rec':
            bits_act = self.entropy_bottleneck.get_actual_bits(latent)
        else:
            bits_act = self.entropy_bottleneck.get_actual_bits(latent, RPM_flag)

        # decompress
        x = self.igdn1(self.dec_conv1(latent_hat))
        x = self.igdn2(self.dec_conv2(x))
        if self.encoder_type == 'rec':
            x, state_dec = self.enc_lstm(x, state_dec)
        x = self.igdn3(self.dec_conv3(x))
        hat = self.dec_conv4(x)
        
        # auxilary loss
        if self.entropy_type == 'rec':
            aux_loss = self.entropy_bottleneck.loss(RPM_flag)/self.channels
        else:
            aux_loss = self.entropy_bottleneck.loss()/self.channels
        
        if self.encoder_type == 'rec':
            hidden = torch.cat((state_enc, state_dec),dim=1)
            
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
    
class PositionalEncoding(nn.Module):
    # max_len: longest sequence length
    # d_model: dimension for positional encoding
    # this encoding is not integrated into the model itself
    # enhance the model’s input to inject the order of words.
    # need residual to avoid vanishing?

    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(2).unsqueeze(3)
        pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x, start):
        return x + self.pe[start:x.size(0)+start, :]
        
def test_PE():
    batch_size = 4
    d_model = 128
    h,w = 14,14
    PE = PositionalEncoding(d_model)
    x = torch.randn(batch_size,d_model,h,w)
    y = PE(x,1)
    print(x.size(),y.size())

def attention(q, k, v, d_model, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_model)
        
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output
        
class Attention(nn.Module):
    def __init__(self, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v):
        
        bs = q.size(0)
        
        # perform linear operation
        
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)
        
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_model, self.dropout)
        
        output = self.out(scores) # bs * sl * d_model
    
        return output
        
class AVGNet(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        
        self.d_model = d_model
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v):
        
        bs = q.size(0)
        
        # perform linear operation
        
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)
        
        # calculate attention using function we will define next
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(self.d_model)
        weights = torch.sum(scores,dim=-1)
        weights = F.softmax(weights,dim=-1).unsqueeze(1)
        
        # qkv:[B,SL,D]
        # weights:[B,SL]
        # out:[B,1,D]
        output = torch.matmul(weights, v)
        
        output = self.out(output.view(bs, self.d_model)) # bs * d_model
    
        return output
        
class KFNet(nn.Module):
    def __init__(self, channels=128, in_channels=3):
        super(KFNet, self).__init__()
        self.enc = nn.Sequential(nn.Conv2d(in_channels, channels, kernel_size=3, stride=2, padding=1),
                                GDN(channels),
                                nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
                                GDN(channels),
                                nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
                                GDN(channels),
                                nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=False)
                                )
        self.dec = nn.Sequential(nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1),
                                GDN(channels, inverse=True),
                                nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1),
                                GDN(channels, inverse=True),
                                nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1),
                                GDN(channels, inverse=True),
                                nn.ConvTranspose2d(channels, in_channels, kernel_size=4, stride=2, padding=1)
                                )
        self.s_attn = Attention(channels)
        self.t_avg = AVGNet(channels)
        self.channels = channels
        
    def forward(self, raw_frames):
        # input: sequence of frames=[B,3,H,W]
        # output: key frame=[1,C,H,W]
        B,_,H,W = raw_frames.size()
        
        # encode original frame to features [B,128,H//16,W//16], e.g., [B,128,14,14]
        features = self.enc(raw_frames)
        _,_,fH,fW = features.size()
        
        # spatial attention
        features = features.view(B,self.channels,-1).transpose(1,2).contiguous() # B,fH*fW,128
        features = self.s_attn(features,features,features) # B,fH*fW,128
        
        # temporal attention average
        features = features.transpose(0,1).contiguous() # fH*fW,B,128
        features = self.t_avg(features,features,features) # fH*fW,128
        features = features.permute(0,1).contiguous().view(1,self.channels,fH,fW)

        # decode features to original size [1,3,H,W]
        x_hat = self.dec(features)
        
        return x_hat
    
        
class SLVC(nn.Module):
    def __init__(self, name, channels=128):
        super(SLVC, self).__init__()
        self.name = name 
        device = torch.device('cuda')
        self.optical_flow = OpticalFlowNet()
        self.MC_network = MCNet()
        self.mv_codec = ComprNet(device, self.name, in_channels=2, channels=channels, kernel1=3, padding1=1, kernel2=4, padding2=1)
        self.res_codec = ComprNet(device, self.name, in_channels=3, channels=channels, kernel1=5, padding1=2, kernel2=6, padding2=2)
        self.kfnet = KFNet(channels)
        self.channels = channels
        self.gamma_img, self.gamma_bpp, self.gamma_flow, self.gamma_aux, self.gamma_app, self.r = 1,1,1,1,0,1024
        # split on multi-gpus
        self.split()

    def split(self):
        self.optical_flow.cuda(0)
        self.mv_codec.cuda(0)
        self.MC_network.cuda(1)
        self.res_codec.cuda(1)
        self.kfnet.cuda(0)
    def forward(self, raw_frames, hidden_states, use_psnr=True):
        # raw_frames=[B,C,H,W]: input sequence of frames
        # 1. BPG compress the first frame
        # 2. compress motion/residual of the rectified frame
        # 3. Use the rectified frame to compress a batch of frames
        batch_size, _, Height, Width = raw_frames.shape
        # hidden states
        rae_mv_hidden, rae_res_hidden, rpm_mv_hidden, rpm_res_hidden = hidden_states
        # key frame will be compressed by BPG
        # one option to compare is to just use the original I frame as key frame
        #I frame = raw_frames[0,:,:,:].unsqueeze(0)
        I_frame = self.kfnet(raw_frames)
        
        # get mv of key frame
        # compress mv of key frame
        # get compensated key frame
        # compress res of key frame
        # get compressed key frame, which will be used as the base for later compression
        #I_frame_hat, _, _, _, _, bpp_act, _ = I_compression(I_frame,'bpg',None, self.r,use_psnr)
        
        
        key_frames = I_frame.repeat(batch_size,1,1,1) # todo
        
        # use the derived key frame to compute optical flow
        mv_tensors, l0, l1, l2, l3, l4 = self.optical_flow(key_frames, raw_frames, batch_size, Height, Width)
        
        # compress optical flow
        # hidden should be initialized once for forward and once for backward
        # it is better to utilize redundancy among frames
        # TODO: another approach is to estimate based on a compressed vector extracted from mv_tensors (same to res_tensors)
        # we can apply spatial-temporal attention to the mv_tensor
        mv_hat_list = [];mv_act_list = [];mv_est_list = [];mv_aux_list = []
        for i in range(batch_size):
            mv_hat,rae_mv_hidden,rpm_mv_hidden,mv_act,mv_est,mv_aux = self.mv_codec(mv_tensors[i,:,:,:].unsqueeze(0), rae_mv_hidden, rpm_mv_hidden, i>0)
            mv_hat_list.append(mv_hat)
            mv_act_list.append(mv_act)
            mv_est_list.append(mv_est)
            mv_aux_list.append(mv_aux)
        mv_hat = torch.cat(mv_hat_list)
        
        # motion compensation
        loc = get_grid_locations(batch_size, Height, Width).type(key_frames.type())
        warped_frames = F.grid_sample(key_frames, loc + mv_hat.permute(0,2,3,1), align_corners=True)
        warp_loss = calc_loss(raw_frames, warped_frames.to(raw_frames.device), self.r, use_psnr)
        MC_input = torch.cat((mv_hat, key_frames, warped_frames), axis=1)
        MC_frames = self.MC_network(MC_input.cuda(1))
        mc_loss = calc_loss(raw_frames, MC_frames.to(raw_frames.device), self.r, use_psnr)
        
        # compress residual
        res_tensors = raw_frames.cuda(1) - MC_frames
        res_hat_list = [];res_act_list = [];res_est_list = [];res_aux_list = []
        for i in range(batch_size):
            res_hat,rae_res_hidden,rpm_res_hidden,res_act,res_est,res_aux = self.res_codec(res_tensors[i,:,:,:].unsqueeze(0), rae_res_hidden, rpm_res_hidden, i>0)
            res_hat_list.append(res_hat)
            res_act_list.append(res_act.cuda(0))
            res_est_list.append(res_est.cuda(0))
            res_aux_list.append(res_aux.cuda(0))
        res_hat = torch.cat(res_hat_list)
        
        # reconstruction
        com_frames = torch.clip(res_hat + MC_frames, min=0, max=1)
        ##### compute bits
        # estimated bits
        bpp_est = (torch.stack(mv_est_list,dim=0).mean(dim=0) + torch.stack(res_est_list,dim=0).mean(dim=0))/(Height * Width * batch_size)
        # actual bits
        bpp_act = (torch.stack(mv_act_list,dim=0).mean(dim=0) + torch.stack(res_act_list,dim=0).mean(dim=0))/(Height * Width * batch_size)
        # auxilary loss
        aux_loss = (torch.stack(mv_aux_list,dim=0).mean(dim=0) + torch.stack(res_aux_list,dim=0).mean(dim=0))/2
        # calculate metrics/loss
        metrics = calc_metrics(raw_frames, com_frames.to(raw_frames.device), use_psnr)
        rec_loss = calc_loss(raw_frames, com_frames.to(raw_frames.device), self.r, use_psnr)
        img_loss = (rec_loss + warp_loss + mc_loss)/3
        flow_loss = (l0+l1+l2+l3+l4)/5*1024
        # hidden states
        hidden_states = (rae_mv_hidden.detach(), rae_res_hidden.detach(), rpm_mv_hidden, rpm_res_hidden)
        return com_frames.cuda(0), hidden_states, bpp_est, img_loss, aux_loss, flow_loss, bpp_act, metrics
    
    def loss(self, app_loss, pix_loss, bpp_loss, aux_loss, flow_loss):
        return self.gamma_app*app_loss + self.gamma_img*pix_loss + self.gamma_bpp*bpp_loss + self.gamma_aux*aux_loss + self.gamma_flow*flow_loss
        
# use context based compression
class SLVC2(nn.Module):
    def __init__(self, name, channels=64, channels2=96):
        super(SLVC2, self).__init__()
        self.name = name 
        device = torch.device('cuda')
        self.ctx_encoder = nn.Sequential(nn.Conv2d(3+channels, channels, kernel_size=5, stride=2, padding=2),
                                        GDN(channels),
                                        ResidualBlock(channels,channels),
                                        nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2),
                                        GDN(channels),
                                        ResidualBlock(channels,channels),
                                        nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2),
                                        GDN(channels),
                                        nn.Conv2d(channels, channels2, kernel_size=5, stride=2, padding=2)
                                        )
        self.ctx_decoder = nn.ModuleList([nn.ConvTranspose2d(channels2, channels, kernel_size=3, stride=2, padding=1),
                                        GDN(channels, inverse=True),
                                        nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=2, padding=1),
                                        GDN(channels, inverse=True),
                                        ResidualBlock(channels,channels),
                                        nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=2, padding=1),
                                        GDN(channels, inverse=True),
                                        ResidualBlock(channels,channels),
                                        nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=2, padding=1),
                                        nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1),
                                        ResidualBlock(channels,channels),
                                        ResidualBlock(channels,channels),
                                        nn.Conv2d(channels, 3, kernel_size=3, stride=1, padding=1)
                                        ])
        self.feature_extract = nn.Sequential(nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1),
                                        ResidualBlock(channels,channels)
                                        )
        self.tmp_prior_encoder = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2),
                                        GDN(channels),
                                        nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2),
                                        GDN(channels),
                                        nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2),
                                        GDN(channels),
                                        nn.Conv2d(channels, channels2, kernel_size=5, stride=2, padding=2)
                                        )
        self.entropy_bottleneck = JointAutoregressiveHierarchicalPriors(channels2)
        self.ref_codec = ComprNet(device, name, in_channels=3, channels=channels, kernel1=3, padding1=1, kernel2=4, padding2=1)
        self.kfnet = KFNet(channels)
        self.channels = channels
        self.gamma_img, self.gamma_bpp, self.gamma_flow, self.gamma_aux, self.gamma_app, self.r = 1,1,1,1,0,1024
        # split on multi-gpus
        self.split()

    def split(self):
        self.kfnet.cuda(0)
        self.ref_codec.cuda(0)
        self.feature_extract.cuda(0)
        self.tmp_prior_encoder.cuda(0)
        self.ctx_encoder.cuda(0)
        self.entropy_bottleneck.cuda(0)
        self.ctx_decoder.cuda(0)
        
    def forward(self, x, hidden_states, use_psnr=True):
        # x=[B,C,H,W]: input sequence of frames
        bs, c, h, w = x.size()
        
        # hidden
        rae_ref_hidden,rpm_ref_hidden = hidden_states
        
        # extract ref frame, which is close to all frames in a sense
        ref_frame = self.kfnet(x)
        
        # compress ref frame, use cheng2020?
        ref_frame_hat,rae_ref_hidden,rpm_ref_hidden,ref_act,ref_est,ref_aux = self.ref_codec(ref_frame, rae_ref_hidden, rpm_ref_hidden, False)
        
        # extract context
        context = self.feature_extract(ref_frame)
        
        # repeat context to match the size of all frames
        context_rep = context.repeat(bs,1,1,1)
        
        # temporal prior
        prior = self.tmp_prior_encoder(context_rep)
        
        # contextual encoder
        y = self.ctx_encoder(torch.cat((x, context_rep), axis=1))
        
        # entropy model
        self.entropy_bottleneck.update()
        y_hat, likelihoods = self.entropy_bottleneck(y, prior, training=self.training)
        y_est = self.entropy_bottleneck.get_estimate_bits(likelihoods)
        y_act = self.entropy_bottleneck.get_actual_bits(y)
        y_aux = self.entropy_bottleneck.loss()
        
        # contextual decoder
        x_hat = y_hat
        for i,m in enumerate(self.ctx_decoder):
            if i in [0,2,5,8]:
                if i==0:
                    sz = torch.Size([bs,c,h//8,w//8])
                elif i==2:
                    sz = torch.Size([bs,c,h//4,w//4])
                elif i==5:
                    sz = torch.Size([bs,c,h//2,w//2])
                else:
                    sz = torch.Size([bs,c,h,w])
                x_hat = m(x_hat,output_size=sz)
            elif i==9:
                x_hat = m(torch.cat((x_hat, context_rep), axis=1))
            else:
                x_hat = m(x_hat)
        
        # estimated bits
        bpp_est = (ref_est + y_est)/(h * w * bs)
        # actual bits
        bpp_act = (ref_act + y_act)/(h * w * bs)
        # auxilary loss
        aux_loss = (ref_aux + y_aux)/2
        # calculate metrics/loss
        metrics = calc_metrics(x, x_hat, use_psnr)
        img_loss = calc_loss(x, x_hat, self.r, use_psnr)
        # flow loss
        flow_loss = torch.FloatTensor([0]).squeeze(0).cuda(0)
        # hidden
        hidden_states = (rae_ref_hidden,rpm_ref_hidden)
        return com_frames, hidden_states, bpp_est, img_loss, aux_loss, flow_loss, bpp_act, metrics
    
    def loss(self, app_loss, pix_loss, bpp_loss, aux_loss, flow_loss):
        return self.gamma_app*app_loss + self.gamma_img*pix_loss + self.gamma_bpp*bpp_loss + self.gamma_aux*aux_loss + self.gamma_flow*flow_loss
        
        
def test_SLVC(name = 'SLVC2'):
    batch_size = 4
    h = w = 224
    channels = 64
    x = torch.randn(batch_size,3,h,w).cuda()
    if name == 'SLVC':
        model = SLVC(name,channels)
    else:
        model = SLVC2(name,channels)
    import torch.optim as optim
    from tqdm import tqdm
    parameters = set(p for n, p in model.named_parameters())
    optimizer = optim.Adam(parameters, lr=1e-4)
    if name == 'SLVC':
        rae_mv_hidden, rae_res_hidden = init_hidden(h,w,channels)
        rpm_mv_hidden, rpm_res_hidden = model.mv_codec.entropy_bottleneck.init_state(), model.res_codec.entropy_bottleneck.init_state()
        hidden_states = (rae_mv_hidden, rae_res_hidden, rpm_mv_hidden, rpm_res_hidden)
    else:
        rae_ref_hidden,_ = init_hidden(h,w,channels)
        rpm_ref_hidden = model.ref_codec.entropy_bottleneck.init_state()
        hidden_states = (rae_ref_hidden,rpm_ref_hidden)
    train_iter = tqdm(range(0,10000))
    for i,_ in enumerate(train_iter):
        optimizer.zero_grad()
        
        com_frames, hidden_states, bpp_est, img_loss, aux_loss, flow_loss, bpp_act, metrics = model(x, hidden_states)
        
        loss = model.loss(0,img_loss,bpp_est,aux_loss,flow_loss)
        loss.backward()
        optimizer.step()
        
        train_iter.set_description(
            f"Batch: {i:4}. "
            f"loss: {float(loss):.2f}. "
            f"img_loss: {float(img_loss):.2f}. "
            f"bits_est: {float(bits_est):.2f}. "
            f"bits_act: {float(bits_act):.2f}. "
            f"aux_loss: {float(aux_loss):.2f}. "
            f"flow_loss: {float(flow_loss):.2f}. ")
            
def test_DCVC():
    batch_size = 1
    h = w = 224
    channels = 64
    x = torch.randn(batch_size,3,h,w).cuda()
    model = DCVC('DCVC').cuda()
    import torch.optim as optim
    from tqdm import tqdm
    parameters = set(p for n, p in model.named_parameters())
    optimizer = optim.Adam(parameters, lr=1e-4)
    rae_mv_hidden, _ = init_hidden(h,w,channels)
    rpm_mv_hidden = model.mv_codec.entropy_bottleneck.init_state()
    hidden_states = (rae_mv_hidden, rpm_mv_hidden)
    train_iter = tqdm(range(0,10000))
    x_hat_prev = x
    for i,_ in enumerate(train_iter):
        optimizer.zero_grad()
        
        x_hat, hidden_states, bpp_est, img_loss, aux_loss, flow_loss, bpp_act, metrics = model(x, x_hat_prev.detach(), hidden_states, i>=1)
        x_hat_prev = x_hat
        
        loss = model.loss(0,img_loss,bpp_est,aux_loss,flow_loss)
        loss.backward()
        optimizer.step()
        
        train_iter.set_description(
            f"Batch: {i:4}. "
            f"loss: {float(loss):.2f}. "
            f"img_loss: {float(img_loss):.2f}. "
            f"bpp_est: {float(bpp_est):.2f}. "
            f"bpp_act: {float(bpp_act):.2f}. "
            f"aux_loss: {float(aux_loss):.2f}. "
            f"flow_loss: {float(flow_loss):.2f}. "
            f"metrics: {float(metrics):.2f}. ")
        
if __name__ == '__main__':
    test_SLVC()