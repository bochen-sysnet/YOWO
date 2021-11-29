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
from compressai.layers import GDN,ResidualBlock,AttentionBlock
from compressai.models import CompressionModel
from codec.entropy_models import RecProbModel,JointAutoregressiveHierarchicalPriors,MeanScaleHyperPriors
from compressai.models.waseda import Cheng2020Attention
import pytorch_msssim
from datasets.clip import *
from core.utils import *

def get_codec_model(name):
    if name in ['MLVC','RLVC','DVC','RAW']:
        model_codec = IterPredVideoCodecs(name)
    elif name in ['DCVC','DCVC_v2']:
        model_codec = DCVC(name)
    elif 'SPVC' in name:
        model_codec = SPVC(name)
    elif name in ['SCVC']:
        model_codec = SCVC(name)
    elif 'SVC' in name:
        model_codec = SVC(name)
    elif name in ['AE3D']:
        model_codec = AE3D(name)
    elif name in ['x264','x265']:
        model_codec = StandardVideoCodecs(name)
    else:
        print('Cannot recognize codec:', name)
        exit(1)
    return model_codec

def compress_video(model, frame_idx, cache, startNewClip):
    if model.name in ['MLVC','RLVC','DVC','DCVC','DCVC_v2']:
        compress_video_sequential(model, frame_idx, cache, startNewClip)
    elif model.name in ['x265','x264']:
        compress_video_group(model, frame_idx, cache, startNewClip)
    elif model.name in ['SCVC','AE3D'] or 'SVC' in model.name or 'SPVC' in model.name:
        compress_video_batch(model, frame_idx, cache, startNewClip)
            
def init_training_params(model):
    model.r_img, model.r_bpp, model.r_flow, model.r_aux = 1,1,1,1
    model.r_app, model.r_rec, model.r_warp, model.r_mc = 1,1,1,1
    
    model.r = 1024 # PSNR:[256,512,1024,2048] MSSSIM:[8,16,32,64]
    model.I_level = 27 # [37,32,27,22] poor->good quality
    
    model.fmt_enc_str = "FL:{0:.4f}\tMV:{1:.4f}\tMC:{2:.4f}\tRES:{3:.4f}"
    model.fmt_dec_str = "MV:{0:.4f}\tMC:{1:.4f}\tRES:{2:.4f}\tREC:{3:.4f}"
    model.meters = {'E-FL':AverageMeter(),'E-MV':AverageMeter(),'E-MC':AverageMeter(),'E-RES':AverageMeter(),
                    'D-MV':AverageMeter(),'D-MC':AverageMeter(),'D-RES':AverageMeter(),'D-REC':AverageMeter()}
    
def showTimer(model):
    if model.name in ['SPVC','SPVC-R','RLVC','DVC']:
        print('------------',model.name,'------------')
        print(model.fmt_enc_str.format(model.meters['E-FL'].avg,model.meters['E-MV'].avg,model.meters['E-MC'].avg,model.meters['E-RES'].avg))
        print(model.fmt_dec_str.format(model.meters["D-MV"].avg,model.meters["D-MC"].avg,model.meters["D-RES"].avg,model.meters["D-REC"].avg))
    
def update_training(model, epoch):
    # warmup with all gamma set to 1
    # optimize for bpp,img loss and focus only reconstruction loss
    # optimize bpp and app loss only
    
    # setup training weights
    if epoch <= 3:
        model.r_img, model.r_bpp, model.r_aux = 1,1,1
        model.r_rec, model.r_flow, model.r_warp, model.r_mc = 1,1,1,1
        model.r_app = 0
    else:
        model.r_img, model.r_bpp, model.r_aux = 0,1,1
        model.r_rec, model.r_flow, model.r_warp, model.r_mc = 1,0,0,0
        model.r_app = 1 # [1,2,4,8]
    
    # whether to compute action detection
    doAD = True if model.r_app > 0 else False
    
    model.epoch = epoch
    
    return doAD
        
def compress_video_group(model, frame_idx, cache, startNewClip):
    if startNewClip:
        imgByteArr = io.BytesIO()
        width,height = 224,224
        fps = 25
        Q = 27#15,19,23,27
        GOP = 13
        output_filename = 'tmp/videostreams/output.mp4'
        if model.name == 'x265':
            cmd = f'/usr/bin/ffmpeg -y -s {width}x{height} -pixel_format bgr24 -f rawvideo -r {fps} -i pipe: -vcodec libx265 -pix_fmt yuv420p -preset veryfast -tune zerolatency -x265-params "crf={Q}:keyint={GOP}:verbose=1" {output_filename}'
        elif model.name == 'x264':
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
        cache['psnr'] = {}
        cache['msssim'] = {}
        cache['aux'] = {}
        cache['end_of_batch'] = {}
        bpp = video_size*1.0/len(clip)/(height*width)
        for i in range(frame_idx-1,len(clip)):
            Y1_raw = transforms.ToTensor()(raw_clip[i]).cuda().unsqueeze(0)
            Y1_com = clip[i].unsqueeze(0)
            cache['img_loss'][i] = torch.FloatTensor([0]).squeeze(0).cuda(0)
            cache['bpp_est'][i] = torch.FloatTensor([0]).cuda(0)
            cache['psnr'][i] = PSNR(Y1_raw, Y1_com)
            cache['msssim'][i] = MSSSIM(Y1_raw, Y1_com)
            cache['bpp_act'][i] = torch.FloatTensor([bpp])
            cache['aux'][i] = torch.FloatTensor([0]).cuda(0)
            cache['end_of_batch'][i] = (i%4==0 or i==(len(clip)-1))
        cache['clip'] = clip
    cache['max_seen'] = frame_idx-1
    return True
        
# depending on training or testing
# the compression time should be recorded accordinglly
def compress_video_sequential(model, frame_idx, cache, startNewClip):
    # process the involving GOP
    # if process in order, some frames need later frames to compress
    if startNewClip:
        # create cache
        cache['bpp_est'] = {}
        cache['img_loss'] = {}
        cache['aux'] = {}
        cache['bpp_act'] = {}
        cache['psnr'] = {}
        cache['msssim'] = {}
        cache['hidden'] = None
        cache['max_proc'] = -1
        cache['end_of_batch'] = {}
        # the first frame to be compressed in a video
    assert frame_idx>=1, 'Frame index less than 1'
    if cache['max_proc'] >= frame_idx-1:
        cache['max_seen'] = frame_idx-1
    else:
        ranges, cache['max_seen'], cache['max_proc'] = index2GOP(frame_idx-1, len(cache['clip']))
        for _range in ranges:
            prev_j = -1
            for loc,j in enumerate(_range):
                progressive_compression(model, j, prev_j, cache, loc==1, loc>=2)
                prev_j = j
        
def compress_video_batch(model, frame_idx, cache, startNewClip):
    # process the involving GOP
    # how to deal with backward P frames?
    # if process in order, some frames need later frames to compress
    if startNewClip:
        # create cache
        cache['bpp_est'] = {}
        cache['img_loss'] = {}
        cache['aux'] = {}
        cache['bpp_act'] = {}
        cache['msssim'] = {}
        cache['psnr'] = {}
        cache['end_of_batch'] = {}
        cache['max_proc'] = -1
    if cache['max_proc'] >= frame_idx-1:
        cache['max_seen'] = frame_idx-1
    else:
        ranges, cache['max_seen'], cache['max_proc'] = index2GOP(frame_idx-1, len(cache['clip']))
        parallel_compression(model, ranges, cache)
      
def progressive_compression(model, i, prev, cache, P_flag, RPM_flag):
    # frame shape
    _,h,w = cache['clip'][0].shape
    # frames to be processed
    Y0_com = cache['clip'][prev].unsqueeze(0) if prev>=0 else None
    Y1_raw = cache['clip'][i].unsqueeze(0)
    # hidden variables
    if P_flag:
        hidden = model.init_hidden(h,w)
    else:
        hidden = cache['hidden']
    Y1_com,hidden,bpp_est,img_loss,aux_loss,bpp_act,psnr,msssim = model(Y0_com, Y1_raw, hidden, RPM_flag)
    cache['hidden'] = hidden
    cache['clip'][i] = Y1_com.detach().squeeze(0).cuda(0)
    cache['img_loss'][i] = img_loss.cuda(0)
    cache['aux'][i] = aux_loss.cuda(0)
    cache['bpp_est'][i] = bpp_est.cuda(0)
    cache['psnr'][i] = psnr.cuda(0)
    cache['msssim'][i] = msssim.cuda(0)
    cache['bpp_act'][i] = bpp_act.cuda(0)
    cache['end_of_batch'][i] = (i%4==0)
    #print(i,float(bpp_est),float(bpp_act),float(psnr))
    # we can record PSNR wrt the distance to I-frame to show error propagation)
        
def parallel_compression(model, ranges, cache):
    # we can summarize the result for each index to study error propagation
    # I compression
    # make it bidirectional?
    # I frame compression
    I_frame_idx = ranges[0][0]
    x_hat, bpp_est, img_loss, aux_loss, bpp_act, psnr, msssim = I_compression(cache['clip'][I_frame_idx].unsqueeze(0), model.I_level)
    cache['clip'][I_frame_idx] = x_hat.squeeze(0).cuda()
    cache['img_loss'][I_frame_idx] = img_loss.cuda()
    cache['aux'][I_frame_idx] = aux_loss.cuda()
    cache['bpp_est'][I_frame_idx] = bpp_est.cuda()
    cache['psnr'][I_frame_idx] = psnr.cuda()
    cache['msssim'][I_frame_idx] = msssim.cuda()
    cache['bpp_act'][I_frame_idx] = bpp_act.cuda()
    cache['end_of_batch'][I_frame_idx] = True
    
    if len(ranges) == 2:
        ranges[1] = [I_frame_idx] + ranges[1]
    
    # P compression, not including I frame
    for _range in ranges:
        img_list = [cache['clip'][k] for k in _range]
        idx_list = _range[1:]
        n = len(idx_list)
        if n==0:continue
        x = torch.stack(img_list, dim=0)
        x_hat, bpp_est, img_loss, aux_loss, bpp_act, psnr, msssim = model(x)
        eob_idx = max(idx_list)
        bob_idx = min(idx_list) # only loss on the begin of batch is valid
        zero = torch.FloatTensor([0]).squeeze(0).cuda(0)
        for pos,j in enumerate(idx_list):
            cache['clip'][j] = x_hat[pos].squeeze(0)
            cache['img_loss'][j] = img_loss if j==bob_idx else None
            cache['aux'][j] = aux_loss if j==bob_idx else None
            cache['bpp_est'][j] = bpp_est if j==bob_idx else None
            cache['psnr'][j] = psnr if j==bob_idx else None
            cache['msssim'][j] = msssim if j==bob_idx else None
            cache['bpp_act'][j] = bpp_act if j==bob_idx else None
            cache['end_of_batch'][j] = False
        cache['end_of_batch'][eob_idx] = True
            
def index2GOP(i, clip_len, fP = 6, bP = 6):
    # bi: fP=bP=6
    # uni:fP=12,bp=0
    # input: 
    # - idx: the frame index of interest
    # output: 
    # - ranges: the range(s) of GOP involving this frame
    # - max_seen: max index has been seen
    # - max_proc: max processed index
    # normally progressive coding will get 1 or 2 range(s)
    # parallel coding will get 1 range
    
    GOP = fP + bP + 1
    # 0 1  2  3  4  5  6  7  8  9  10 11 12 13
    # I fP fP fP fP fP fP bP bP bP bP bP bP I 
    ranges = []
    # <      case 1    >  
    # first time calling this function will mostly fall in case 1
    # case 1 will create one range
    if i%GOP <= fP:
        # e.g.: i=4,left=0,right=6,mid=0
        mid = i
        left = i
        right = min(i//GOP*GOP+fP,clip_len-1)
        _range = [j for j in range(mid,right+1)]
        ranges += [_range]
    #                     <      case 2   >
    # later calling this function will fall in case 2
    # case 2 will create one range if parallel or two ranges if progressive
    else:
        # e.g.: i=8,left=7,right=19,mid=13
        mid = min((i//GOP+1)*GOP,clip_len-1)
        left = i
        right = min((i//GOP+1)*GOP+fP,clip_len-1)
        possible_I = (i//GOP+1)*GOP
        # first backward
        _range = [j for j in range(mid,left-1,-1)]
        ranges += [_range]
        # then forward
        if right >= mid+1:
            _range = [j for j in range(mid+1,right+1)]
            ranges += [_range]
    max_seen, max_proc = i, right
    return ranges, max_seen, max_proc
    
# DVC,RLVC,MLVC
# Need to measure time and implement decompression for demo
# cache should store start/end-of-GOP information for the action detector to stop; test will be based on it
class IterPredVideoCodecs(nn.Module):
    def __init__(self, name, channels=128, noMeasure=True):
        super(IterPredVideoCodecs, self).__init__()
        self.name = name 
        self.optical_flow = OpticalFlowNet()
        self.MC_network = MCNet()
        self.mv_codec = Coder2D(self.name, in_channels=2, channels=channels, kernel=3, padding=1, noMeasure=noMeasure)
        self.res_codec = Coder2D(self.name, in_channels=3, channels=channels, kernel=5, padding=2, noMeasure=noMeasure)
        self.channels = channels
        init_training_params(self)
        self.epoch = -1
        self.noMeasure = noMeasure
        
        # split on multi-gpus
        self.split()

    def split(self):
        self.optical_flow.cuda(0)
        self.mv_codec.cuda(0)
        self.MC_network.cuda(1)
        self.res_codec.cuda(1)

    def forward(self, Y0_com, Y1_raw, hidden_states, RPM_flag, use_psnr=True):
        # Y0_com: compressed previous frame, [1,c,h,w]
        # Y1_raw: uncompressed current frame
        batch_size, _, Height, Width = Y1_raw.shape
        if self.name == 'RAW':
            bpp_est = bpp_act = metrics = torch.FloatTensor([0]).cuda(0)
            aux_loss = img_loss = torch.FloatTensor([0]).squeeze(0).cuda(0)
            return Y1_raw, hidden_states, bpp_est, img_loss, aux_loss, bpp_act, metrics
        if Y0_com is None:
            Y1_com, bpp_est, img_loss, aux_loss, bpp_act, psnr, msssim = I_compression(Y1_raw, self.I_level)
            return Y1_com, hidden_states, bpp_est, img_loss, aux_loss, bpp_act, psnr, msssim
        # otherwise, it's P frame
        # hidden states
        rae_mv_hidden, rae_res_hidden, rpm_mv_hidden, rpm_res_hidden = hidden_states
        # estimate optical flow
        t_0 = time.perf_counter()
        mv_tensor, l0, l1, l2, l3, l4 = self.optical_flow(Y0_com, Y1_raw)
        if not self.noMeasure:
            self.meters['E-FL'].update(time.perf_counter() - t_0)
        # compress optical flow
        mv_hat,rae_mv_hidden,rpm_mv_hidden,mv_act,mv_est,mv_aux = self.mv_codec(mv_tensor, rae_mv_hidden, rpm_mv_hidden, RPM_flag)
        if not self.noMeasure:
            self.meters['E-MV'].update(self.mv_codec.enc_t)
            self.meters['D-MV'].update(self.mv_codec.dec_t)
        # motion compensation
        t_0 = time.perf_counter()
        loc = get_grid_locations(batch_size, Height, Width).type(Y0_com.type())
        Y1_warp = F.grid_sample(Y0_com, loc + mv_hat.permute(0,2,3,1), align_corners=True)
        MC_input = torch.cat((mv_hat, Y0_com, Y1_warp), axis=1)
        Y1_MC = self.MC_network(MC_input.cuda(1))
        t_comp = time.perf_counter() - t_0
        if not self.noMeasure:
            self.meters['E-MC'].update(t_comp)
            self.meters['D-MC'].update(t_comp)
        warp_loss = calc_loss(Y1_raw, Y1_warp.to(Y1_raw.device), self.r, use_psnr)
        mc_loss = calc_loss(Y1_raw, Y1_MC.to(Y1_raw.device), self.r, use_psnr)
        # compress residual
        res_tensor = Y1_raw.cuda(1) - Y1_MC
        res_hat,rae_res_hidden,rpm_res_hidden,res_act,res_est,res_aux = self.res_codec(res_tensor, rae_res_hidden, rpm_res_hidden, RPM_flag)
        if not self.noMeasure:
            self.meters['E-RES'].update(self.res_codec.enc_t)
            self.meters['D-RES'].update(self.res_codec.dec_t)
        # reconstruction
        t_0 = time.perf_counter()
        Y1_com = torch.clip(res_hat + Y1_MC, min=0, max=1)
        if not self.noMeasure:
            self.meters['D-REC'].update(time.perf_counter() - t_0)
        ##### compute bits
        # estimated bits
        bpp_est = (mv_est + res_est.cuda(0))/(Height * Width * batch_size)
        # actual bits
        bpp_act = (mv_act + res_act.to(mv_act.device))/(Height * Width * batch_size)
        # auxilary loss
        aux_loss = (mv_aux + res_aux.to(mv_aux.device))/2
        # calculate metrics/loss
        psnr = PSNR(Y1_raw, Y1_com.to(Y1_raw.device))
        msssim = MSSSIM(Y1_raw, Y1_com.to(Y1_raw.device))
        rec_loss = calc_loss(Y1_raw, Y1_com.to(Y1_raw.device), self.r, use_psnr)
        img_loss = (self.r_rec*rec_loss + self.r_warp*warp_loss + self.r_mc*mc_loss)
        img_loss += (l0+l1+l2+l3+l4)/5*1024*self.r_flow
        # hidden states
        hidden_states = (rae_mv_hidden.detach(), rae_res_hidden.detach(), rpm_mv_hidden, rpm_res_hidden)
        return Y1_com.cuda(0), hidden_states, bpp_est, img_loss, aux_loss, bpp_act, psnr, msssim
        
    def loss(self, pix_loss, bpp_loss, aux_loss, app_loss=None):
        loss = self.r_img*pix_loss + self.r_bpp*bpp_loss + self.r_aux*aux_loss
        if self.name in ['MLVC','RAW']:
            if app_loss is not None:
                loss += self.r_app*app_loss
        return loss
    
    def init_hidden(self, h, w):
        rae_mv_hidden = torch.zeros(1,self.channels*4,h//4,w//4).cuda()
        rae_res_hidden = torch.zeros(1,self.channels*4,h//4,w//4).cuda()
        rpm_mv_hidden = torch.zeros(1,self.channels*2,h//16,w//16).cuda()
        rpm_res_hidden = torch.zeros(1,self.channels*2,h//16,w//16).cuda()
        return (rae_mv_hidden, rae_res_hidden, rpm_mv_hidden, rpm_res_hidden)
            
# DCVC?
# adding MC network doesnt help much
class DCVC(nn.Module):
    def __init__(self, name, channels=64, channels2=96, noMeasure=True):
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
        self.ctx_decoder1 = nn.Sequential(nn.ConvTranspose2d(channels2, channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                                        GDN(channels, inverse=True),
                                        nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                                        GDN(channels, inverse=True),
                                        ResidualBlock(channels,channels),
                                        nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                                        GDN(channels, inverse=True),
                                        ResidualBlock(channels,channels),
                                        nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                                        )
        self.ctx_decoder2 = nn.Sequential(nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1),
                                        ResidualBlock(channels,channels),
                                        ResidualBlock(channels,channels),
                                        nn.Conv2d(channels, 3, kernel_size=3, stride=1, padding=1)
                                        )
        if name == 'DCVC_v2':
            self.MC_network = MCNet()
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
        self.mv_codec = Coder2D(name, in_channels=2, channels=channels, kernel=3, padding=1, noMeasure=noMeasure)
        self.latent_codec = Coder2D('joint', channels=channels2, noMeasure=noMeasure, downsample=False)
        init_training_params(self)
        self.name = name
        self.channels = channels
        self.split()
        self.updated = False
        self.noMeasure = noMeasure

    def split(self):
        self.optical_flow.cuda(0)
        self.mv_codec.cuda(0)
        if self.name == 'DCVC_v2':
            self.MC_network.cuda(1)
        self.feature_extract.cuda(1)
        self.ctx_refine.cuda(1)
        self.tmp_prior_encoder.cuda(1)
        self.ctx_encoder.cuda(1)
        self.latent_codec.cuda(1)
        self.ctx_decoder1.cuda(1)
        self.ctx_decoder2.cuda(1)
    
    def forward(self, x_hat_prev, x, hidden_states, RPM_flag, use_psnr=True):
        if not self.noMeasure:
            self.enc_t = [];self.dec_t = []
            
        # I-frame compression
        if x_hat_prev is None:
            x_hat, bpp_est, img_loss, aux_loss, bpp_act, psnr, msssim = I_compression(x,self.I_level)
            return x_hat, hidden_states, bpp_est, img_loss, aux_loss, bpp_act, psnr, msssim
        # size
        bs,c,h,w = x.size()
        
        # hidden states
        rae_mv_hidden, rpm_mv_hidden = hidden_states
                
        # motion estimation
        t_0 = time.perf_counter()
        mv, l0, l1, l2, l3, l4 = self.optical_flow(x, x_hat_prev)
        t_flow = time.perf_counter() - t_0
        if not self.noMeasure:
            self.enc_t += [t_flow]
        
        # compress optical flow
        mv_hat,rae_mv_hidden,rpm_mv_hidden,mv_act,mv_est,mv_aux = self.mv_codec(mv, rae_mv_hidden, rpm_mv_hidden, RPM_flag)
        if not self.noMeasure:
            self.enc_t += [self.mv_codec.enc_t]
            self.dec_t += [self.mv_codec.dec_t]
        
        # warping
        t_0 = time.perf_counter()
        loc = get_grid_locations(bs, h, w).cuda(1)
        if self.name == 'DCVC':
            # feature extraction
            x_feat = self.feature_extract(x_hat_prev.cuda(1))
            
            # motion compensation
            x_feat_warp = F.grid_sample(x_feat, loc + mv_hat.permute(0,2,3,1).cuda(1), align_corners=True) # the difference
            x_tilde = F.grid_sample(x_hat_prev.cuda(1), loc + mv_hat.permute(0,2,3,1).cuda(1), align_corners=True)
            warp_loss = calc_loss(x, x_tilde.to(x.device), self.r, use_psnr)
        else:
            # motion compensation
            x_warp = F.grid_sample(x_hat_prev.cuda(1), loc + mv_hat.permute(0,2,3,1).cuda(1), align_corners=True) # the difference
            warp_loss = calc_loss(x, x_warp.to(x.device), self.r, use_psnr)
            x_mc = self.MC_network(torch.cat((mv_hat.cuda(1), x_hat_prev.cuda(1), x_warp), axis=1).cuda(1))
            mc_loss = calc_loss(x, x_mc.to(x.device), self.r, use_psnr)
            
            # feature extraction
            x_feat_warp = self.feature_extract(x_mc)
        t_warp = time.perf_counter() - t_0
        if not self.noMeasure:
            self.enc_t += [t_warp]
            self.dec_t += [t_warp]
        
        # context refinement
        t_0 = time.perf_counter()
        context = self.ctx_refine(x_feat_warp)
        t_refine = time.perf_counter() - t_0
        if not self.noMeasure:
            self.enc_t += [t_refine]
            self.dec_t += [t_refine]
        
        # temporal prior
        t_0 = time.perf_counter()
        prior = self.tmp_prior_encoder(context)
        t_prior = time.perf_counter() - t_0
        if not self.noMeasure:
            self.enc_t += [t_prior]
            self.dec_t += [t_prior]
        
        # contextual encoder
        y = self.ctx_encoder(torch.cat((x, context.to(x.device)), axis=1).cuda(1))
        
        # entropy model
        y_hat,_,_,y_act,y_est,y_aux = self.latent_codec(y, prior=prior)
        
        # contextual decoder
        t_0 = time.perf_counter()
        x_hat = self.ctx_decoder1(y_hat)
        x_hat = self.ctx_decoder2(torch.cat((x_hat, context.to(x_hat.device)), axis=1)).to(x.device)
        if not self.noMeasure:
            self.dec_t += [time.perf_counter() - t_0]
        
        # estimated bits
        bpp_est = (mv_est + y_est.cuda(0))/(h * w * bs)
        # actual bits
        bpp_act = (mv_act + y_act.cuda(0))/(h * w * bs)
        #print(float(mv_est/(h * w * bs)), float(mv_act/(h * w * bs)), float(y_est/(h * w * bs)), float(y_act/(h * w * bs)))
        # auxilary loss
        aux_loss = (mv_aux + y_aux.cuda(0))/2
        # calculate metrics/loss
        psnr = PSNR(x, x_hat.cuda(0))
        msssim = MSSSIM(x, x_hat.cuda(0))
        rec_loss = calc_loss(x, x_hat.cuda(0), self.r, use_psnr)
        if self.name == 'DCVC':
            img_loss = (self.r_rec*rec_loss + self.r_warp*warp_loss)
        else:
            img_loss = (self.r_rec*rec_loss + self.r_warp*warp_loss + self.r_mc*mc_loss)
        img_loss += (l0+l1+l2+l3+l4)/5*1024*self.r_flow
        # hidden states
        hidden_states = (rae_mv_hidden.detach(), rpm_mv_hidden)
        if not self.noMeasure:
            print(np.sum(self.enc_t),np.sum(self.dec_t),self.enc_t,self.dec_t)
        return x_hat.cuda(0), hidden_states, bpp_est, img_loss, aux_loss, bpp_act, psnr, msssim
        
    def loss(self, pix_loss, bpp_loss, aux_loss, app_loss=None):
        if app_loss is None:
            return self.r_img*pix_loss + self.r_bpp*bpp_loss + self.r_aux*aux_loss
        else:
            return self.r_app*app_loss + self.r_img*pix_loss + self.r_bpp*bpp_loss + self.r_aux*aux_loss
        
    def init_hidden(self, h, w):
        rae_mv_hidden = torch.zeros(1,self.channels*4,h//4,w//4).cuda()
        rpm_mv_hidden = torch.zeros(1,self.channels*2,h//16,w//16).cuda()
        return (rae_mv_hidden, rpm_mv_hidden)
        
class StandardVideoCodecs(nn.Module):
    def __init__(self, name):
        super(StandardVideoCodecs, self).__init__()
        self.name = name # x264, x265?
        self.placeholder = torch.nn.Parameter(torch.zeros(1))
        init_training_params(self)
    
    def loss(self, pix_loss, bpp_loss, aux_loss, app_loss=None):
        if app_loss is None:
            return self.r_img*pix_loss + self.r_bpp*bpp_loss + self.r_aux*aux_loss
        else:
            return self.r_app*app_loss + self.r_img*pix_loss + self.r_bpp*bpp_loss + self.r_aux*aux_loss
        
def I_compression(Y1_raw, I_level):
    # we can compress with bpg,deepcod ...
    batch_size, _, Height, Width = Y1_raw.shape
    prename = "tmp/frames/prebpg"
    binname = "tmp/frames/bpg"
    postname = "tmp/frames/postbpg"
    raw_img = transforms.ToPILImage()(Y1_raw.squeeze(0))
    raw_img.save(prename + '.jpg')
    pre_bits = os.path.getsize(prename + '.jpg')*8
    os.system('bpgenc -f 444 -m 9 ' + prename + '.jpg -o ' + binname + '.bin -q ' + str(I_level))
    os.system('bpgdec ' + binname + '.bin -o ' + postname + '.jpg')
    post_bits = os.path.getsize(binname + '.bin')*8/(Height * Width * batch_size)
    bpp_act = torch.FloatTensor([post_bits]).squeeze(0)
    bpg_img = Image.open(postname + '.jpg').convert('RGB')
    Y1_com = transforms.ToTensor()(bpg_img).unsqueeze(0)
    psnr = PSNR(Y1_raw, Y1_com)
    msssim = MSSSIM(Y1_raw, Y1_com)
    bpp_est = bpp_act
    loss = aux_loss = torch.FloatTensor([0]).squeeze(0)
    return Y1_com, bpp_est, loss, aux_loss, bpp_act, psnr, msssim
    
def load_state_dict_only(model, state_dict, keyword):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if keyword not in name: continue
        if name in own_state:
            own_state[name].copy_(param)
    
def load_state_dict_whatever(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name.endswith("._offset") or name.endswith("._quantized_cdf") or name.endswith("._cdf_length") or name.endswith(".scale_table"):
             continue
        if name in own_state and own_state[name].size() == param.size():
            own_state[name].copy_(param)
            
def load_state_dict_all(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name.endswith("._offset") or name.endswith("._quantized_cdf") or name.endswith("._cdf_length") or name.endswith(".scale_table"):
             continue
        own_state[name].copy_(param)
    
def PSNR(Y1_raw, Y1_com, use_list=False):
    Y1_com = Y1_com.to(Y1_raw.device)
    log10 = torch.log(torch.FloatTensor([10])).squeeze(0).to(Y1_raw.device)
    if not use_list:
        train_mse = torch.mean(torch.pow(Y1_raw - Y1_com, 2))
        quality = 10.0*torch.log(1/train_mse)/log10
    else:
        b = Y1_raw.size()[0]
        quality = []
        for i in range(b):
            train_mse = torch.mean(torch.pow(Y1_raw[i:i+1] - Y1_com[i:i+1].unsqueeze(0), 2))
            psnr = 10.0*torch.log(1/train_mse)/log10
            quality.append(psnr)
        quality = torch.stack(quality,dim=0).mean(dim=0)
    return quality

def MSSSIM(Y1_raw, Y1_com, use_list=False):
    Y1_com = Y1_com.to(Y1_raw.device)
    if not use_list:
        quality = pytorch_msssim.ms_ssim(Y1_raw, Y1_com)
    else:
        bs = Y1_raw.size()[0]
        quality = []
        for i in range(bs):
            quality.append(pytorch_msssim.ms_ssim(Y1_raw[i].unsqueeze(0), Y1_com[i].unsqueeze(0)))
        quality = torch.stack(quality,dim=0).mean(dim=0)
    return quality
    
def calc_loss(Y1_raw, Y1_com, r, use_psnr):
    if use_psnr:
        loss = torch.mean(torch.pow(Y1_raw - Y1_com.to(Y1_raw.device), 2))*r
    else:
        metrics = MSSSIM(Y1_raw, Y1_com.to(Y1_raw.device))
        loss = r*(1-metrics)
    return loss

# pyramid flow estimation
class OpticalFlowNet(nn.Module):
    def __init__(self):
        super(OpticalFlowNet, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=0)
        self.loss = LossNet()

    def forward(self, im1_4, im2_4):
        # im1_4,im2_4:[1,c,h,w]
        # flow_4:[1,2,h,w]
        batch, _, h, w = im1_4.size()
        
        im1_3 = self.pool(im1_4)
        im1_2 = self.pool(im1_3)
        im1_1 = self.pool(im1_2)
        im1_0 = self.pool(im1_1)

        im2_3 = self.pool(im2_4)
        im2_2 = self.pool(im2_3)
        im2_1 = self.pool(im2_2)
        im2_0 = self.pool(im2_1)

        flow_zero = torch.zeros(batch, 2, h//16, w//16).to(im1_4.device)

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
        loc = get_grid_locations(batch_size, H, W).to(im1.device)
        flow = flow.to(im1.device)
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
    
def get_actual_bits(self, string):
    bits_act = torch.FloatTensor([len(b''.join(string))*8]).squeeze(0)
    return bits_act
        
def get_estimate_bits(self, likelihoods):
    log2 = torch.log(torch.FloatTensor([2])).squeeze(0).to(likelihoods.device)
    bits_est = torch.sum(torch.log(likelihoods)) / (-log2)
    return bits_est

class Coder2D(nn.Module):
    def __init__(self, keyword, in_channels=2, channels=128, kernel=3, padding=1, noMeasure=True, downsample=True):
        super(Coder2D, self).__init__()
        if downsample:
            self.enc_conv1 = nn.Conv2d(in_channels, channels, kernel_size=kernel, stride=2, padding=padding)
            self.enc_conv2 = nn.Conv2d(channels, channels, kernel_size=kernel, stride=2, padding=padding)
            self.enc_conv3 = nn.Conv2d(channels, channels, kernel_size=kernel, stride=2, padding=padding)
            self.enc_conv4 = nn.Conv2d(channels, channels, kernel_size=kernel, stride=2, padding=padding, bias=False)
            self.gdn1 = GDN(channels)
            self.gdn2 = GDN(channels)
            self.gdn3 = GDN(channels)
            self.dec_conv1 = nn.ConvTranspose2d(channels, channels, kernel_size=kernel, stride=2, padding=padding, output_padding=1)
            self.dec_conv2 = nn.ConvTranspose2d(channels, channels, kernel_size=kernel, stride=2, padding=padding, output_padding=1)
            self.dec_conv3 = nn.ConvTranspose2d(channels, channels, kernel_size=kernel, stride=2, padding=padding, output_padding=1)
            self.dec_conv4 = nn.ConvTranspose2d(channels, in_channels, kernel_size=kernel, stride=2, padding=padding, output_padding=1)
            self.igdn1 = GDN(channels, inverse=True)
            self.igdn2 = GDN(channels, inverse=True)
            self.igdn3 = GDN(channels, inverse=True)
        if keyword in ['MLVC','RLVC','rpm']:
            # for recurrent sequential model
            self.entropy_bottleneck = RecProbModel(channels)
            self.conv_type = 'rec'
            self.entropy_type = 'rpm'
        elif keyword in ['attn']:
            # for batch model
            self.entropy_bottleneck = MeanScaleHyperPriors(channels,useAttention=True)
            #self.entropy_bottleneck = MeanScaleHyperPriors(channels,useAttention=False)
            self.conv_type = 'attn'
            #self.conv_type = 'non-rec'
            self.entropy_type = 'mshp'
        elif keyword in ['mshp']:
            # for image codec, single frame
            self.entropy_bottleneck = MeanScaleHyperPriors(channels,useAttention=False)
            self.conv_type = 'non-rec' # not need for single image compression
            self.entropy_type = 'mshp'
        elif keyword in ['DVC','base','DCVC','DCVC_v2']:
            # for sequential model with no recurrent network
            from compressai.entropy_models import EntropyBottleneck
            EntropyBottleneck.get_actual_bits = get_actual_bits
            EntropyBottleneck.get_estimate_bits = get_estimate_bits
            self.entropy_bottleneck = EntropyBottleneck(channels)
            self.conv_type = 'non-rec'
            self.entropy_type = 'base'
        elif keyword == 'joint-attn':
            self.entropy_bottleneck = JointAutoregressiveHierarchicalPriors(channels,useAttention=True)
            self.conv_type = 'none'
            self.entropy_type = 'joint'
        elif keyword == 'joint':
            self.entropy_bottleneck = JointAutoregressiveHierarchicalPriors(channels,useAttention=False)
            self.conv_type = 'none'
            self.entropy_type = 'joint'
        else:
            print('Bottleneck not implemented for:',keyword)
            exit(1)
        print('Conv type:',self.conv_type,'entropy type:',self.entropy_type)
        self.channels = channels
        if self.conv_type == 'rec':
            self.enc_lstm = ConvLSTM(channels)
            self.dec_lstm = ConvLSTM(channels)
        elif self.conv_type == 'attn':
            self.s_attn_a = AttentionBlock(channels)
            self.s_attn_s = AttentionBlock(channels)
            self.t_attn_a = Attention(channels)
            self.t_attn_s = Attention(channels)
            #self.s_attn_a = Attention(channels)
            #self.s_attn_s = Attention(channels)
            
        self.downsample = downsample
        self.updated = False
        self.noMeasure = noMeasure
        # include two average meter to measure time
        
    def forward(self, x, rae_hidden=None, rpm_hidden=None, RPM_flag=False, prior=None):
        # update only once during testing
        if not self.updated and not self.training:
            self.entropy_bottleneck.update(force=True)
            self.updated = True
            
        if not self.noMeasure:
            self.enc_t = self.dec_t = 0
        
        # latent states
        if self.conv_type == 'rec':
            state_enc, state_dec = torch.split(rae_hidden.to(x.device),self.channels*2,dim=1)
            
        # Time measurement: start
        if not self.noMeasure:
            t_0 = time.perf_counter()
            
        # compress
        if self.downsample:
            x = self.gdn1(self.enc_conv1(x))
            x = self.gdn2(self.enc_conv2(x))
            
            if self.conv_type == 'rec':
                x, state_enc = self.enc_lstm(x, state_enc)
            elif self.conv_type == 'attn':
                # use attention
                B,C,H,W = x.size()
                x = self.s_attn_a(x)
                x = self.t_attn_a(x)
                
            x = self.gdn3(self.enc_conv3(x))
            latent = self.enc_conv4(x) # latent optical flow
        else:
            latent = x
        
        # Time measurement: end
        if not self.noMeasure:
            self.enc_t += time.perf_counter() - t_0
        
        # quantization + entropy coding
        if self.entropy_type == 'base':
            if self.noMeasure:
                latent_hat, likelihoods = self.entropy_bottleneck(latent, training=self.training)
                if not self.training:
                    latent_string = self.entropy_bottleneck.compress(latent)
            else:
                # encoding
                t_0 = time.perf_counter()
                latent_string = self.entropy_bottleneck.compress(latent)
                self.entropy_bottleneck.enc_t = time.perf_counter() - t_0
                # decoding
                t_0 = time.perf_counter()
                latent_hat = self.entropy_bottleneck.decompress(latent_string, latent.size()[-2:])
                self.entropy_bottleneck.dec_t = time.perf_counter() - t_0
        elif self.entropy_type == 'mshp':
            if self.noMeasure:
                latent_hat, likelihoods = self.entropy_bottleneck(latent, training=self.training)
                if not self.training:
                    latent_string = self.entropy_bottleneck.compress(latent)
            else:
                latent_string, shape = self.entropy_bottleneck.compress_slow(latent)
                latent_hat = self.entropy_bottleneck.decompress_slow(latent_string, shape)
        elif self.entropy_type == 'joint':
            if self.noMeasure:
                latent_hat, likelihoods = self.entropy_bottleneck(latent, prior, training=self.training)
                if not self.training:
                    latent_string = self.entropy_bottleneck.compress(latent)
            else:
                latent_string,shape = self.entropy_bottleneck.compress_slow(latent, prior)
                latent_hat = self.entropy_bottleneck.decompress_slow(latent_string, shape, prior)
        else:
            self.entropy_bottleneck.set_RPM(RPM_flag)
            if self.noMeasure:
                latent_hat, likelihoods, rpm_hidden = self.entropy_bottleneck(latent, rpm_hidden, training=self.training)
                if not self.training:
                    latent_string = self.entropy_bottleneck.compress(latent)
            else:
                latent_string, _ = self.entropy_bottleneck.compress_slow(latent,rpm_hidden)
                latent_hat, rpm_hidden = self.entropy_bottleneck.decompress_slow(latent_string, latent.size()[-2:], rpm_hidden)
            self.entropy_bottleneck.set_prior(latent)
            
        # add in the time in entropy bottleneck
        if not self.noMeasure:
            self.enc_t += self.entropy_bottleneck.enc_t
            self.dec_t += self.entropy_bottleneck.dec_t
        
        # calculate bpp (estimated) if it is training else it will be set to 0
        if self.noMeasure:
            bits_est = self.entropy_bottleneck.get_estimate_bits(likelihoods)
        else:
            bits_est = torch.FloatTensor([0]).squeeze(0).to(x.device)
        
        # calculate bpp (actual)
        if not self.training:
            bits_act = self.entropy_bottleneck.get_actual_bits(latent_string)
        else:
            bits_act = bits_est

        # Time measurement: start
        if not self.noMeasure:
            t_0 = time.perf_counter()
            
        # decompress
        if self.downsample:
            x = self.igdn1(self.dec_conv1(latent_hat))
            x = self.igdn2(self.dec_conv2(x))
            
            if self.conv_type == 'rec':
                x, state_dec = self.enc_lstm(x, state_dec)
            elif self.conv_type == 'attn':
                # use attention
                B,C,H,W = x.size()
                x = self.s_attn_s(x)
                x = self.t_attn_s(x)
                
            x = self.igdn3(self.dec_conv3(x))
            hat = self.dec_conv4(x)
        else:
            hat = latent_hat
        
        # Time measurement: end
        if not self.noMeasure:
            self.dec_t += time.perf_counter() - t_0
        
        # auxilary loss
        aux_loss = self.entropy_bottleneck.loss()/self.channels
        
        if self.conv_type == 'rec':
            rae_hidden = torch.cat((state_enc, state_dec),dim=1)
            
        return hat, rae_hidden, rpm_hidden, bits_act, bits_est, aux_loss
            
    def compress_sequence(self,x):
        bs,c,h,w = x.size()
        x_est = torch.FloatTensor([0]).squeeze(0).cuda()
        x_act = torch.FloatTensor([0]).squeeze(0).cuda()
        x_aux = torch.FloatTensor([0]).squeeze(0).cuda()
        if not self.downsample:
            rpm_hidden = torch.zeros(1,self.channels*2,h,w)
        else:
            rpm_hidden = torch.zeros(1,self.channels*2,h//16,w//16)
        rae_hidden = torch.zeros(1,self.channels*4,h//4,w//4)
        enc_t = dec_t = 0
        x_hat_list = []
        for frame_idx in range(bs):
            x_i = x[frame_idx,:,:,:].unsqueeze(0)
            x_hat_i,rae_hidden,rpm_hidden,x_act_i,x_est_i,x_aux_i = self.forward(x_i, rae_hidden, rpm_hidden, frame_idx>=1)
            x_hat_list.append(x_hat_i.squeeze(0))
            
            # calculate bpp (estimated) if it is training else it will be set to 0
            x_est += x_est_i.cuda()
            
            # calculate bpp (actual)
            x_act += x_act_i.cuda()
            
            # aux
            x_aux += x_aux_i.cuda()
            
            enc_t += self.enc_t
            dec_t += self.dec_t
        x_hat = torch.stack(x_hat_list, dim=0)
        self.enc_t,self.dec_t = enc_t,dec_t
        return x_hat,x_act,x_est,x_aux

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
    
    def forward(self, x):
        bs,C,H,W = x.size()
        x = x.view(bs,C,-1).permute(2,0,1).contiguous()
        
        # perform linear operation
        
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_model, self.dropout)
        
        output = self.out(scores) # bs * sl * d_model
        
        output = output.permute(1,2,0).view(bs,C,H,W).contiguous()
    
        return output
        
def set_model_grad(model,requires_grad=True):
    for k,v in model.named_parameters():
        v.requires_grad = requires_grad
        
def motion_compensation(mc_model,x,motion):
    bs, c, h, w = x.size()
    loc = get_grid_locations(bs, h, w).to(motion.device)
    warped_frames = F.grid_sample(x.to(motion.device), loc + motion.permute(0,2,3,1), align_corners=True)
    MC_input = torch.cat((motion, x.to(motion.device), warped_frames), axis=1)
    MC_frames = mc_model(MC_input)
    return MC_frames,warped_frames
    
class SVC(nn.Module):
    def __init__(self, name, channels=128, noMeasure=True):
        super(SVC, self).__init__()
        self.name = name 
        self.optical_flow = OpticalFlowNet()
        self.MC_network = MCNet()
        self.mv_codec = Coder2D('rpm', in_channels=2, channels=channels, kernel=3, padding=1, noMeasure=noMeasure)
        self.res_codec = Coder2D('rpm', in_channels=3, channels=channels, kernel=5, padding=2, noMeasure=noMeasure)
        self.channels = channels
        init_training_params(self)
        self.epoch = -1
        self.noMeasure = noMeasure
        # option 1: use raw frames for flow estimation
        # option 2: 
        
        # split on multi-gpus
        self.split()
        self.use_batch = False

    def split(self):
        self.optical_flow.cuda(0)
        self.mv_codec.cuda(0)
        self.MC_network.cuda(1)
        self.res_codec.cuda(1)
        
    def forward(self, x, use_psnr=True):
        if self.use_batch:
            return self._batch_process(x, use_psnr)
        else:
            return self._seq_process(x, use_psnr)
        
    def _batch_process(self, x, use_psnr=True):
        bs, c, h, w = x[1:].size()
        
        # flow estimation
        mv_tensor, l0, l1, l2, l3, l4 = self.optical_flow(x[:-1], x[1:])
        
        # flow compression
        mv_hat, mv_act, mv_est, mv_aux = self._compress_sequence(self.mv_codec, mv_tensor)
        
        # motion compensation
        loc = get_grid_locations(bs, h, w).to(x.device)
        Y1_warp = F.grid_sample(x[:-1], loc + mv_hat.permute(0,2,3,1), align_corners=True)
        MC_input = torch.cat((mv_hat, x[:-1], Y1_warp), axis=1)
        Y1_MC = self.MC_network(MC_input.cuda(1))
        
        # residual compression
        res_tensor = x[1:].cuda(1) - Y1_MC
        res_hat, res_act, res_est, res_aux = self._compress_sequence(self.res_codec, res_tensor)
        
        # reconstruction
        x_hat = torch.clip(res_hat + Y1_MC, min=0, max=1).to(x.device)
        
        # estimated bits
        bpp_est = (mv_est + res_est.cuda(0))/(h * w * bs)
        # actual bits
        bpp_act = (mv_act + res_act.to(mv_act.device))/(h * w * bs)
        # auxilary loss
        aux_loss = (mv_aux + res_aux.to(mv_aux.device))/(2 * bs)
        
        # compute loss and metrics
        psnr = PSNR(x[1:], x_hat, use_list=True)
        msssim = MSSSIM(x[1:], x_hat, use_list=True)
        warp_loss = calc_loss(x[1:], Y1_warp.to(x.device), self.r, use_psnr)
        mc_loss = calc_loss(x[1:], Y1_MC.to(x.device), self.r, use_psnr)
        rec_loss = calc_loss(x[1:], x_hat, self.r, use_psnr)
        flow_loss = (l0+l1+l2+l3+l4)/5*1024*self.r_flow
        img_loss = self.r_rec*rec_loss + self.r_warp*warp_loss + self.r_mc*mc_loss +self.r_flow*flow_loss
        
        return x_hat, bpp_est.repeat(bs), img_loss.repeat(bs), aux_loss.repeat(bs), bpp_act.repeat(bs), psnr, msssim
        
    def _compress_sequence(self, codec, mv_tensor):
        bs,_,h,w = mv_tensor.size()
        rae_mv_hidden, _, rpm_mv_hidden, _ = self.init_hidden(h,w)
        mv_hat=[]
        mv_act=torch.FloatTensor([0]).squeeze(0).cuda()
        mv_est=torch.FloatTensor([0]).squeeze(0).cuda()
        mv_aux=torch.FloatTensor([0]).squeeze(0).cuda()
        for k in range(bs):
            mv_hat_k,rae_mv_hidden,rpm_mv_hidden,mv_act_k,mv_est_k,mv_aux_k = codec(mv_tensor[k:k+1], rae_mv_hidden, rpm_mv_hidden, RPM_flag=(k>0))
            mv_hat += [mv_hat_k];mv_act += mv_act_k.cuda();mv_est += mv_est_k.cuda();mv_aux += mv_aux_k.cuda()
        mv_hat = torch.cat(mv_hat, dim=0)
        return mv_hat, mv_act, mv_est, mv_aux
        
    def _seq_process(self, x, use_psnr=True): 
        # input: raw frames=bs+1,c,h,w 
        bs, c, h, w = x[1:].size()
        hidden = self.init_hidden(h,w)
        bpp_est = {};img_loss = {};aux_loss = {};bpp_act = {};psnr = {};msssim = {};x_hat = {}
        # graph-based compression 
        g = generate_graph('3layers')
        #g = generate_graph('default')
        start = 0
        # BFS
        for start in g:
            if start>bs:continue
            if start == 0:
                Y0_com = x[:1]
            else:
                Y0_com = x_hat[start-1]
            for k in g[start]:
                if k>bs:continue
                Y1_raw = x[k:k+1]
                Y1_com, hidden, bpp_est_k, img_loss_k, aux_loss_k, bpp_act_k, psnr_k, msssim_k = \
                    self._process(Y0_com.detach(), Y1_raw, hidden, RPM_flag=(k not in g[0]), use_psnr=use_psnr)
                k = k-1
                x_hat[k] = Y1_com;bpp_est[k] = bpp_est_k;bpp_act[k] = bpp_act_k;img_loss[k] = img_loss_k;aux_loss[k] = aux_loss_k;psnr[k] = psnr_k;msssim[k] = msssim_k
        return x_hat, bpp_est, img_loss, aux_loss, bpp_act, psnr, msssim
        
    def _process(self, Y0_com, Y1_raw, hidden, RPM_flag, use_psnr=True):
        # Y0_com: compressed previous frame, [1,c,h,w]
        # Y1_raw: uncompressed current frame
        batch_size, _, Height, Width = Y1_raw.shape
        # otherwise, it's P frame
        # hidden states
        rae_mv_hidden, rae_res_hidden, rpm_mv_hidden, rpm_res_hidden = hidden
        # estimate optical flow
        mv_tensor, l0, l1, l2, l3, l4 = self.optical_flow(Y0_com, Y1_raw)
        # compress optical flow
        mv_hat,rae_mv_hidden,rpm_mv_hidden,mv_act,mv_est,mv_aux = self.mv_codec(mv_tensor, rae_mv_hidden, rpm_mv_hidden, RPM_flag)
        # motion compensation
        loc = get_grid_locations(batch_size, Height, Width).to(Y0_com.device)
        Y1_warp = F.grid_sample(Y0_com, loc + mv_hat.permute(0,2,3,1), align_corners=True)
        MC_input = torch.cat((mv_hat, Y0_com, Y1_warp), axis=1)
        Y1_MC = self.MC_network(MC_input.cuda(1))
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
        psnr = PSNR(Y1_raw, Y1_com.to(Y1_raw.device))
        msssim = MSSSIM(Y1_raw, Y1_com.to(Y1_raw.device))
        warp_loss = calc_loss(Y1_raw, Y1_warp.to(Y1_raw.device), self.r, use_psnr)
        mc_loss = calc_loss(Y1_raw, Y1_MC.to(Y1_raw.device), self.r, use_psnr)
        rec_loss = calc_loss(Y1_raw, Y1_com.to(Y1_raw.device), self.r, use_psnr)
        flow_loss = (l0+l1+l2+l3+l4)/5*1024*self.r_flow
        img_loss = self.r_rec*rec_loss + self.r_warp*warp_loss + self.r_mc*mc_loss +self.r_flow*flow_loss
        # hidden states
        hidden = (rae_mv_hidden.detach(), rae_res_hidden.detach(), rpm_mv_hidden, rpm_res_hidden)
        return Y1_com.cuda(0), hidden, bpp_est, img_loss, aux_loss, bpp_act, psnr, msssim
        
    def loss(self, pix_loss, bpp_loss, aux_loss, app_loss=None):
        loss = self.r_img*pix_loss + self.r_bpp*bpp_loss + self.r_aux*aux_loss
        return loss
    
    def init_hidden(self, h, w):
        rae_mv_hidden = torch.zeros(1,self.channels*4,h//4,w//4).cuda()
        rae_res_hidden = torch.zeros(1,self.channels*4,h//4,w//4).cuda()
        rpm_mv_hidden = torch.zeros(1,self.channels*2,h//16,w//16).cuda()
        rpm_res_hidden = torch.zeros(1,self.channels*2,h//16,w//16).cuda()
        return (rae_mv_hidden, rae_res_hidden, rpm_mv_hidden, rpm_res_hidden)
        
def generate_graph(graph_type='default'):
    # 7 nodes, 6 edges
    # the order to iterate graph also matters, leave it now
    # BFS or DFS?
    if graph_type == 'default':
        g = {}
        for k in range(6):
            g[k] = [k+1]
    elif graph_type == '3layers':
        g = {0:[1,4],1:[2,3],4:[5,6]}
    else:
        print('Undefined graph type:',graph_type)
        exit(1)
    return g
        
class SPVC(nn.Module):
    def __init__(self, name, channels=128, noMeasure=True):
        super(SPVC, self).__init__()
        self.name = name 
        self.optical_flow = OpticalFlowNet()
        self.MC_network = MCNet()
        if self.name in ['SPVC','SPVC-P','SPVC-M','SPVC-L']:
            # use attention in encoder and entropy model
            self.mv_codec = Coder2D('attn', in_channels=2, channels=channels, kernel=3, padding=1, noMeasure=noMeasure)
            self.res_codec = Coder2D('attn', in_channels=3, channels=channels, kernel=5, padding=2, noMeasure=noMeasure)
        elif self.name == 'SPVC-R':
            # use rpm for encoder and entropy
            self.mv_codec = Coder2D('rpm', in_channels=2, channels=channels, kernel=3, padding=1, noMeasure=noMeasure)
            self.res_codec = Coder2D('rpm', in_channels=3, channels=channels, kernel=5, padding=2, noMeasure=noMeasure)
        self.channels = channels
        init_training_params(self)
        # split on multi-gpus
        self.split()
        self.noMeasure = noMeasure
        self.use_psnr = False if self.name == 'SPVC-M' else True

    def split(self):
        self.optical_flow.cuda(0)
        self.mv_codec.cuda(0)
        self.MC_network.cuda(1)
        self.res_codec.cuda(1)
        
    def forward(self, x):
        x = x.cuda(0)
        bs, c, h, w = x[1:].size()
        
        # BATCH:compute optical flow
        t_0 = time.perf_counter()
        # obtain reference frames from a graph
        x_tar = x[1:]
        if self.name == 'SPVC-L':
            g = generate_graph('default')
        else:
            g = generate_graph('3layers')
        ref_index = [-1 for _ in x_tar]
        for start in g:
            if start>bs:continue
            for k in g[start]:
                if k>bs:continue
                ref_index[k-1] = start
        mv_tensors, l0, l1, l2, l3, l4 = self.optical_flow(x[ref_index], x_tar)
        if not self.noMeasure:
            self.meters['E-FL'].update(time.perf_counter() - t_0)
        
        # BATCH:compress optical flow
        if self.name in ['SPVC','SPVC-P','SPVC-M','SPVC-L']:
            mv_hat,_,_,mv_act,mv_est,mv_aux = self.mv_codec(mv_tensors)
        elif self.name == 'SPVC-R':
            mv_hat,mv_act,mv_est,mv_aux = self.mv_codec.compress_sequence(mv_tensors)
        if not self.noMeasure:
            self.meters['E-MV'].update(self.mv_codec.enc_t)
            self.meters['D-MV'].update(self.mv_codec.dec_t)
        
        # SEQ:motion compensation
        t_0 = time.perf_counter()
        MC_frame_list = [None for _ in x_tar]
        warped_frame_list = [None for _ in x_tar]
        for start in g:
            if start>bs:continue
            if start == 0:
                Y0_com = x[:1]
            else:
                Y0_com = MC_frame_list[start-1]
            for k in g[start]:
                # k = 1...6
                if k>bs:continue
                mv = mv_hat[k-1:k].cuda(1)
                MC_frame,warped_frame = motion_compensation(self.MC_network,Y0_com,mv)
                MC_frame_list[k-1] = MC_frame
                warped_frame_list[k-1] = warped_frame
        MC_frames = torch.cat(MC_frame_list,dim=0)
        warped_frames = torch.cat(warped_frame_list,dim=0)
        t_comp = time.perf_counter() - t_0
        if not self.noMeasure:
            self.meters['E-MC'].update(t_comp)
            self.meters['D-MC'].update(t_comp)
        
        # BATCH:compress residual
        res_tensors = x_tar.to(MC_frames.device) - MC_frames
        if self.name in ['SPVC','SPVC-P','SPVC-M','SPVC-L']:
            res_hat,_, _,res_act,res_est,res_aux = self.res_codec(res_tensors)
        elif self.name == 'SPVC-R':
            res_hat,res_act,res_est,res_aux = self.res_codec.compress_sequence(res_tensors)
        if not self.noMeasure:
            self.meters['E-RES'].update(self.res_codec.enc_t)
            self.meters['D-RES'].update(self.res_codec.dec_t)
        
        # reconstruction
        t_0 = time.perf_counter()
        com_frames = torch.clip(res_hat + MC_frames, min=0, max=1).to(x.device)
        if not self.noMeasure:
            self.meters['D-REC'].update(time.perf_counter() - t_0)
            
        ##### compute bits
        # estimated bits
        bpp_est = (mv_est.cuda(0) + res_est.cuda(0))/(h * w * bs)
        bpp_est = bpp_est
        # actual bits
        bpp_act = (mv_act.cuda(0) + res_act.cuda(0))/(h * w * bs)
        bpp_act = bpp_act
        # auxilary loss
        aux_loss = (mv_aux.cuda(0) + res_aux.cuda(0))/(2 * bs)
        aux_loss = aux_loss
        # calculate metrics/loss
        psnr = PSNR(x_tar, com_frames, use_list=True)
        msssim = MSSSIM(x_tar, com_frames, use_list=True)
        mc_loss = calc_loss(x_tar, MC_frames, self.r, self.use_psnr)
        warp_loss = calc_loss(x_tar, warped_frames, self.r, self.use_psnr)
        rec_loss = calc_loss(x_tar, com_frames, self.r, self.use_psnr)
        flow_loss = (l0+l1+l2+l3+l4).cuda(0)/5*1024
        img_loss = (self.r_rec*rec_loss + \
                    self.r_warp*warp_loss + \
                    self.r_mc*mc_loss + \
                    self.r_flow*flow_loss)
        img_loss = img_loss
        
        return com_frames, bpp_est, img_loss, aux_loss, bpp_act, psnr, msssim
    
    def loss(self, pix_loss, bpp_loss, aux_loss, app_loss=None):
        loss = self.r_img*pix_loss.cuda(0) + self.r_bpp*bpp_loss.cuda(0) + self.r_aux*aux_loss.cuda(0)
        if app_loss is not None:
            loss += self.r_app*app_loss.cuda(0)
        return loss
        
    def init_hidden(self, h, w):
        return None
         
# conditional coding
class SCVC(nn.Module):
    def __init__(self, name, channels=64, channels2=96, noMeasure=True):
        super(SCVC, self).__init__()
        self.name = name 
        device = torch.device('cuda')
        self.optical_flow = OpticalFlowNet()
        self.mv_codec = Coder2D('attn', in_channels=2, channels=channels, kernel=3, padding=1, noMeasure=noMeasure)
        self.MC_network = MCNet()
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
        self.ctx_decoder1 = nn.Sequential(nn.ConvTranspose2d(channels2, channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                                        GDN(channels, inverse=True),
                                        nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                                        GDN(channels, inverse=True),
                                        ResidualBlock(channels,channels),
                                        nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                                        GDN(channels, inverse=True),
                                        ResidualBlock(channels,channels),
                                        nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                                        )
        self.ctx_decoder2 = nn.Sequential(nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1),
                                        ResidualBlock(channels,channels),
                                        ResidualBlock(channels,channels),
                                        nn.Conv2d(channels, 3, kernel_size=3, stride=1, padding=1)
                                        )
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
        self.latent_codec = Coder2D('joint-attn', channels=channels2, noMeasure=noMeasure, downsample=False)
        self.channels = channels
        init_training_params(self)
        # split on multi-gpus
        self.split()
        self.noMeasure = noMeasure

    def split(self):
        self.optical_flow.cuda(0)
        self.mv_codec.cuda(0)
        self.feature_extract.cuda(0)
        self.MC_network.cuda(0)
        self.tmp_prior_encoder.cuda(1)
        self.ctx_encoder.cuda(1)
        self.latent_codec.cuda(1)
        self.ctx_decoder1.cuda(1)
        self.ctx_decoder2.cuda(1)
        
    def forward(self, x, use_psnr=True):
        # x=[B,C,H,W]: input sequence of frames
        bs, c, h, w = x[1:].size()
        
        ref_frame = x[:1]
        
        # BATCH:compute optical flow
        t_0 = time.perf_counter()
        mv_tensors, l0, l1, l2, l3, l4 = self.optical_flow(x[:-1], x[1:])
        t_flow = time.perf_counter() - t_0
        #print('Flow:',t_flow)
        
        # BATCH:compress optical flow
        t_0 = time.perf_counter()
        mv_hat,_,_,mv_act,mv_est,mv_aux = self.mv_codec(mv_tensors)
        t_mv = time.perf_counter() - t_0
        #print('MV entropy:',t_mv)
        
        # SEQ:motion compensation
        t_0 = time.perf_counter()
        MC_frame_list = []
        warped_frame_list = []
        for i in range(x.size(0)-1):
            MC_frame,warped_frame = motion_compensation(self.MC_network,ref_frame,mv_hat[i:i+1])
            ref_frame = MC_frame.detach()
            MC_frame_list.append(MC_frame)
            warped_frame_list.append(warped_frame)
        MC_frames = torch.cat(MC_frame_list,dim=0)
        warped_frames = torch.cat(warped_frame_list,dim=0)
        t_comp = time.perf_counter() - t_0
        #print('Compensation:',t_comp)
        
        t_0 = time.perf_counter()
        # BATCH:extract context
        context = self.feature_extract(MC_frames).cuda(1)
        
        # BATCH:temporal prior
        prior = self.tmp_prior_encoder(context)
        
        # contextual encoder
        y = self.ctx_encoder(torch.cat((x[1:].cuda(1), context), axis=1))
        t_ctx = time.perf_counter() - t_0
        #print('Context:',t_ctx)
        
        # entropy model
        t_0 = time.perf_counter()
        y_hat,_,_,y_act,y_est,y_aux = self.latent_codec(y, prior=prior)
        t_y = time.perf_counter() - t_0
        #print('Y entropy:',t_y)
        
        # contextual decoder
        t_0 = time.perf_counter()
        x_hat = self.ctx_decoder1(y_hat)
        x_hat = self.ctx_decoder2(torch.cat((x_hat, context), axis=1)).to(x.device)
        t_ctx_dec = time.perf_counter() - t_0
        #print('Context dec:',t_ctx_dec)
        
        # estimated bits
        bpp_est = (mv_est + y_est.to(mv_est.device))/(h * w * bs)
        # actual bits
        bpp_act = (mv_act + y_act.to(mv_act.device))/(h * w * bs)
        # auxilary loss
        aux_loss = (mv_aux + y_aux.to(mv_aux.device))/2
        # calculate metrics/loss
        psnr = PSNR(x[1:], x_hat.to(x.device), use_list=True)
        msssim = MSSSIM(x[1:], x_hat.to(x.device), use_list=True)
        mc_loss = calc_loss(x[1:], MC_frames, self.r, use_psnr)
        warp_loss = calc_loss(x[1:], warped_frames, self.r, use_psnr)
        rec_loss = calc_loss(x[1:], com_frames, self.r, use_psnr)
        flow_loss = (l0+l1+l2+l3+l4).cuda(0)/5*1024
        img_loss = self.r_warp*warp_loss + \
                    self.r_mc*mc_loss + \
                    self.r_rec*rec_loss + \
                    self.r_flow*flow_loss
        img_loss = img_loss.repeat(bs)
        
        return x_hat, bpp_est, img_loss, aux_loss, bpp_act, psnr, msssim
    
    def loss(self, pix_loss, bpp_loss, aux_loss, app_loss=None):
        loss = self.r_img*pix_loss.cuda(0) + self.r_bpp*bpp_loss.cuda(0) + self.r_aux*aux_loss.cuda(0)
        if app_loss is not None:
            loss += self.r_app*app_loss.cuda(0)
        return loss
        
    def init_hidden(self, h, w):
        return None
        
class AE3D(nn.Module):
    def __init__(self, name, noMeasure=True):
        super(AE3D, self).__init__()
        self.name = name 
        device = torch.device('cuda')
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=5, stride=(1,2,2), padding=2), 
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=5, stride=(1,2,2), padding=2), 
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            ResBlockB(),
            ResBlockB(),
            ResBlockB(),
            ResBlockB(),
            ResBlockB(),
            ResBlockA(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(128, 32, kernel_size=5, stride=(1,2,2), padding=2), 
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.latent_codec = Coder2D('rpm', channels=32, kernel=5, padding=2, noMeasure=noMeasure, downsample=False)
        self.deconv1 = nn.Sequential( 
            nn.ConvTranspose3d(32, 128, kernel_size=5, stride=(1,2,2), padding=2, output_padding=(0,1,1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.deconv2 = nn.Sequential(
            ResBlockB(),
            ResBlockB(),
            ResBlockB(),
            ResBlockB(),
            ResBlockB(),
            ResBlockA(),
        )
        self.deconv3 = nn.Sequential( 
            nn.ConvTranspose3d(128, 64, kernel_size=5, stride=(1,2,2), padding=2, output_padding=(0,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 3, kernel_size=5, stride=(1,2,2), padding=2, output_padding=(0,1,1)),
            nn.BatchNorm3d(3),
        )
        self.channels = 128
        init_training_params(self)
        self.r = 1024 # PSNR:[256,512,1024,2048] MSSSIM:[8,16,32,64]
        # split on multi-gpus
        self.split()
        self.noMeasure = noMeasure

    def split(self):
        # too much on cuda:0
        self.conv1.cuda(0)
        self.conv2.cuda(0)
        self.conv3.cuda(0)
        self.deconv1.cuda(1)
        self.deconv2.cuda(1)
        self.deconv3.cuda(1)
        self.latent_codec.cuda(0)
        
    def forward(self, x, RPM_flag=False, use_psnr=True):
            
        x = x[1:]
            
        if not self.noMeasure:
            self.enc_t = [];self.dec_t = []
            
        # x=[B,C,H,W]: input sequence of frames
        x = x.permute(1,0,2,3).contiguous().unsqueeze(0)
        bs, c, t, h, w = x.size()
        
        # encoder
        t_0 = time.perf_counter()
        x1 = self.conv1(x)
        x2 = self.conv2(x1) + x1
        latent = self.conv3(x2)
        if not self.noMeasure:
            self.enc_t += [time.perf_counter() - t_0]
        
        # entropy
        # compress each frame sequentially
        latent = latent.squeeze(0).permute(1,0,2,3).contiguous()
        latent_hat,bpp_act,bpp_est,aux_loss = self.latent_codec.compress_sequence(latent)
        latent_hat = latent_hat.permute(1,0,2,3).unsqueeze(0).contiguous()
        bpp_act = bpp_act.repeat(bs)
        bpp_est = bpp_est.repeat(bs)
        aux_loss = aux_loss.repeat(bs)
        
        # decoder
        t_0 = time.perf_counter()
        x3 = self.deconv1(latent_hat.cuda(1))
        x4 = self.deconv2(x3) + x3
        x_hat = self.deconv3(x4)
        if not self.noMeasure:
            self.dec_t += [time.perf_counter() - t_0]
        
        # reshape
        x = x.permute(0,2,1,3,4).contiguous().squeeze(0)
        x_hat = x_hat.permute(0,2,1,3,4).contiguous().squeeze(0)
        
        # calculate metrics/loss
        psnr = PSNR(x, x_hat.to(x.device), use_list=True)
        msssim = MSSSIM(x, x_hat.to(x.device), use_list=True)
        
        # calculate img loss
        img_loss = calc_loss(x, x_hat.to(x.device), self.r, use_psnr)
        img_loss = img_loss.repeat(bs)
        
        if not self.noMeasure:
            print(np.sum(self.enc_t)/bs,np.sum(self.dec_t)/bs,self.enc_t,self.dec_t)
        
        return x_hat.cuda(0), bpp_est, img_loss, aux_loss, bpp_act, psnr, msssim
    
    def init_hidden(self, h, w):
        return None
        
    def loss(self, pix_loss, bpp_loss, aux_loss, app_loss=None):
        loss = self.r_img*pix_loss.cuda(0) + self.r_bpp*bpp_loss.cuda(0) + self.r_aux*aux_loss.cuda(0)
        if app_loss is not None:
            loss += self.r_app*app_loss.cuda(0)
        return loss
        
class ResBlockA(nn.Module):
    "A ResNet-like block with the GroupNorm normalization providing optional bottle-neck functionality"
    def __init__(self, ch=128, k_size=3, stride=1, p=1):
        super(ResBlockA, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch, ch, kernel_size=k_size, stride=stride, padding=p), 
            nn.BatchNorm3d(ch),
            nn.ReLU(inplace=True),

            nn.Conv3d(ch, ch, kernel_size=k_size, stride=stride, padding=p),  
            nn.BatchNorm3d(ch),
        )
        
    def forward(self, x):
        out = self.conv(x) + x
        return out
        
class ResBlockB(nn.Module):
    def __init__(self, ch=128, k_size=3, stride=1, p=1):
        super(ResBlockB, self).__init__()
        self.conv = nn.Sequential(
            ResBlockA(ch, k_size, stride, p), 
            ResBlockA(ch, k_size, stride, p), 
            ResBlockA(ch, k_size, stride, p), 
        )
        
    def forward(self, x):
        out = self.conv(x) + x
        return out
        
def test_batch_proc(name = 'SPVC'):
    print('test',name)
    batch_size = 7
    h = w = 224
    channels = 64
    x = torch.randn(batch_size,3,h,w).cuda()
    if name in ['SPVC','SPVC_v2']:
        model = SPVC(name,channels,noMeasure=False)
    elif name == 'SCVC':
        model = SCVC(name,channels,noMeasure=False)
    elif name == 'AE3D':
        model = AE3D(name,noMeasure=False)
    elif name == 'SVC':
        model = SVC(name,channels)
    else:
        print('Not implemented.')
    import torch.optim as optim
    from tqdm import tqdm
    parameters = set(p for n, p in model.named_parameters())
    optimizer = optim.Adam(parameters, lr=1e-4)
    timer = AverageMeter()
    train_iter = tqdm(range(0,2))
    model.eval()
    for i,_ in enumerate(train_iter):
        optimizer.zero_grad()
        
        # measure start
        t_0 = time.perf_counter()
        com_frames, bpp_est, img_loss, aux_loss, bpp_act, psnr, sim = model(x)
        d = time.perf_counter() - t_0
        timer.update(d)
        # measure end
        loss = model.loss(img_loss[0],bpp_est[0],aux_loss[0])
        loss.backward()
        optimizer.step()
        
        train_iter.set_description(
            f"Batch: {i:4}. "
            f"loss: {float(loss):.2f}. "
            f"img_loss: {float(img_loss[0]):.2f}. "
            f"bits_est: {float(bpp_est[0]):.2f}. "
            f"bits_act: {float(bpp_act[0]):.2f}. "
            f"aux_loss: {float(aux_loss[0]):.2f}. "
            f"duration: {timer.avg:.3f}. ")
    showTimer(model)
            
def test_seq_proc(name='RLVC'):
    print('test',name)
    batch_size = 1
    h = w = 224
    x = torch.rand(batch_size,3,h,w).cuda()
    if name == 'DCVC' or name == 'DCVC_v2':
        model = DCVC(name,noMeasure=False)
    else:
        model = IterPredVideoCodecs(name,noMeasure=False)
    import torch.optim as optim
    from tqdm import tqdm
    parameters = set(p for n, p in model.named_parameters())
    optimizer = optim.Adam(parameters, lr=1e-4)
    timer = AverageMeter()
    hidden_states = model.init_hidden(h,w)
    train_iter = tqdm(range(0,13))
    model.eval()
    x_hat_prev = x
    for i,_ in enumerate(train_iter):
        optimizer.zero_grad()
        
        # measure start
        t_0 = time.perf_counter()
        x_hat, hidden_states, bpp_est, img_loss, aux_loss, bpp_act, p,m = model(x, x_hat_prev.detach(), hidden_states, i%13!=0)
        d = time.perf_counter() - t_0
        timer.update(d)
        # measure end
        
        x_hat_prev = x_hat
        
        loss = model.loss(img_loss,bpp_est,aux_loss)
        loss.backward()
        optimizer.step()
        
        train_iter.set_description(
            f"Batch: {i:4}. "
            f"loss: {float(loss):.2f}. "
            f"img_loss: {float(img_loss):.2f}. "
            f"bpp_est: {float(bpp_est):.2f}. "
            f"bpp_act: {float(bpp_act):.2f}. "
            f"aux_loss: {float(aux_loss):.2f}. "
            f"psnr: {float(p):.2f}. "
            f"duration: {timer.avg:.3f}. ")
    showTimer(model)
            
# integrate all codec models
# measure the speed of all codecs
# two types of test
# 1. (de)compress random images, faster
# 2. (de)compress whole datasets, record time during testing 
# need to implement 3D-CNN compression
# ***************each model can have a timer member that counts enc/dec time
# in training, counts total time, in testing, counts enc/dec time
# how to deal with big batch in training? hybrid mode
# update CNN alternatively?
    
if __name__ == '__main__':
    test_batch_proc('SVC')
    test_batch_proc('SPVC')
    #test_batch_proc('AE3D')
    #test_batch_proc('SCVC')
    #test_seq_proc('DCVC')
    #test_seq_proc('DCVC_v2')
    #test_seq_proc('DVC')
    #test_seq_proc('RLVC')