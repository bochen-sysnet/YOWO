#!/usr/bin/python
# encoding: utf-8

import os
import glob
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image

from datasets.clip import *


class UCF_JHMDB_Dataset(Dataset):

    # clip duration = 8, i.e, for each time 8 frames are considered together
    def __init__(self, base, root, dataset='ucf24', shape=None,
                 transform=None, target_transform=None, 
                 train=False, clip_duration=16, sampling_rate=1):
        with open(root, 'r') as file:
            self.lines = file.readlines()

        self.base_path = base
        self.dataset = dataset
        self.nSamples  = len(self.lines)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.shape = shape
        self.clip_duration = clip_duration + 9 # for GOP=10
        self.sampling_rate = sampling_rate

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()

        if self.train: # For Training
            jitter = 0.2
            hue = 0.1
            saturation = 1.5 
            exposure = 1.5

            frame_idx, clip, label = load_data_detection(self.base_path, imgpath,  self.train, self.clip_duration, self.sampling_rate, self.shape, self.dataset, jitter, hue, saturation, exposure)

        else: # For Testing
            frame_idx, clip, label = load_data_detection(self.base_path, imgpath, False, self.clip_duration, self.sampling_rate, self.shape, self.dataset)
            clip = [img.resize(self.shape) for img in clip]

        if self.transform is not None:
            clip = [self.transform(img) for img in clip]

        # (self.duration, -1) + self.shape = (8, -1, 224, 224)
        clip = torch.cat(clip, 0).view((self.clip_duration, -1) + self.shape).permute(1, 0, 2, 3)

        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.train:
            return (frame_idx, clip, label)
        else:
            return (frame_idx, clip, label)
            
class UCF_JHMDB_Dataset_codec(Dataset):

    # clip duration = 8, i.e, for each time 8 frames are considered together
    def __init__(self, base, root, dataset='ucf24', shape=None,
                 transform=None, target_transform=None, 
                 train=False, clip_duration=16, sampling_rate=1):
        with open(root, 'r') as file:
            self.lines = file.readlines()

        self.base_path = base
        self.dataset = dataset
        self.nSamples  = len(self.lines)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.shape = shape
        self.clip_duration = clip_duration
        self.sampling_rate = sampling_rate
        self.cache = None # cache for current video clip
        self.prev_video = '' # previous video name to determine whether its a whole new video

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()
        
        if self.train: # For Training
            frame_idx, clip, label, bpp, loss = load_data_detection_from_cache(self.base_path, imgpath,  self.train, self.clip_duration, self.sampling_rate, self.cache, self.dataset)

        else: # For Testing
            frame_idx, clip, label, bpp, loss = load_data_detection_from_cache(self.base_path, imgpath,  self.train, self.clip_duration, self.sampling_rate, self.cache, self.dataset)
        
        # (self.duration, -1) + self.shape = (8, -1, 224, 224)
        clip = torch.cat(clip, 0).view((self.clip_duration, -1) + self.shape).permute(1, 0, 2, 3)

        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.train:
            return (frame_idx, clip, label, bpp, loss)
        else:
            return (frame_idx, clip, label, bpp, loss)
            
    def preprocess(self, index, model_codec):
        # called by the optimization code in each iteration
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()
        im_split = imgpath.split('/')
        num_parts = len(im_split)
        im_ind = int(im_split[num_parts-1][0:5])
        cur_video = im_split[1]
        # if this is a whole new video, load whole clip and compress the batch
        # also additional frames need to be compressed for the first clip
        # else just compress the batch
        if cur_video != self.prev_video:
            # read raw video clip
            clip,misc = read_video_clip(self.base_path, imgpath,  self.train, self.clip_duration, self.sampling_rate, self.shape, self.dataset)
            # frame shape
            h,w = clip[0].width,clip[0].height
            # create cache
            self.cache = {}
            if not self.train:
                clip = [img.resize(self.shape) for img in clip]
            if self.transform is not None:
                clip = [self.transform(img).cuda() for img in clip]
            self.cache['clip'] = clip
            self.cache['misc'] = misc
            self.cache['bpp_est'] = []
            self.cache['loss'] = []
            self.cache['bpp_act'] = []
            self.cache['metrics'] = []
            # compress from the first frame of the first clip to the current frame
            Iframe_idx = (im_ind - (self.clip_duration-1) * self.sampling_rate - 1)//10*10
            for i in range(im_ind):
                if i<Iframe_idx:
                    # fill ignored frame data with 0
                    self.cache['loss'].append(0)
                    self.cache['bpp_est'].append(0)
                    continue
                Y1_raw = self.cache['clip'][i].unsqueeze(0)
                if (i-Iframe_idx)%10 == 0:
                    # init hidden states
                    rae_hidden, rpm_hidden = init_hidden(h,w)
                    latent = None
                    # compressing the I frame 
                    Y1_com, bpp_est, img_loss =\
                        model_codec(None, Y1_raw, None, None, None, False, True)
                elif (i-Iframe_idx)%10 == 1:
                    # init hidden states
                    rae_hidden, rpm_hidden = init_hidden(h,w)
                    latent = None
                    # compress for first P frame
                    Y1_com,rae_hidden,rpm_hidden,latent,bpp_est,img_loss = \
                        model_codec(Y0_com, Y1_raw, rae_hidden, rpm_hidden, latent, False, False)
                else:
                    # compress for later P frames
                    Y1_com, rae_hidden,rpm_hidden,latent,bpp_est,img_loss = \
                        model_codec(Y0_com, Y1_raw, rae_hidden, rpm_hidden, latent, True, False)
                self.cache['clip'][i] = Y1_com.detach().squeeze(0)
                self.cache['loss'].append(img_loss)
                self.cache['bpp_est'].append(bpp_est)
                Y0_com = Y1_com
            self.cache['rae_hidden'] = rae_hidden.detach()
            self.cache['rpm_hidden'] = rpm_hidden.detach()
            self.cache['latent'] = latent.detach()
        else:
            assert im_ind >= 2, 'index error of the non-first frame'
            Y0_com = self.cache['clip'][im_ind-2].unsqueeze(0)
            Y1_raw = self.cache['clip'][im_ind-1].unsqueeze(0)
            # frame shape
            _,h,w = self.cache['clip'][0].shape
            # intermediate states
            rae_hidden = self.cache['rae_hidden']
            rpm_hidden = self.cache['rpm_hidden']
            latent = self.cache['latent']
            if (im_ind-1)%10 == 0:
                # no need for Y0_com, latent, hidden when compressing the I frame 
                Y1_com, bpp_est, img_loss =\
                    model_codec(None, Y1_raw, None, None, None, False, True)
            elif (im_ind-1)%10 == 1:
                #### initialization for the first P frame
                # init hidden states
                rae_hidden, rpm_hidden = init_hidden(h,w)
                # previous compressed motion vector and residual
                latent = None
                # compress for first P frame
                Y1_com,rae_hidden,rpm_hidden,latent,bpp_est,img_loss = \
                    model_codec(Y0_com, Y1_raw, rae_hidden, rpm_hidden, latent, False, False)
            else:
                # compress for later P frames
                Y1_com, rae_hidden,rpm_hidden,latent,bpp_est,img_loss = \
                    model_codec(Y0_com, Y1_raw, rae_hidden, rpm_hidden, latent, True, False)
            self.cache['clip'][im_ind-1] = Y1_com.detach().squeeze(0)
            self.cache['loss'].append(img_loss)
            self.cache['bpp_est'].append(bpp_est)
            self.cache['rae_hidden'] = rae_hidden.detach()
            self.cache['rpm_hidden'] = rpm_hidden.detach()
            self.cache['latent'] = latent.detach()
        self.prev_video = cur_video

def init_hidden(h,w):
    # mv_hidden = torch.split(torch.zeros(4,128,h//4,w//4).cuda(),1)
    # res_hidden = torch.split(torch.zeros(4,128,h//4,w//4).cuda(),1)
    # hidden_rpm_mv = torch.split(torch.zeros(2,128,h//16,w//16).cuda(),1)
    # hidden_rpm_res = torch.split(torch.zeros(2,128,h//16,w//16).cuda(),1)
    # hidden = (mv_hidden, res_hidden, hidden_rpm_mv, hidden_rpm_res)
    rae_hidden = torch.zeros(1,128*8,h//4,w//4).cuda()
    rpm_hidden = torch.zeros(1,128*4,h//16,w//16).cuda()
    return rae_hidden, rpm_hidden