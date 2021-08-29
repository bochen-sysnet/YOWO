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
            frame_idx, clip, label, bpp_est, loss = load_data_detection_from_cache(self.base_path, imgpath, self.train, self.clip_duration, self.sampling_rate, self.cache, self.dataset)

        else: # For Testing
            frame_idx, clip, label, bpp_est, loss, bpp_act, metrics = load_data_detection_from_cache(self.base_path, imgpath,  self.train, self.clip_duration, self.sampling_rate, self.cache, self.dataset)
        
        # (self.duration, -1) + self.shape = (8, -1, 224, 224)
        clip = torch.cat(clip, 0).view((self.clip_duration, -1) + self.shape).permute(1, 0, 2, 3)

        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.train:
            return (frame_idx, clip, label, bpp_est, loss)
        else:
            return (frame_idx, clip, label, bpp_est, loss, bpp_act, metrics)
            
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
        # or if the index is not continuous
        if cur_video != self.prev_video or self.cache['max_idx'] != im_ind-2:
            # read raw video clip
            clip,misc = read_video_clip(self.base_path, imgpath, self.train, self.clip_duration, self.sampling_rate, self.shape, self.dataset)
            # frame shape
            h,w = 224,224
            # create cache
            self.cache = {}
            clip = [img.resize(self.shape) for img in clip]
            if self.transform is not None:
                clip = [self.transform(img).cuda() for img in clip]
            self.cache['clip'] = clip
            self.cache['misc'] = misc
            self.cache['bpp_est'] = {}
            self.cache['loss'] = {}
            self.cache['bpp_act'] = {}
            self.cache['metrics'] = {}
            self.cache['max_idx'] = im_ind-1
            # compress from the first frame of the first clip to the current frame
            Iframe_idx = (im_ind - (self.clip_duration-1) * self.sampling_rate - 1)//10*10
            for i in range(Iframe_idx,im_ind):
                Y1_raw = self.cache['clip'][i].unsqueeze(0)
                if (i-Iframe_idx)%10 == 0:
                    # compressing the I frame 
                    if self.train:
                        Y1_com, bpp_est, img_loss =\
                            model_codec(None, Y1_raw, None, None, None, False, True)
                    else:
                        Y1_com, bpp_est, img_loss, bpp_act, metrics =\
                            model_codec(None, Y1_raw, None, None, None, False, True)
                elif (i-Iframe_idx)%10 == 1:
                    # init hidden states
                    rae_hidden, rpm_hidden = init_hidden(h,w)
                    latent = torch.zeros(1,8,4,4).cuda()
                    # compress for first P frame
                    if self.train:
                        Y1_com,rae_hidden,rpm_hidden,latent,bpp_est,img_loss = \
                            model_codec(Y0_com, Y1_raw, rae_hidden, rpm_hidden, latent, False, False)
                    else:
                        Y1_com,rae_hidden,rpm_hidden,latent,bpp_est,img_loss, bpp_act, metrics = \
                            model_codec(Y0_com, Y1_raw, rae_hidden, rpm_hidden, latent, False, False)
                    self.cache['rae_hidden'] = rae_hidden.detach()
                    self.cache['rpm_hidden'] = rpm_hidden.detach()
                    self.cache['latent'] = latent.detach()
                else:
                    # compress for later P frames
                    if self.train:
                        Y1_com, rae_hidden,rpm_hidden,latent,bpp_est,img_loss = \
                            model_codec(Y0_com, Y1_raw, self.cache['rae_hidden'], self.cache['rpm_hidden'], self.cache['latent'], True, False)
                    else:
                        Y1_com, rae_hidden,rpm_hidden,latent,bpp_est,img_loss, bpp_act, metrics = \
                            model_codec(Y0_com, Y1_raw, self.cache['rae_hidden'], self.cache['rpm_hidden'], self.cache['latent'], True, False)
                    self.cache['rae_hidden'] = rae_hidden.detach()
                    self.cache['rpm_hidden'] = rpm_hidden.detach()
                    self.cache['latent'] = latent.detach()
                print(torch.mean(Y1_raw),Y1_raw)
                print(torch.mean(Y1_com),Y1_com)
                self.cache['clip'][i] = Y1_com.detach().squeeze(0)
                self.cache['loss'][i] = img_loss
                self.cache['bpp_est'][i] = bpp_est
                if not self.train:
                    self.cache['metrics'][i] = metrics
                    self.cache['bpp_act'][i] = bpp_act
                Y0_com = Y1_com
        else:
            assert im_ind-2 == self.cache['max_idx'], 'index error of the non-first frame'
            Y0_com = self.cache['clip'][im_ind-2].unsqueeze(0)
            Y1_raw = self.cache['clip'][im_ind-1].unsqueeze(0)
            # frame shape
            _,h,w = self.cache['clip'][0].shape
            if (im_ind-1)%10 == 0:
                # compressing the I frame 
                if self.train:
                    Y1_com, bpp_est, img_loss =\
                        model_codec(None, Y1_raw, None, None, None, False, True)
                else:
                    Y1_com, bpp_est, img_loss, bpp_act, metrics =\
                        model_codec(None, Y1_raw, None, None, None, False, True)
            elif (im_ind-1)%10 == 1:
                #### initialization for the first P frame
                # init hidden states
                rae_hidden, rpm_hidden = init_hidden(h,w)
                # previous compressed motion vector and residual
                latent = torch.zeros(1,8,4,4).cuda()
                # compress for first P frame
                if self.train:
                    Y1_com,rae_hidden,rpm_hidden,latent,bpp_est,img_loss = \
                        model_codec(Y0_com, Y1_raw, rae_hidden, rpm_hidden, latent, False, False)
                else:
                    Y1_com,rae_hidden,rpm_hidden,latent,bpp_est,img_loss, bpp_act, metrics = \
                        model_codec(Y0_com, Y1_raw, rae_hidden, rpm_hidden, latent, False, False)
                self.cache['rae_hidden'] = rae_hidden.detach()
                self.cache['rpm_hidden'] = rpm_hidden.detach()
                self.cache['latent'] = latent.detach()
            else:
                # compress for later P frames
                if self.train:
                    Y1_com, rae_hidden,rpm_hidden,latent,bpp_est,img_loss = \
                        model_codec(Y0_com, Y1_raw, self.cache['rae_hidden'], self.cache['rpm_hidden'], self.cache['latent'], True, False)
                else:
                    Y1_com, rae_hidden,rpm_hidden,latent,bpp_est,img_loss, bpp_act, metrics = \
                        model_codec(Y0_com, Y1_raw, self.cache['rae_hidden'], self.cache['rpm_hidden'], self.cache['latent'], True, False)
                self.cache['rae_hidden'] = rae_hidden.detach()
                self.cache['rpm_hidden'] = rpm_hidden.detach()
                self.cache['latent'] = latent.detach()
            self.cache['clip'][im_ind-1] = Y1_com.detach().squeeze(0)
            self.cache['loss'][im_ind-1] = img_loss
            self.cache['bpp_est'][im_ind-1] = bpp_est
            if not self.train:
                self.cache['metrics'][im_ind-1] = metrics
                self.cache['bpp_act'][im_ind-1] = bpp_act
            self.cache['max_idx'] = im_ind-1
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