#!/usr/bin/python
# encoding: utf-8

import os
import glob
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
from codec.models import compress_video
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
        self.clip_duration = clip_duration + 9 
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
        self.cache = {} # cache for current video clip
        self.prev_video = '' # previous video name to determine whether its a whole new video
        self.last_frame = False

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()
        
        frame_idx, clip, label, bpp_est, img_loss, aux_loss, flow_loss, bpp_act, psnr, msssim = load_data_detection_from_cache(self.base_path, imgpath, self.train, self.clip_duration, self.sampling_rate, self.cache, self.dataset)
        
        # (self.duration, -1) + self.shape = (8, -1, 224, 224)
        clip = torch.cat(clip, 0).view((self.clip_duration, -1) + self.shape).permute(1, 0, 2, 3)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return (frame_idx, clip, label, bpp_est, img_loss, aux_loss, flow_loss, bpp_act, psnr, msssim)
            
    def preprocess(self, index, model_codec, max_len):
        # called by the optimization code in each iteration
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()
        im_split = imgpath.split('/')
        num_parts = len(im_split)
        im_ind = int(im_split[num_parts-1][0:5])
        cur_video = im_split[1]
        startNewClip = (cur_video != self.prev_video or self.cache['max_seen'] != im_ind-2)
        # x265/x264/MRLVC/RLVC/DVC
        # read whole video
        if startNewClip:
            self.cache = {}
            self.cache['clip'] = read_video_clip(self.base_path, imgpath, self.shape, self.dataset)
            if (self.transform is not None) and (model_codec.name not in ['x265', 'x264']):
                self.cache['clip'] = [self.transform(img).cuda() for img in self.cache['clip']]
        else:
            clip = None
        end_of_batch = compress_video(model_codec, im_ind, self.cache, startNewClip, max_len)
        self.prev_video = cur_video
        # check if the last frame of a clip or the last frame of a batch
        # it tells whether to split the processing for action detection
        if index == len(self)-1:
            self.last_frame = True and end_of_batch
        else:
            imgpath = self.lines[index+1].rstrip()
            im_split = imgpath.split('/')
            num_parts = len(im_split)
            nxt_im_ind = int(im_split[num_parts-1][0:5])
            nxt_video = im_split[1]
            self.last_frame = (cur_video != nxt_video) and end_of_batch
        
    
def read_video_clip(base_path, imgpath, shape, dataset_use='ucf24', jitter=0.2, hue=0.1, saturation=1.5, exposure=1.5):
    # load whole video as a clip for further processing
    # all frames in a video should be processed with the same augmentation or no augmentation
    # the data will be loaded from the current clip
    
    im_split = imgpath.split('/')

    img_folder = os.path.join(base_path, 'rgb-images', im_split[0], im_split[1])
    if dataset_use == 'ucf24':
        max_num = len(os.listdir(img_folder))
    elif dataset_use == 'jhmdb21':
        max_num = len(os.listdir(img_folder)) - 1

    clip = []

    for i in range(max_num):
        
        if dataset_use == 'ucf24':
            path_tmp = os.path.join(base_path, 'rgb-images', im_split[0], im_split[1] ,'{:05d}.jpg'.format(i+1))
        elif dataset_use == 'jhmdb21':
            path_tmp = os.path.join(base_path, 'rgb-images', im_split[0], im_split[1] ,'{:05d}.png'.format(i+1))

        clip.append(Image.open(path_tmp).convert('RGB'))
    
    clip = [img.resize(shape) for img in clip]
    
    return clip
    
def load_data_detection_from_cache(base_path, imgpath, train, train_dur, sample_rate, cache, dataset_use='ucf24'):
    # load 8/16 frames from video clips
    
    im_split = imgpath.split('/')
    num_parts = len(im_split)
    im_ind = int(im_split[num_parts-1][0:5])
    labpath = os.path.join(base_path, 'labels', im_split[0], im_split[1] ,'{:05d}.txt'.format(im_ind))
    
    img_folder = os.path.join(base_path, 'rgb-images', im_split[0], im_split[1])
    if dataset_use == 'ucf24':
        max_num = len(os.listdir(img_folder))
    elif dataset_use == 'jhmdb21':
        max_num = len(os.listdir(img_folder)) - 1
    
    clip_tmp = cache['clip']
    
    ### We change downsampling rate throughout training as a       ###
    ### temporal augmentation, which brings around 1-2 frame       ###
    ### mAP. During test time it is set to cfg.DATA.SAMPLING_RATE. ###
    d = sample_rate
        
    clip = []
    for i in reversed(range(train_dur)):
        # make it as a loop
        i_temp = im_ind - i * d - 1
        if i_temp < 0:
            i_temp = 0
        elif i_temp > max_num-1:
            i_temp = max_num-1
        clip.append(clip_tmp[i_temp])
        
    _,h,w = clip[0].shape
    label = torch.zeros(50*5)
    try:
        tmp = torch.from_numpy(read_truths_args(labpath, 8.0/w).astype('float32'))
    except Exception:
        tmp = torch.zeros(1,5)

    tmp = tmp.view(-1)
    tsz = tmp.numel()

    if tsz > 50*5:
        label = tmp[0:50*5]
    elif tsz > 0:
        label[0:tsz] = tmp
    
    if train:
        return im_ind, clip, label, cache['bpp_est'][im_ind-1], cache['img_loss'][im_ind-1], cache['aux'][im_ind-1], \
                cache['flow_loss'][im_ind-1], cache['bpp_act'][im_ind-1], cache['psnr'][im_ind-1], cache['msssim'][im_ind-1]
    else:
        return im_split[0] + '_' +im_split[1] + '_' + im_split[2], clip, label, cache['bpp_est'][im_ind-1], cache['img_loss'][im_ind-1], \
                cache['aux'][im_ind-1], cache['flow_loss'][im_ind-1], cache['bpp_act'][im_ind-1], cache['psnr'][im_ind-1], cache['msssim'][im_ind-1]