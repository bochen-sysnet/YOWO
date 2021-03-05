import cv2
import numpy as np
import time
import torch
import glob

from torchvision import transforms
from torch.utils.data import Dataset
from utils import *
from eval_results import *
from cfg import parse_cfg
from collections import OrderedDict
from PIL import  ImageChops
# todo
# change quality in a tile

dataset = 'ucf101-24'

def get_edge_feature(frame, edge_blur_rad=11, edge_blur_var=0, edge_canny_low=101, edge_canny_high=255):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	start = time.perf_counter()
	blur = cv2.GaussianBlur(gray, (edge_blur_rad, edge_blur_rad), edge_blur_var)
	edge = cv2.Canny(blur, edge_canny_low, edge_canny_high)
	end = time.perf_counter()
	return edge, end-start
    

def get_KAZE_feature(frame):
	alg = cv2.KAZE_create()
	start = time.perf_counter()
	kps = alg.detect(frame)
	end = time.perf_counter()
	kps = sorted(kps, key=lambda x: -x.response)[:32]
	points = [p.pt for p in kps]
	return points, end-start

def get_harris_corner(frame):
	img = frame.copy()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	start = time.perf_counter()
	dst = cv2.cornerHarris(gray,2,3,0.04)
	end = time.perf_counter()

	# Threshold for an optimal value, it may vary depending on the image.
	dst[dst>0.01*dst.max()]=[255]
	dst[dst<255]=[0]
	return dst, end-start

def get_GFTT(frame):
	img = frame.copy()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	start = time.perf_counter()
	corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
	end = time.perf_counter()
	if corners is not None:
		corners = np.int0(corners) 
		points = [i.ravel() for i in corners]
	else:
		points = []
	return points, end-start

# pip install opencv-python==3.4.2.16
# pip install opencv-contrib-python==3.4.2.16
def get_SIFT(frame):
	img = frame.copy()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	sift = cv2.xfeatures2d.SIFT_create()
	start = time.perf_counter()
	kps = sift.detect(gray,None)
	end = time.perf_counter()
	points = [p.pt for p in kps]
	return points, end-start

def get_SURF(frame):
	img = frame.copy()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	surf = cv2.xfeatures2d.SURF_create()
	start = time.perf_counter()
	kps = surf.detect(gray,None)
	end = time.perf_counter()
	points = [p.pt for p in kps]
	return points, end-start

def get_FAST(frame):
	img = frame.copy()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	start = time.perf_counter()
	fast = cv2.FastFeatureDetector_create(threshold=50)
	kps = fast.detect(img,None)
	end = time.perf_counter()
	points = [p.pt for p in kps]
	return points, end-start

def get_STAR(frame):
	img = frame.copy()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# Initiate STAR detector
	star = cv2.xfeatures2d.StarDetector_create()

	# find the keypoints with STAR
	start = time.perf_counter()
	kps = star.detect(img,None)
	end = time.perf_counter()
	points = [p.pt for p in kps]
	return points, end-start

def get_ORB(frame):
	img = frame.copy()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	orb = cv2.ORB_create()
	start = time.perf_counter()
	kps = orb.detect(img,None)
	end = time.perf_counter()
	points = [p.pt for p in kps]
	return points, end-start

def count_point_in_ROI(ROI, points):
	counter = 0
	for px,py in points:
		inROI = False
		x1,y1,x2,y2 = ROI
		if x1<=px and x2>px and y1<=py and y2>py:
			inROI = True
		if inROI:counter += 1
	return counter

def count_mask_in_ROI(ROI, mp):
	total_pts = np.count_nonzero(mp)
	x1,y1,x2,y2 = ROI
	mp[y1:y2,x1:x2] = 0
	nonroi_pts = np.count_nonzero(mp)
	return total_pts-nonroi_pts

def ROI_area(ROIs,w,h):
	im = np.zeros((h,w),dtype=np.uint8)
	for x1,y1,x2,y2 in ROIs:
		im[y1:y2,x1:x2] = 1
	roi = np.count_nonzero(im)
	return roi

def path_to_disturbed_image(pil_image, label, r_in, r_out):
	b,g,r = cv2.split(np.array(pil_image))
	np_img = cv2.merge((b,g,r))
	np_img = region_disturber(np_img,label, r_in, r_out)
	pil_image = Image.fromarray(np_img)
	return pil_image

# change quality of non-ROI
# r_in is the scaled ratio of ROIs
# r_out is the scaled ratio of the whole image
def region_disturber(image,label,r_in,r_out):
	# get the original content from ROI
	# downsample rest, then upsample
	# put roi back
	w,h = 320,240
	dsize_out = (int(w*r_out),int(h*r_out))
	crops = []
	for _,cx,cy,imgw,imgh  in label:
		cx=int(cx*320);cy=int(cy*240);imgw=int(imgw*320);imgh=int(imgh*320)
		x1=max(cx-imgw//2,0);x2=min(cx+imgw//2,w);y1=max(cy-imgw//2,0);y2=min(cy+imgw//2,h)
		crop = image[y1:y2,x1:x2]
		if r_in<1:
			dsize_in = (int((x2-x1)*r_in),int((y2-y1)*r_in))
			crop_d = cv2.resize(crop, dsize=dsize_in, interpolation=cv2.INTER_LINEAR)
			crop = cv2.resize(crop_d, dsize=(x2-x1,y2-y1), interpolation=cv2.INTER_LINEAR)
		crops.append((x1,y1,x2,y2,crop))
	if r_out<1:
		# downsample
		downsample = cv2.resize(image, dsize=dsize_out, interpolation=cv2.INTER_LINEAR)
		# upsample
		image = cv2.resize(downsample, dsize=(w,h), interpolation=cv2.INTER_LINEAR)
	for x1,y1,x2,y2,crop  in crops:
		image[y1:y2,x1:x2] = crop
	
	return image

# analyze static and motion feature points
# need to count the number of features ROI and not in ROI
# calculate the density
# should compare  
# percentage of features/percentage of area
def analyzer(images,targets):
	means = (104, 117, 123)
	w,h = 1024,512
	cnt = 0
	avg_dens1,avg_dens2 = np.zeros(9,dtype=np.float64),np.zeros(9,dtype=np.float64)
	for image,label in zip(images,targets):
		for ch in range(0,3):
			image[ch,:,:] += means[2-ch]
		rgb_frame = image.permute(1,2,0).numpy().astype(np.uint8)
		bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
		label = label.numpy()
		ROIs  =[]
		for x1,y1,x2,y2,_  in label:
			x1*=1024;x2*=1024;y1*=512;y2*=512
			ROIs.append([int(x1),int(y1),int(x2),int(y2)])
		# edge diff
		edge, _ = get_edge_feature(bgr_frame)
		# *kaze feat
		kaze, _ = get_KAZE_feature(bgr_frame)
		# harris corner
		hc, _ = get_harris_corner(bgr_frame)
		# GFTT
		gftt, _ = get_GFTT(bgr_frame)
		# *SIFT
		sift, _ = get_SIFT(bgr_frame)
		# *SURF
		surf, _ = get_SURF(bgr_frame)
		# FAST
		fast, _ = get_FAST(bgr_frame)
		# STAR
		star, _ = get_STAR(bgr_frame)
		# ORB
		orb, _ = get_ORB(bgr_frame)

		point_features = [gftt, kaze, sift, surf, fast, star, orb]
		map_features = [edge,hc]
		in_roi,out_roi = ROI_area(ROIs,w,h)
		density1,density2 = [],[]
		for mp in map_features:
			c1,c2 = count_mask_in_ROI(ROIs,mp)
			density1+=['{:0.6f}'.format(c1*1.0/in_roi)]
			density2+=['{:0.6f}'.format(c2*1.0/out_roi)]
		for points in point_features:
			c1,c2 = count_point_in_ROI(ROIs,points)
			density1+=['{:0.6f}'.format(c1*1.0/in_roi)]
			density2+=['{:0.6f}'.format(c2*1.0/out_roi)]

		for ch in range(0,3):
			image[ch,:,:] -= means[2-ch]

		cnt += 1
		avg_dens1 += np.array(density1,dtype=np.float64)
		avg_dens2 += np.array(density2,dtype=np.float64)
	return avg_dens1/4,avg_dens2/4

class LRU(OrderedDict):

	def __init__(self, maxsize=128):
		self.maxsize = maxsize
		super().__init__()

	def __getitem__(self, key):
		value = super().__getitem__(key)
		self.move_to_end(key)
		return value

	def __setitem__(self, key, value):
		if key in self:
			self.move_to_end(key)
		super().__setitem__(key, value)
		if len(self) > self.maxsize:
			oldest = next(iter(self))
			del self[oldest]

def tile_disturber(image, C_param):
	# analyze features in image
	feat_start = time.perf_counter()
	bgr_frame = np.array(image)
	# edge diff
	# edge, _ = get_edge_feature(bgr_frame)
	# harris corner
	hc, _ = get_harris_corner(bgr_frame)
	# GFTT
	gftt, _ = get_GFTT(bgr_frame)
	# FAST
	fast, _ = get_FAST(bgr_frame)
	# STAR
	star, _ = get_STAR(bgr_frame)
	# ORB
	orb, _ = get_ORB(bgr_frame)

	calc_start = time.perf_counter()
	point_features = [gftt, fast, star, orb]
	map_features = []
	num_features = len(point_features) + len(map_features)
	# divide [320,240] image to 4*3 tiles
	ROIs = []
	num_w, num_h = 4,3
	tilew,tileh = 320//num_w,240//num_h
	for row in range(num_w):
		for col in range(num_h):
			x1 = col*tilew; x2 = (col+1)*tilew; y1 = row*tileh; y2 = (row+1)*tileh
			ROIs.append([x1,y1,x2,y2])
	counts = np.zeros((num_w*num_h,num_features))
	for roi_idx,ROI in enumerate(ROIs):
		roi_start = time.perf_counter()
		feat_idx = 0
		for mf in map_features:
			c = count_mask_in_ROI(ROI,mf)
			counts[roi_idx,feat_idx] = c
			feat_idx += 1
		for pf in point_features:
			c = count_point_in_ROI(ROI,pf)
			counts[roi_idx,feat_idx] = c
			feat_idx += 1
		roi_end = time.perf_counter()

	# weight of different features
	weights = C_param[:num_features]
	# (0,1) indicating the total quality after compression
	A = C_param[num_features]
	# order to adjust the concentration of the  scores
	k = int(C_param[num_features+1])
	order_choices = [1./3,1./2,1,2,3]
	normalized_score = counts/(np.sum(counts,axis=0)+1e-6)
	weights /= (np.sum(weights)+1e-6)
	# ws of all tiles sum up to 1
	weighted_scores = np.matmul(normalized_score,weights)
	print(C_param)
	print(weighted_scores)
	# the weight is more valuable when its value is higher
	weighted_scores = weighted_scores**order_choices[k]
	weighted_scores /= (np.max(weighted_scores)+1e-6)
	# quality of each tile?
	quality = A*weighted_scores
	print(quality)

	tile_sizes = [(int(np.rint(tilew*max(r,0.1))),int(np.rint(tileh*max(r,0.1)))) for r in quality]

	# not used for training,but can be used for 
	# ploting the pareto front
	compressed_size = 0
	tile_size = tilew * tileh
	for roi,dsize in zip(ROIs,tile_sizes):
		if dsize == (tilew,tileh):
			compressed_size += tilew*tileh
			continue
		x1,y1,x2,y2 = roi
		crop = bgr_frame[y1:y2,x1:x2].copy()
		if dsize[0]==0 or dsize[1]==0:
			bgr_frame[y1:y2,x1:x2] = [0]
		else:
			crop_d = cv2.resize(crop, dsize=dsize, interpolation=cv2.INTER_LINEAR)
			crop = cv2.resize(crop_d, dsize=(tilew,tileh), interpolation=cv2.INTER_LINEAR)
			compressed_size += dsize[0]*dsize[1]
			bgr_frame[y1:y2,x1:x2] = crop
	pil_image = Image.fromarray(bgr_frame)

	feat_end = time.perf_counter()
	# print(img_index,feat_end-feat_start)
	return image,compressed_size,tile_sizes

def JPEG_disturber(image, C_param):
	return image

# define a class for transformation
class Transformer:
	def __init__(self,name):
		# need a dict as buffer to store transformed image of a range
		self.name = name
		self.lru = LRU(16) # size of clip

	def transform(self, image=None, label=None, C_param=None, img_index=None):
		# Rule 1: more feature more quality
		# Rule 2: some features are more important
		if img_index in self.lru: return self.lru[img_index]

		image,_,tile_sizes = tile_disturber(image, C_param)
		print(img_index,tile_sizes)

		self.lru[img_index] = image
		return image

if __name__ == "__main__":
    # img = cv2.imread('/home/bo/research/dataset/ucf24/compressed/000000.jpg')
    img = cv2.imread('/home/bo/research/dataset/ucf24/rgb-images/Basketball/v_Basketball_g01_c01/00001.jpg')
    