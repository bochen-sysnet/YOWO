import torch
import torch.nn as nn
import numpy as np
import os
import glob
from opts import parse_opts
from cfg import parse_cfg
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.io import loadmat
from model import YOWO
from utils import *
from eval_results import *
import cv2, imutils

opt = parse_opts()

dataset = opt.dataset
assert dataset == 'ucf101-24' or dataset == 'jhmdb-21', 'invalid dataset'

use_train     = opt.use_train
sample_thresh = opt.sample_thresh
datacfg       = opt.data_cfg
cfgfile       = opt.cfg_file
gt_file       = 'finalAnnots.mat' # Necessary for ucf

data_options  = read_data_cfg(datacfg)
net_options   = parse_cfg(cfgfile)[0]
loss_options  = parse_cfg(cfgfile)[1]

base_path     = data_options['base']
testlist      = os.path.join(base_path, 'testlist_video.txt') if use_train==0 else os.path.join(base_path, 'trainlist_video.txt')

clip_duration = int(net_options['clip_duration'])
anchors       = loss_options['anchors'].split(',')
anchors       = [float(i) for i in anchors]
num_anchors   = int(loss_options['num'])
num_classes   = opt.n_classes

# Test parameters
conf_thresh   = 0.005
nms_thresh    = 0.4
eps           = 1e-5

use_cuda = True
kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}


# Create model
model       = YOWO(opt)
model       = model.cuda()
model       = nn.DataParallel(model, device_ids=None) # in multi-gpu case
print(model)

# Load resume path 
if opt.resume_path:
    print("===================================================================")
    print('loading checkpoint {}'.format(opt.resume_path))
    checkpoint = torch.load(opt.resume_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("===================================================================")



def get_clip(root, imgpath, train_dur, dataset):
    im_split = imgpath.split('/')
    num_parts = len(im_split)
    class_name = im_split[-3]
    file_name = im_split[-2]
    im_ind = int(im_split[num_parts - 1][0:5])
    if dataset == 'ucf101-24':
        img_name = os.path.join(class_name, file_name, '{:05d}.jpg'.format(im_ind))
        labpath = os.path.join(base_path, 'labels', class_name, file_name ,'{:05d}.txt'.format(im_ind))
    elif dataset == 'jhmdb-21':
        img_name = os.path.join(class_name, file_name, '{:05d}.png'.format(im_ind))
        labpath = os.path.join(base_path, 'labels', class_name + '_' + file_name + '_' + '{:05d}.txt'.format(im_ind))
        
    img_folder = os.path.join(base_path, 'rgb-images', class_name, file_name)
    max_num = len(os.listdir(img_folder))
    clip = [] 

    for i in reversed(range(train_dur)):
        i_img = im_ind - i * 1
        if i_img < 1:
            i_img = 1
        elif i_img > max_num:
            i_img = max_num

        if dataset == 'ucf101-24':
            path_tmp = os.path.join(base_path, 'rgb-images', class_name, file_name, '{:05d}.jpg'.format(i_img))
        elif dataset == 'jhmdb-21':
            path_tmp = os.path.join(base_path, 'rgb-images', class_name, file_name, '{:05d}.png'.format(i_img))      
        clip.append(Image.open(path_tmp).convert('RGB'))

    label = torch.zeros(50 * 5)
    try:
        tmp = torch.from_numpy(read_truths_args(labpath, 8.0 / clip[0].width).astype('float32'))
    except Exception:
        tmp = torch.zeros(1, 5)

    tmp = tmp.view(-1)
    tsz = tmp.numel()

    if tsz > 50 * 5:
        label = tmp[0:50 * 5]
    elif tsz > 0:
        label[0:tsz] = tmp

    return clip, label, img_name

class testData(Dataset):
    def __init__(self, root, shape=None, transform=None, clip_duration=16):

        self.root = root
        if dataset == 'ucf101-24':
            self.label_paths = sorted(glob.glob(os.path.join(root, '*.jpg')))
        elif dataset == 'jhmdb-21':
            self.label_paths = sorted(glob.glob(os.path.join(root, '*.png')))

        self.shape = shape
        self.transform = transform
        self.clip_duration = clip_duration

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        label_path = self.label_paths[index]

        clip, label, img_name = get_clip(self.root, label_path, self.clip_duration, dataset)
        clip = [img.resize(self.shape) for img in clip]

        if self.transform is not None:
            clip = [self.transform(img) for img in clip]

        clip = torch.cat(clip, 0).view((self.clip_duration, -1) + self.shape).permute(1, 0, 2, 3)

        return clip, label, img_name
    
def get_pixel_feature(frame):
    return frame

def get_edge_feature(frame, edge_blur_rad=11, edge_blur_var=0, edge_canny_low=101, edge_canny_high=255):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (edge_blur_rad, edge_blur_rad), edge_blur_var)
    edge = cv2.Canny(blur, edge_canny_low, edge_canny_high)
    return edge

def get_area_feature(frame, area_blur_rad=11, area_blur_var=0):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (area_blur_rad, area_blur_rad), area_blur_var)
    return blur

def calc_pixel_diff(frame, prev_frame, pixel_thresh_low_bound=21):
    if prev_frame is None: return 0
    total_pixels = frame.shape[0] * frame.shape[1]
    frame_diff = cv2.absdiff(frame, prev_frame)
    frame_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.threshold(frame_diff, pixel_thresh_low_bound, 255, cv2.THRESH_BINARY)[1]
    changed_pixels = cv2.countNonZero(frame_diff)
    fraction_changed = changed_pixels / total_pixels
    return fraction_changed
    
def calc_edge_diff(frame, prev_frame, edge_thresh_low_bound=21):
    if prev_frame is None: return 0
    total_pixels = frame.shape[0] * frame.shape[1]
    frame_diff = cv2.absdiff(frame, prev_frame)
    frame_diff = cv2.threshold(frame_diff,edge_thresh_low_bound, 255,
                               cv2.THRESH_BINARY)[1]
    changed_pixels = cv2.countNonZero(frame_diff)
    fraction_changed = changed_pixels / total_pixels
    return fraction_changed

def calc_area_diff(frame, prev_frame, area_thresh_low_bound=21):
    if prev_frame is None: return 0
    total_pixels = frame.shape[0] * frame.shape[1]
    frame_delta = cv2.absdiff(frame, prev_frame)
    thresh = cv2.threshold(frame_delta, area_thresh_low_bound, 255,
                           cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    if not contours:
        return 0.0
    return max([cv2.contourArea(c) / total_pixels for c in contours])
    
    
def extract_n_filter_one_batch(batch, prev_frame):
    batch = batch.squeeze(2)
    last_frame = None
    pixel_feat_list = []
    edge_feat_list = []
    area_feat_list = []
    # get all features
    if prev_frame is None:
        pixel_feat_list.append(None)
        edge_feat_list.append(None)
        area_feat_list.append(None)
    else:
        pixel_feat_list.append(get_pixel_feature(prev_frame))
        edge_feat_list.append(get_edge_feature(prev_frame))
        area_feat_list.append(get_area_feature(prev_frame))
    for i in range(batch.size(0)):
        rgb_frame = batch[i,:,:,:].squeeze(0).permute(1, 2, 0).numpy()
        rgb_frame = (rgb_frame*255).astype(np.uint8)
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        # pixel diff
        pixel_feat = get_pixel_feature(bgr_frame)
        pixel_feat_list.append(pixel_feat)
        # edge diff
        edge_feat = get_edge_feature(bgr_frame)
        edge_feat_list.append(edge_feat)
        # area diff
        area_feat = get_area_feature(bgr_frame)
        area_feat_list.append(area_feat)
        # update prev frame
        if i == batch.size(0)-1:
            last_frame = bgr_frame
    
    pixel_diff_list = []
    edge_diff_list = []
    area_diff_list = []
    # calc feature diffs
    for i in range(batch.size(0)):
        pixel_diff_list.append(calc_pixel_diff(pixel_feat_list[i+1], pixel_feat_list[i]))
        edge_diff_list.append(calc_edge_diff(edge_feat_list[i+1], edge_feat_list[i]))
        area_diff_list.append(calc_area_diff(area_feat_list[i+1], area_feat_list[i]))
          
    return last_frame, pixel_diff_list, edge_diff_list, area_diff_list

def video_mAP_ucf():
    """
    Calculate video_mAP over the test dataset
    """
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    CLASSES = ('Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling', 
               'Diving', 'Fencing', 'FloorGymnastics', 'GolfSwing', 'HorseRiding',
               'IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing', 'SalsaSpin',
               'SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling', 'Surfing',
               'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog')
    
    video_testlist = []
    sample_cnt = 0
    with open(testlist, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.rstrip()
            if sample_cnt < sample_thresh:
                video_testlist.append(line)
            sample_cnt = (sample_cnt + 1)%10

    detected_boxes = {}
    gt_videos = {}

    gt_data = loadmat(gt_file)['annot']
    n_videos = gt_data.shape[1]
    for i in range(n_videos):
        video_name = gt_data[0][i][1][0]
        if video_name in video_testlist:
            n_tubes = len(gt_data[0][i][2][0])
            v_annotation = {}
            all_gt_boxes = []
            for j in range(n_tubes):  
                gt_one_tube = [] 
                tube_start_frame = gt_data[0][i][2][0][j][1][0][0]
                tube_end_frame = gt_data[0][i][2][0][j][0][0][0]
                tube_class = gt_data[0][i][2][0][j][2][0][0]
                tube_data = gt_data[0][i][2][0][j][3]
                tube_length = tube_end_frame - tube_start_frame + 1
            
                for k in range(tube_length):
                    gt_boxes = []
                    gt_boxes.append(int(tube_start_frame+k))
                    gt_boxes.append(float(tube_data[k][0]))
                    gt_boxes.append(float(tube_data[k][1]))
                    gt_boxes.append(float(tube_data[k][0]) + float(tube_data[k][2]))
                    gt_boxes.append(float(tube_data[k][1]) + float(tube_data[k][3]))
                    gt_one_tube.append(gt_boxes)
                all_gt_boxes.append(gt_one_tube)

            v_annotation['gt_classes'] = tube_class
            v_annotation['tubes'] = np.array(all_gt_boxes)
            gt_videos[video_name] = v_annotation

    bbx_det_start = time.perf_counter()
    all_feat = np.zeros((len(lines),3))
    for lidx, line in enumerate(lines):
        print(line)
        line = line.rstrip()
        test_loader = torch.utils.data.DataLoader(
                          testData(os.path.join(base_path, 'rgb-images', line),
                          shape=(224, 224), transform=transforms.Compose([
                          transforms.ToTensor()]), clip_duration=clip_duration),
                          batch_size=64, shuffle=False, **kwargs)

        prev_frame = None
        pixel_diff_list, edge_diff_list, area_diff_list = [], [], []
        for batch_idx, (data, target, img_name) in enumerate(test_loader):
            prev_frame, pixel_diff, edge_diff, area_diff = extract_n_filter_one_batch(data[:, :, -1, :, :], prev_frame)
            pixel_diff_list += pixel_diff
            edge_diff_list += edge_diff
            area_diff_list += area_diff
            if use_cuda:
                data = data.cuda()
            with torch.no_grad():
                data = Variable(data)
                output = model(data).data

                all_boxes = get_region_boxes_video(output, conf_thresh, num_classes, anchors, num_anchors, 0, 1)
                for i in range(output.size(0)):
                    boxes = all_boxes[i]
                    boxes = nms(boxes, nms_thresh)
                    n_boxes = len(boxes)

                    # generate detected tubes for all classes
                    # save format: {img_name: {cls_ind: array[[x1,y1,x2,y2, cls_score], [], ...]}}
                    img_annotation = {}
                    for cls_idx in range(num_classes):
                        cls_idx += 1    # index begins from 1
                        cls_boxes = np.zeros([n_boxes, 5], dtype=np.float32)
                        for b in range(n_boxes):
                            cls_boxes[b][0] = max(float(boxes[b][0]-boxes[b][2]/2.0) * 320.0, 0.0)
                            cls_boxes[b][1] = max(float(boxes[b][1]-boxes[b][3]/2.0) * 240.0, 0.0)
                            cls_boxes[b][2] = min(float(boxes[b][0]+boxes[b][2]/2.0) * 320.0, 320.0)
                            cls_boxes[b][3] = min(float(boxes[b][1]+boxes[b][3]/2.0) * 240.0, 240.0)
                            cls_boxes[b][4] = float(boxes[b][5+(cls_idx-1)*2])
                        img_annotation[cls_idx] = cls_boxes
                    detected_boxes[img_name[i]] = img_annotation
        all_feat[lidx,:] = [np.mean(pixel_diff_list), np.mean(edge_diff_list), np.mean(area_diff_list)]
    bbx_det_end = time.perf_counter()
    bbx_pred_t = (bbx_det_end - bbx_det_start)

    # iou_list = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75]
    iou_list = [0.1, 0.2, 0.5, 0.75]
    ref_frame_list = [1000000]
    skip_cnt_list = [0, 1, 3, 7, 15, 17, 19, 23, 29]
    file_name = 'ucf24_pred_result_' + str(use_train) + '.txt'
    tube_score_file = 'ucf24_tube_score_' + str(use_train) + '.txt'
    with open(file_name, 'w') as f:
        f.write('v_cnt\tacc\tvmAP_old\tvmAP_new\tloc_t_old\tloc_t_new\tEALR_old\tEALR_new\tmiss_r\n')
    for iou_th in iou_list:
        print('iou is: ', iou_th)
        all_tube_scores = np.zeros((len(lines), len(skip_cnt_list) + 3))
        for idx, skip_cnt in enumerate(skip_cnt_list):
            print_str, tube_scores = evaluate_videoAP(gt_videos, detected_boxes, CLASSES, bbx_pred_t, iou_th, True, ref_frame_list[0], skip_cnt)
            with open(file_name, 'a+') as f:
                f.write(str(iou_th) + '\t' + str(skip_cnt) + '\t')
                f.write(print_str)
            all_tube_scores[:,idx] = tube_scores
        all_tube_scores[:,idx+1:idx+4] = all_feat
        if iouth == iou_list[0]:
            with open(tube_score_file,'wb') as f:
                for line in all_tube_scores:
                    np.savetxt(f, line, fmt='%.3f')


def video_mAP_jhmdb():
    """
    Calculate video_mAP over the test set
    """
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    CLASSES = ('brush_hair', 'catch', 'clap', 'climb_stairs', 'golf', 
                    'jump', 'kick_ball', 'pick', 'pour', 'pullup', 'push',
                    'run', 'shoot_ball', 'shoot_bow', 'shoot_gun', 'sit',
                    'stand', 'swing_baseball', 'throw', 'walk', 'wave')

    with open(testlist, 'r') as file:
        lines = file.readlines()

    detected_boxes = {}
    gt_videos = {}
    bbx_det_start = time.perf_counter()
    all_feat = np.zeros((len(lines),3))
    for lidx, line in enumerate(lines):
        print(line)

        line = line.rstrip()

        test_loader = torch.utils.data.DataLoader(
                          testData(os.path.join(base_path, 'rgb-images', line),
                          shape=(224, 224), transform=transforms.Compose([
                          transforms.ToTensor()]), clip_duration=clip_duration),
                          batch_size=1, shuffle=False, **kwargs)

        video_name = ''
        v_annotation = {}
        all_gt_boxes = []
        t_label = -1
        
        prev_frame = None
        pixel_diff_list, edge_diff_list, area_diff_list = [], [], []
        for batch_idx, (data, target, img_name) in enumerate(test_loader):
            prev_frame, pixel_diff, edge_diff, area_diff = extract_n_filter_one_batch(data[:, :, -1, :, :], prev_frame)
            pixel_diff_list += pixel_diff
            edge_diff_list += edge_diff
            area_diff_list += area_diff
            
            path_split = img_name[0].split('/')
            if video_name == '':
                video_name = os.path.join(path_split[0], path_split[1])

            if use_cuda:
                data = data.cuda()
            with torch.no_grad():
                data = Variable(data)
                output = model(data).data
                all_boxes = get_region_boxes_video(output, conf_thresh, num_classes, anchors, num_anchors, 0, 1)

                assert(output.size(0) == 1)
                for i in range(output.size(0)):
                    boxes = all_boxes[i]
                    boxes = nms(boxes, nms_thresh)
                    n_boxes = len(boxes)
                    truths = target[i].view(-1, 5)

                    if t_label == -1:
                        t_label = int(truths[0][0]) + 1

                    # generate detected tubes for all classes
                    # save format: {img_name: {cls_ind: array[[x1,y1,x2,y2, cls_score], [], ...]}}
                    img_annotation = {}
                    for cls_idx in range(num_classes):
                        cls_idx += 1    # index begins from 1
                        cls_boxes = np.zeros([n_boxes, 5], dtype=np.float32)
                        for b in range(n_boxes):
                            cls_boxes[b][0] = max(float(boxes[b][0]-boxes[b][2]/2.0) * 320.0, 0.0)
                            cls_boxes[b][1] = max(float(boxes[b][1]-boxes[b][3]/2.0) * 240.0, 0.0)
                            cls_boxes[b][2] = min(float(boxes[b][0]+boxes[b][2]/2.0) * 320.0, 320.0)
                            cls_boxes[b][3] = min(float(boxes[b][1]+boxes[b][3]/2.0) * 240.0, 240.0)
                            cls_boxes[b][4] = float(boxes[b][5+(cls_idx-1)*2])
                        img_annotation[cls_idx] = cls_boxes
                    detected_boxes[img_name[0]] = img_annotation
                        
                for i in range(output.size(0)):
                    truths = target[i].view(-1, 5)
                    num_gts = truths_length(truths)
                    # generate corresponding gts
                    # save format: {v_name: {tubes: [[frame_index, x1,y1,x2,y2]], gt_classes: vlabel}} 
                    gt_boxes = []
                    for g in range(num_gts):
                        gt_boxes.append(int(path_split[2][:5]))
                        gt_boxes.append(float(truths[g][1]-truths[g][3]/2.0) * 320.0)
                        gt_boxes.append(float(truths[g][2]-truths[g][4]/2.0) * 240.0)
                        gt_boxes.append(float(truths[g][1]+truths[g][3]/2.0) * 320.0)
                        gt_boxes.append(float(truths[g][2]+truths[g][4]/2.0) * 240.0)
                        all_gt_boxes.append(gt_boxes)
                    
        v_annotation['gt_classes'] = t_label
        v_annotation['tubes'] = np.expand_dims(np.array(all_gt_boxes), axis=0)
        gt_videos[video_name] = v_annotation
        
        all_feat[lidx,:] = [np.mean(pixel_diff_list), np.mean(edge_diff_list), np.mean(area_diff_list)]

    bbx_det_end = time.perf_counter()
    bbx_pred_t = (bbx_det_end - bbx_det_start)

    # iou_list = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75]
    iou_list = [0.1, 0.2, 0.5, 0.75]
    ref_frame_list = [1000000]
    skip_cnt_list = [0, 1, 3, 7, 15, 17, 19, 23, 29]
    file_name = 'jhmdb_pred_result_' + str(use_train) + '.txt'
    tube_score_file = 'jhmdb_tube_score_' + str(use_train) + '.txt'
    with open(file_name, 'w') as f:
        f.write('v_cnt\tacc\tvmAP_old\tvmAP_new\tloc_t_old\tloc_t_new\tEALR_old\tEALR_new\tmiss_r\n')
    for iou_th in iou_list:
        print('iou is: ', iou_th)
        all_tube_scores = np.zeros((len(lines), len(skip_cnt_list)))
        for idx, skip_cnt in enumerate(skip_cnt_list):
            print_str, tube_scores = evaluate_videoAP(gt_videos, detected_boxes, CLASSES, bbx_pred_t, iou_th, True, ref_frame_list[0], skip_cnt)
            with open(file_name, 'a+') as f:
                f.write(str(iou_th) + ',' + str(skip_cnt) + '\t')
                f.write(print_str)
            all_tube_scores[:,idx] = tube_scores
        all_tube_scores[:,idx+1:idx+4] = all_feat
        if iouth == iou_list[0]:
            with open(tube_score_file,'wb') as f:
                for line in all_tube_scores:
                    np.savetxt(f, line, fmt='%.3f')
                    
                

if __name__ == '__main__':
    if opt.dataset == 'ucf101-24':
        video_mAP_ucf()
    elif opt.dataset == 'jhmdb-21':
        video_mAP_jhmdb()
    
