# -*- coding:utf-8 -*-
import numpy as np
import os,time
from utils import *

def compute_score_one_class(bbox1, bbox2, w_iou=1.0, w_scores=1.0, w_scores_mul=0.5):
    # bbx: <x1> <y1> <x2> <y2> <class score>
    n_bbox1 = bbox1.shape[0]
    n_bbox2 = bbox2.shape[0]
    # for saving all possible scores between each two bbxes in successive frames
    scores = np.zeros([n_bbox1, n_bbox2], dtype=np.float32)
    for i in range(n_bbox1):
        box1 = bbox1[i, :4]
        for j in range(n_bbox2):
            box2 = bbox2[j, :4]
            bbox_iou_frames = bbox_iou(box1, box2, x1y1x2y2=True)
            sum_score_frames = bbox1[i, 4] + bbox2[j, 4]
            mul_score_frames = bbox1[i, 4] * bbox2[j, 4]
            scores[i, j] = w_iou * bbox_iou_frames + w_scores * sum_score_frames + w_scores_mul * mul_score_frames

    return scores

def link_bbxes_between_frames(bbox_list, w_iou=1.0, w_scores=1.0, w_scores_mul=0.5):
    # bbx_list: list of bounding boxes <x1> <y1> <x2> <y2> <class score>
    # check no empty detections
    det_t = []
    t_start = time.perf_counter()
    ind_notempty = []
    nfr = len(bbox_list)
    for i in range(nfr):
        if np.array(bbox_list[i]).size:
            ind_notempty.append(i)
    # no detections at all
    if not ind_notempty:
        return []
    # miss some frames
    elif len(ind_notempty)!=nfr:     
        for i in range(nfr):
            if not np.array(bbox_list[i]).size:
                # copy the nearest detections to fill in the missing frames
                ind_dis = np.abs(np.array(ind_notempty) - i)
                nn = np.argmin(ind_dis)
                bbox_list[i] = bbox_list[ind_notempty[nn]]

    
    detect = bbox_list
    nframes = len(detect)
    res = []

    isempty_vertex = np.zeros([nframes,], dtype=np.bool)
    edge_scores = [compute_score_one_class(detect[i], detect[i+1], w_iou=w_iou, w_scores=w_scores, w_scores_mul=w_scores_mul) for i in range(nframes-1)]
    copy_edge_scores = edge_scores

    while not np.any(isempty_vertex):
        # initialize
        scores = [np.zeros([d.shape[0],], dtype=np.float32) for d in detect]
        index = [np.nan*np.ones([d.shape[0],], dtype=np.float32) for d in detect]
        # viterbi
        # from the second last frame back
        for i in range(nframes-2, -1, -1):
            edge_score = edge_scores[i] + scores[i+1]
            # find the maximum score for each bbox in the i-th frame and the corresponding index
            scores[i] = np.max(edge_score, axis=1)
            index[i] = np.argmax(edge_score, axis=1)
        # decode
        idx = -np.ones([nframes], dtype=np.int32)
        idx[0] = np.argmax(scores[0])
        for i in range(0, nframes-1):
            idx[i+1] = index[i][idx[i]]
        # remove covered boxes and build output structures
        this = np.empty((nframes, 6), dtype=np.float32)
        this[:, 0] = 1 + np.arange(nframes)
        for i in range(nframes):
            j = idx[i]
            iouscore = 0
            if i < nframes-1:
                iouscore = copy_edge_scores[i][j, idx[i+1]] - bbox_list[i][j, 4] - bbox_list[i+1][idx[i+1], 4]

            if i < nframes-1: edge_scores[i] = np.delete(edge_scores[i], j, 0)
            if i > 0: edge_scores[i-1] = np.delete(edge_scores[i-1], j, 1)
            this[i, 1:5] = detect[i][j, :4]
            this[i, 5] = detect[i][j, 4]
            detect[i] = np.delete(detect[i], j, 0)
            isempty_vertex[i] = (detect[i].size==0) # it is true when there is no detection in any frame
        res.append( this )
        t_end = time.perf_counter()
        det_t.append(t_end - t_start)
        t_start = t_end
        if len(res) == 3:
            break
        
    return res, det_t


def link_video_one_class(vid_det, bNMS3d = False, gtlen=None):
    '''
    linking for one class in a video (in full length)
    vid_det: a list of [frame_index, [bbox cls_score]]
    gtlen: the mean length of gt in training set
    return a list of tube [array[frame_index, x1,y1,x2,y2, cls_score]]
    '''
    # list of bbox information [[bbox in frame 1], [bbox in frame 2], ...]
    # need to check the time cost of building each tube
    vdets = [vid_det[i][1] for i in range(len(vid_det))]
    vres, det_t = link_bbxes_between_frames(vdets) 
    if len(vres) != 0:
        if bNMS3d:
            tube = [b[:, :5] for b in vres]
            # compute score for each tube
            tube_scores = [np.mean(b[:, 5]) for b in vres]
            dets = [(tube[t], tube_scores[t]) for t in range(len(tube))]
            # nms for tubes
            keep = nms_3d(dets, 0.3) # bug for nms3dt
            if np.array(keep).size:
                vres_keep = [vres[k] for k in keep]
                det_t_keep = [det_t[k] for k in keep]
                # max subarray with penalization -|Lc-L|/Lc
                if gtlen:
                    vres = temporal_check(vres_keep, gtlen)
                else:
                    vres = vres_keep
                det_t = det_t_keep
    return vres, det_t


def video_ap_one_class(gt, pred_videos, potential_class, iou_thresh = 0.2, bTemporal = False, gtlen = None):
    '''
    gt: [ video_index, array[frame_index, x1,y1,x2,y2] ]
    pred_videos: [ video_index, [ [frame_index, [[x1,y1,x2,y2, score]] ] ] ]
    '''
    # link for prediction
    pred = []
    for pred_v in pred_videos:
        video_index = pred_v[0]
        pred_link_v, det_t = link_video_one_class(pred_v[1], True, gtlen) # [array<frame_index, x1,y1,x2,y2, cls_score>]
        for tube, t in zip(pred_link_v, det_t):
            pred.append((video_index, tube, t))

    # sort tubes according to scores (descending order)
    argsort_scores = np.argsort(-np.array([np.mean(b[:, 5]) for _, b, _ in pred])) 
    pr = np.empty((len(pred)+1, 2), dtype=np.float32) # precision, recall
    pr[0,0] = 1.0
    pr[0,1] = 0.0
    fn = len(gt) #sum([len(a[1]) for a in gt])
    fp = 0
    tp = 0

    # for computing mAP with class pred
    pr_new = np.empty((len(pred)+1, 2), dtype=np.float32) # precision, recall
    pr_new[0,0] = 1.0
    pr_new[0,1] = 0.0
    fn_new = len(gt) #sum([len(a[1]) for a in gt])
    fp_new = 0
    tp_new = 0

    gt_v_index = [g[0] for g in gt]
    pos_t, neg_t = 0, 0
    saved_t = 0
    missed_actions = 0
    actual_t = 0
    v_cnt = len(pred_videos)
    for i, k in enumerate(argsort_scores):
        # check each tube
        # if i % 100 == 0:
        #     print ("%6.2f%% boxes processed, %d positives found, %d remain" %(100*float(i)/argsort_scores.size, tp, fn))
        video_index, boxes, t = pred[k]
        # if we can decide what class should be considered in each video index
        # then other class can be skipped
        # the result will not be counted in AP and the time will also not be counted
        ispositive = False
        if video_index in gt_v_index:
            gt_this_index, gt_this = [], []
            # find all gt tubes of one class in one video
            for j, g in enumerate(gt):
                if g[0] == video_index:
                    gt_this.append(g[1])
                    gt_this_index.append(j)
            if len(gt_this) > 0:
                if bTemporal:
                    iou = np.array([iou3dt(np.array(g), boxes[:, :5]) for g in gt_this])
                else:            
                    if boxes.shape[0] > gt_this[0].shape[0]:
                        # in case some frame don't have gt 
                        iou = np.array([iou3d(g, boxes[int(g[0,0]-1):int(g[-1,0]),:5]) for g in gt_this]) 
                    elif boxes.shape[0]<gt_this[0].shape[0]:
                        # in flow case 
                        iou = np.array([iou3d(g[int(boxes[0,0]-1):int(boxes[-1,0]),:], boxes[:,:5]) for g in gt_this]) 
                    else:
                        iou = np.array([iou3d(g, boxes[:,:5]) for g in gt_this]) 
                
                if iou.size > 0: # on ucf101 if invalid annotation ....
                    argmax = np.argmax(iou)
                    # check if this det tube matches any gt tube
                    if iou[argmax] >= iou_thresh:
                        ispositive = True
                        del gt[gt_this_index[argmax]]
        if potential_class[video_index-1]:
            actual_t += t
        if ispositive:
            tp += 1
            fn -= 1
            # add to positive time
            pos_t += t
            if not potential_class[video_index-1]:
                missed_actions += 1
            else:
                tp_new += 1
                fn_new -= 1
        else:
            fp += 1
            # add to negative time
            neg_t += t
            if not potential_class[video_index-1]:
                saved_t += t
            else:
                fp_new += 1
        pr[i+1,0] = float(tp)/float(tp+fp)
        pr[i+1,1] = float(tp)/float(tp+fn + 0.00001)
        pr_new[i+1,0] = float(tp_new)/float(tp_new+fp_new + 0.00001)
        pr_new[i+1,1] = float(tp_new)/float(tp_new+fn_new + 0.00001)
    ap = voc_ap(pr)
    ap_new = voc_ap(pr_new)

    return ap, ap_new, pos_t/v_cnt,  neg_t/v_cnt, saved_t/v_cnt, missed_actions, tp, actual_t/v_cnt


def gt_to_videts(gt_v):
    # return  [label, video_index, [[frame_index, x1,y1,x2,y2], [], []] ]
    keys = list(gt_v.keys())
    keys.sort()
    res = []
    for i in range(len(keys)):
        # annotation of the video: tubes and gt_classes
        v_annot = gt_v[keys[i]]
        for j in range(len(v_annot['tubes'])):
            res.append([v_annot['gt_classes'], i+1, v_annot['tubes'][j]])
    return res

def class_prediction(n_videos, CLASSES, pred_videos_format, ref_frame_cnt = 20):
    # input: pred_videos_format:array<cls_ind, v_ind, v_dets>
    # output: pred_videos_classes:array<v_ind, pred_classes>
    # extra time usage
    # we can probably use the transision of states
    # existance of classes in one frame can be the state
    # it can transit to the classes of the next frame
    # it is also like predicting rest of a sentence using a few words
    # only one class per video
    potential_class = np.zeros([len(CLASSES), n_videos], dtype=np.bool)
    # potential_class[0,:] = np.ones([n_videos,], dtype=np.bool)
    for v_ind in range(n_videos):
        # extract bbxs of one video
        pred_bbxs = [p for p in pred_videos_format if p[1]-1==v_ind]
        # analyze class scores
        class_scores = np.zeros([len(CLASSES),])
        for cls_ind, _, v_dets in pred_bbxs:
            cls_score = 0
            for frame_index, img_cls_dets in v_dets:
                for cls_box in img_cls_dets:
                    cls_score += cls_box[4]
                if frame_index >= ref_frame_cnt-1:
                    break
            class_scores[cls_ind-1] = cls_score
        cls_ind = np.argmax(class_scores)
        potential_class[cls_ind,v_ind] = True

    return potential_class

def eval_class_prediction(potential_class, gt_videos_format, n_videos, CLASSES):
    acc = 0
    for v_ind in range(n_videos):
        one_video_result = 0
        pred_cls = potential_class[:,v_ind]
        gt_class_ind = [g[0]-1 for g in gt_videos_format if g[1]-1==v_ind]
        for gt_cls_ind in gt_class_ind:
            one_video_result = potential_class[gt_cls_ind,v_ind]
        acc += one_video_result
    acc /= n_videos
    return acc

def evaluate_videoAP(gt_videos, all_boxes, CLASSES, bbx_pred_t, iou_thresh = 0.2, bTemporal = False, ref_frame_cnt = 20, prior_length = None):
    '''
    gt_videos: {vname:{tubes: [[frame_index, x1,y1,x2,y2]], gt_classes: vlabel}} 
    all_boxes: {imgname:{cls_ind:array[x1,y1,x2,y2, cls_score]}}
    '''
    def imagebox_to_videts(img_boxes, CLASSES):
        # put bboxes in one video of the same class into one v_dets
        # image names
        keys = list(all_boxes.keys())
        keys.sort()
        res = []
        # without 'background'
        for cls_ind, cls in enumerate(CLASSES[0:]):
            v_cnt = 1
            frame_index = 1
            v_dets = []
            cls_ind += 1
            # get the directory path of images
            preVideo = os.path.dirname(keys[0])
            for i in range(len(keys)):
                curVideo = os.path.dirname(keys[i])
                img_cls_dets = img_boxes[keys[i]][cls_ind]
                v_dets.append([frame_index, img_cls_dets])
                frame_index += 1
                if preVideo!=curVideo:
                    preVideo = curVideo
                    frame_index = 1
                    # tmp_dets = v_dets[-1]
                    del v_dets[-1]
                    res.append([cls_ind, v_cnt, v_dets])
                    v_cnt += 1
                    v_dets = []
                    # v_dets.append(tmp_dets)
                    v_dets.append([frame_index, img_cls_dets])
                    frame_index += 1
            # the last video
            # print('num of videos:{}'.format(v_cnt))
            res.append([cls_ind, v_cnt, v_dets])
        return res, v_cnt

    print_str = ""
    gt_videos_format = gt_to_videts(gt_videos)
    pred_videos_format, v_cnt = imagebox_to_videts(all_boxes, CLASSES)
    # predict potential classes of each video based on first few frames
    pred_start = time.perf_counter()
    potential_class = class_prediction(v_cnt, CLASSES, pred_videos_format, ref_frame_cnt)
    pred_end = time.perf_counter()
    cls_pred_t = (pred_end-pred_start)/v_cnt
    print_str += "Cls pred time:" + str(cls_pred_t) + ", video num:" + str(v_cnt) + '\n'
    # evaluate class prediction
    pred_acc = eval_class_prediction(potential_class, gt_videos_format, v_cnt, CLASSES)
    print_str += "Pred accuracy:" + str(pred_acc) + '\n'

    ap_all = [] 
    ap_new_all = []   
    pos_t_all = []
    neg_t_all = []
    saved_t_all = []
    missed_actions_all = []
    gt_actions_all = []
    actual_t_all = []
    link_start = time.perf_counter()
    # look at different classes and link frames of that class
    for cls_ind, cls in enumerate(CLASSES[0:]):
        cls_ind += 1
        # [ video_index, [[frame_index, x1,y1,x2,y2]] ]
        gt = [g[1:] for g in gt_videos_format if g[0]==cls_ind]
        pred_cls = [p[1:] for p in pred_videos_format if p[0]==cls_ind]
        cls_len = None
        ap, ap_new, pos_t, neg_t, saved_t, missed_actions, gt_actions, actual_t = video_ap_one_class(gt, pred_cls, potential_class[cls_ind-1,:], iou_thresh, bTemporal, cls_len)
        ap_all.append(ap)
        ap_new_all.append(ap_new)
        pos_t_all.append(pos_t)
        neg_t_all.append(neg_t)
        saved_t_all.append(saved_t)
        missed_actions_all.append(missed_actions)
        gt_actions_all.append(gt_actions)
        actual_t_all.append(actual_t)
    link_end = time.perf_counter()
    act_loc_t_old = bbx_pred_t/v_cnt + np.sum(pos_t_all) + np.sum(neg_t_all)
    act_loc_t_new = bbx_pred_t/v_cnt + cls_pred_t + np.sum(actual_t_all)
    print_str += "Video mAP:" + str(np.mean(ap_all)) + ',' + str(np.mean(ap_new_all)) + '\n'
    print_str += "Total loc time:" + str(act_loc_t_old) + ',' + str(act_loc_t_new) + '\n'
    print_str += "EALR:" + str(np.mean(ap_all)/act_loc_t_old) + ',' +  str(np.mean(ap_new_all)/act_loc_t_new) + '\n'
    print_str += "Pos time:" + str(np.sum(pos_t_all)) + ", neg time:" + str(np.sum(neg_t_all)) + '\n'
    print_str += "Saved time:" + str(np.sum(saved_t_all)) + ", Actual link time:" + str(np.sum(actual_t_all)) + '\n'
    print_str += "Miss ratio:" + str(np.sum(missed_actions_all)) + '/' + str(np.sum(gt_actions_all)) + '\n'
    # print_str += "Link time:" + str((link_end - link_start)/v_cnt) + '\n'
    # print_str += "Video AP:" + str(ap_all) + '\n'
    # print_str += "Video AP new:" + str(ap_new_all) + '\n'

    return print_str
