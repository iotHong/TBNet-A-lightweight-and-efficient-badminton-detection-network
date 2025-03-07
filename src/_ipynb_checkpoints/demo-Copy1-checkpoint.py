"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.06.10
# email: nguyenmaudung93.kstn@gmail.com
# project repo: https://github.com/maudzung/TTNet-Realtime-for-Table-Tennis-Pytorch
-----------------------------------------------------------------------------------
# Description: This script for demonstration
"""

import os
import sys
from collections import deque

import cv2
import numpy as np
import torch

import ffmpeg

sys.path.append('./')

from data_process.ttnet_video_loader import TTNet_Video_Loader
from models.model_utils import create_model, load_pretrained_model
from config.config import parse_configs
from utils.post_processing import post_processing
from utils.misc import time_synchronized


def demo(configs):
    alltime=0
    video_loader = TTNet_Video_Loader(configs.video_path, configs.input_size, configs.num_frames_sequence)
    result_filename = os.path.join(configs.save_demo_dir, 'results.txt')
    frame_rate = video_loader.video_fps
    if configs.save_demo_output:
        configs.frame_dir = os.path.join(configs.save_demo_dir, 'frame')
        if not os.path.isdir(configs.frame_dir):
            os.makedirs(configs.frame_dir)

    configs.device = torch.device('cuda:{}'.format(configs.gpu_idx))

    # model
    model = create_model(configs)
    model.cuda()

    assert configs.pretrained_path is not None, "Need to load the pre-trained model"
    model = load_pretrained_model(model, configs.pretrained_path, configs.gpu_idx, configs.overwrite_global_2_local)

    model.eval()
    middle_idx = int(configs.num_frames_sequence / 2)
    queue_frames = deque(maxlen=middle_idx + 1)
    frame_idx = 0
    w_original, h_original = 1280, 720
    w_resize, h_resize = 320, 128
    w_ratio = w_original / w_resize
    h_ratio = h_original / h_resize
    with torch.no_grad():
        for count, resized_imgs in video_loader:
            # take the middle one
            img = cv2.resize(resized_imgs[3 * middle_idx: 3 * (middle_idx + 1)].transpose(1, 2, 0), (w_original, h_original))
            # Expand the first dim
            resized_imgs = torch.from_numpy(resized_imgs).to(configs.device, non_blocking=True).float().unsqueeze(0)
            t1 = time_synchronized()
            pred_ball_global, pred_ball_local, pred_events, pred_seg = model.run_demo(resized_imgs)
            t2 = time_synchronized()
            prediction_global, prediction_local, prediction_seg, prediction_events = post_processing(
                pred_ball_global, pred_ball_local, pred_events, pred_seg, configs.input_size[0],
                configs.thresh_ball_pos_mask, configs.seg_thresh, configs.event_thresh)
            prediction_ball_final = [
                int(prediction_global[0] * w_ratio + prediction_local[0] - w_resize / 2),
                int(prediction_global[1] * h_ratio + prediction_local[1] - h_resize / 2)
            ]

            # Get infor of the (middle_idx + 1)th frame
            if len(queue_frames) == middle_idx + 1:
                frame_pred_infor = queue_frames.popleft()
                seg_img = frame_pred_infor['seg'].astype(np.uint8)
                ball_pos = frame_pred_infor['ball']
                seg_img = cv2.resize(seg_img, (w_original, h_original))
                print("frame num:",frame_idx+4)
                ploted_img = plot_detection(img, ball_pos, seg_img, prediction_events)

                ploted_img = cv2.cvtColor(ploted_img, cv2.COLOR_RGB2BGR)
                if configs.show_image:
                    # cv2.imshow('ploted_img', ploted_img)
                    cv2.waitKey(10)
                if configs.save_demo_output:
                    cv2.imwrite(os.path.join(configs.frame_dir, '{:06d}.jpg'.format(frame_idx-5)), ploted_img)

            frame_pred_infor = {
                'seg': prediction_seg,
                'ball': prediction_ball_final
            }
            queue_frames.append(frame_pred_infor)

            frame_idx += 1
            # if frame_idx >
            print('Done frame_idx {} - time {:.3f}s'.format(frame_idx-1+4, t2 - t1))
            print("-----------------------------")
            alltime=alltime+t2 - t1
            if frame_idx >=507:
                break
        print("process time:",alltime)
        print("FPS:",frame_idx/alltime)
        
    if configs.output_format == 'video':
        output_video_path = os.path.join(configs.save_demo_dir, configs.video_path[13:-4]+'result.mp4')
        print(output_video_path)
        cmd_str = 'ffmpeg -f image2 -i {}/%06d.jpg -b 5000k -c:v mpeg4 {}'.format(
            os.path.join(configs.frame_dir), output_video_path)
        os.system(cmd_str)


def plot_detection(img, ball_pos, seg_img, events):
    """Show the predicted information in the image"""
    img = cv2.addWeighted(img, 1., seg_img * 255, 0.3, 0)
#     ++++++++++++++++++++++++++++++++++++++++++++++++++++
# ===========================================================================
    h_pred = (seg_img * 255)
    h_pred = h_pred.astype('uint8')
    OutputFrame = cv2.cvtColor(h_pred, cv2.COLOR_BGR2GRAY)
    # cvtColor(h_pred, OutputFrame, CV_BGR2GRAY)

    (cnts, _) = cv2.findContours(OutputFrame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in cnts]
    max_area_idx = 0
    # print("+++++++++", max_area_idx)
    if len(rects) == 0:
        # TP += 1
        # break
        img = cv2.circle(img, tuple(ball_pos), 5, (255, 0, 255), -1)
        # event_name = 'is bounce: {:.2f}, is net: {:.2f}'.format(events[0], events[1])
        # img = cv2.putText(img, event_name, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        return img
    
    max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
    for j in range(len(rects)):
        area = rects[j][2] * rects[j][3]
        if area > max_area:
            max_area_idx = j
            max_area = area
    target = rects[max_area_idx]
    (cx_pred, cy_pred) = (int(target[0] + target[2] / 2), int(target[1] + target[3] / 2))
    print(cx_pred, cy_pred)
    img = cv2.circle(img, (cx_pred, cy_pred), 5, (255, 0, 0), -1)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    img = cv2.circle(img, tuple(ball_pos), 5, (255, 0, 255), -1)
    # event_name = 'is land: {:.2f}, is hitnet: {:.2f}, is shot: {:.2f}'.format(events[0], events[1], events[0])
    # img = cv2.putText(img, event_name, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

    return img


if __name__ == '__main__':
    configs = parse_configs()
    demo(configs=configs)
