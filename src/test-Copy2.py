import time
import sys
import os
import warnings
import datetime
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from tqdm import tqdm
import cv2
import math

sys.path.append('./')

from data_process.ttnet_dataloader import create_test_dataloader
from models.model_utils import create_model, load_pretrained_model, make_data_parallel, get_num_parameters
from utils.misc import AverageMeter
from config.config import parse_configs
from utils.post_processing import get_prediction_ball_pos, get_prediction_seg, prediction_get_events
from utils.metrics import SPCE, PCE

def display_track(TP, TN, FP1, FP2, FN):
	print('======================Evaluate=======================')
	print("Number of true positive:", TP)
	print("Number of true negative:", TN)
	print("Number of false positive FP1:", FP1)
	print("Number of false positive FP2:", FP2)
	print("Number of false negative:", FN)
	(accuracy, precision, recall) = evaluation_track(TP, TN, FP1, FP2, FN)
	print("Accuracy:", accuracy)
	print("Precision:", precision)
	print("Recall:", recall)
	print('=====================================================')

def evaluation_track(TP, TN, FP1, FP2, FN):
	try:
		accuracy = (TP + TN) / (TP + TN + FP1 + FP2 + FN)
	except:
		accuracy = 0
	try:
		precision = TP / (TP + FP1 + FP2)
	except:
		precision = 0
	try:
		recall = TP / (TP + FN)
	except:
		recall = 0
	return (accuracy, precision, recall)

def outcome_track(y_pred, y_true, tol):
	n = y_pred.shape[0]
	print(n)
	i = 0
	TP = TN = FP1 = FP2 = FN = 0
	y_pred = torch.tensor(y_pred)
	y_true = torch.tensor(y_true)
	while i < n:
		for j in range(1):
			if torch.max(y_pred[i][j]) == 0 and torch.max(y_true[i][j]) == 0:
				TN += 1
			elif torch.max(y_pred[i][j]) > 0 and torch.max(y_true[i][j]) == 0:
				FP2 += 1
			elif torch.max(y_pred[i][j]) == 0 and torch.max(y_true[i][j]) > 0:
				FN += 1
			elif torch.max(y_pred[i][j]) > 0 and torch.max(y_true[i][j]) > 0:
				h_pred = (y_pred[i][j] * 255).cpu().numpy()
				h_true = (y_true[i][j] * 255).cpu().numpy()
				h_pred = h_pred.astype('uint8')
				h_true = h_true.astype('uint8')
				#h_pred
				(cnts, _) = cv2.findContours(h_pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				rects = [cv2.boundingRect(ctr) for ctr in cnts]
				max_area_idx = 0
				print("+++++++++",max_area_idx)
				if len(rects) == 0:
					TP +=1
					break
				max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
				for j in range(len(rects)):
					area = rects[j][2] * rects[j][3]
					if area > max_area:
						max_area_idx = j
						max_area = area
				target = rects[max_area_idx]
				(cx_pred, cy_pred) = (int(target[0] + target[2] / 2), int(target[1] + target[3] / 2))
				print(cx_pred, cy_pred)
				#h_true
				(cnts, _) = cv2.findContours(h_true.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				rects = [cv2.boundingRect(ctr) for ctr in cnts]
				print(rects)
				max_area_idx = 0
				max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
				for j in range(len(rects)):
					area = rects[j][2] * rects[j][3]
					if area > max_area:
						max_area_idx = j
						max_area = area
				target = rects[max_area_idx]
				(cx_true, cy_true) = (int(target[0] + target[2] / 2), int(target[1] + target[3] / 2))
				# print((cx_true, cy_true))
				dist = math.sqrt(pow(cx_pred-cx_true, 2)+pow(cy_pred-cy_true, 2))
				if dist > tol:
					FP1 += 1
				else:
					TP += 1
		i += 1
	return (TP, TN, FP1, FP2, FN)

def main():
    configs = parse_configs()

    if configs.gpu_idx is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if configs.dist_url == "env://" and configs.world_size == -1:
        configs.world_size = int(os.environ["WORLD_SIZE"])

    configs.distributed = configs.world_size > 1 or configs.multiprocessing_distributed

    if configs.multiprocessing_distributed:
        configs.world_size = configs.ngpus_per_node * configs.world_size
        mp.spawn(main_worker, nprocs=configs.ngpus_per_node, args=(configs,))
    else:
        main_worker(configs.gpu_idx, configs)


def main_worker(gpu_idx, configs):
    configs.gpu_idx = gpu_idx

    if configs.gpu_idx is not None:
        print("Use GPU: {} for training".format(configs.gpu_idx))
        configs.device = torch.device('cuda:{}'.format(configs.gpu_idx))

    if configs.distributed:
        if configs.dist_url == "env://" and configs.rank == -1:
            configs.rank = int(os.environ["RANK"])
        if configs.multiprocessing_distributed:
            configs.rank = configs.rank * configs.ngpus_per_node + gpu_idx

        dist.init_process_group(backend=configs.dist_backend, init_method=configs.dist_url,
                                world_size=configs.world_size, rank=configs.rank)

    configs.is_master_node = (not configs.distributed) or (
            configs.distributed and (configs.rank % configs.ngpus_per_node == 0))

    # model
    model = create_model(configs)
    model = make_data_parallel(model, configs)

    if configs.is_master_node:
        num_parameters = get_num_parameters(model)
        print('number of trained parameters of the model: {}'.format(num_parameters))

    if configs.pretrained_path is not None:
        model = load_pretrained_model(model, configs.pretrained_path, gpu_idx, configs.overwrite_global_2_local)
    # Load dataset
    test_loader = create_test_dataloader(configs)
    test(test_loader, model, configs)


def test(test_loader, model, configs):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    iou_seg = AverageMeter('IoU_Seg', ':6.4f')
    mse_global = AverageMeter('MSE_Global', ':6.4f')
    mse_local = AverageMeter('MSE_Local', ':6.4f')
    mse_overall = AverageMeter('MSE_Overall', ':6.4f')
    pce = AverageMeter('PCE', ':6.4f')
    spce = AverageMeter('Smooth_PCE', ':6.4f')
    w_original = 1280.
    h_original = 720.
    TP = TN = FP1 = FP2 = FN = 0
    w, h = configs.input_size
    alll=0
    suuuu=0
    starttime = datetime.datetime.now()
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        for batch_idx, (resized_imgs, org_ball_pos_xy, global_ball_pos_xy, target_events, target_seg) in enumerate(
            tqdm(test_loader)):

            print('\n===================== batch_idx: {} ================================'.format(batch_idx))

            data_time.update(time.time() - start_time)
            batch_size = resized_imgs.size(0)
            target_seg = target_seg.to(configs.device, non_blocking=True)
            resized_imgs = resized_imgs.to(configs.device, non_blocking=True).float()
            # compute output

            pred_ball_global, pred_ball_local, pred_events, pred_seg, local_ball_pos_xy, total_loss, _ = model(
                resized_imgs, org_ball_pos_xy, global_ball_pos_xy, target_events, target_seg)

            org_ball_pos_xy = org_ball_pos_xy.numpy()
            global_ball_pos_xy = global_ball_pos_xy.numpy()
            # Transfer output to cpu
#             +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            f_target_seg=target_seg.cpu()
            target_seg = target_seg.cpu().numpy()
            # print(target_seg)
            # mtg=target_seg.size()
            # print(mtg)
            prediction_seg = get_prediction_seg(pred_seg, configs.seg_thresh)
            (tp, tn, fp1, fp2, fn) = outcome_track(pred_seg, f_target_seg, 3)
            TP += tp
            TN += tn
            FP1 += fp1
            FP2 += fp2
            FN += fn
            
            
            for sample_idx in range(batch_size):
                # Get target
                sample_org_ball_pos_xy = org_ball_pos_xy[sample_idx]
                sample_global_ball_pos_xy = global_ball_pos_xy[sample_idx]  # Target
                # Process the global stage
                sample_pred_ball_global = pred_ball_global[sample_idx]
                sample_prediction_ball_global_xy = get_prediction_ball_pos(sample_pred_ball_global, w,
                                                                           configs.thresh_ball_pos_mask)
#                 +++++++++++++++++++++++++++++++++++++++=
                

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

                # Calculate the MSE
                if (sample_global_ball_pos_xy[0] > 0) and (sample_global_ball_pos_xy[1] > 0) and (
                        sample_prediction_ball_global_xy[0] > 0) and (sample_prediction_ball_global_xy[1] > 0):
                    mse = (sample_prediction_ball_global_xy[0] - sample_global_ball_pos_xy[0]) ** 2 + \
                          (sample_prediction_ball_global_xy[1] - sample_global_ball_pos_xy[1]) ** 2
                    mse_global.update(mse)

                print('\nBall Detection - \t Global stage: \t (x, y) - gt = ({}, {}), prediction = ({}, {})'.format(
                    sample_global_ball_pos_xy[0], sample_global_ball_pos_xy[1], sample_prediction_ball_global_xy[0],
                    sample_prediction_ball_global_xy[1]))

                sample_pred_org_x = sample_prediction_ball_global_xy[0] * (w_original / w)
                sample_pred_org_y = sample_prediction_ball_global_xy[1] * (h_original / h)

                # Process local ball stage
                if pred_ball_local is not None:
                    # Get target
                    local_ball_pos_xy = local_ball_pos_xy.cpu().numpy()  # Ground truth of the local stage
                    sample_local_ball_pos_xy = local_ball_pos_xy[sample_idx]  # Target
                    # Process the local stage
                    sample_pred_ball_local = pred_ball_local[sample_idx]
                    sample_prediction_ball_local_xy = get_prediction_ball_pos(sample_pred_ball_local, w,
                                                                              configs.thresh_ball_pos_mask)

                    # Calculate the MSE
                    if (sample_local_ball_pos_xy[0] > 0) and (sample_local_ball_pos_xy[1] > 0):
                        mse = (sample_prediction_ball_local_xy[0] - sample_local_ball_pos_xy[0]) ** 2 + (
                                sample_prediction_ball_local_xy[1] - sample_local_ball_pos_xy[1]) ** 2
                        mse_local.update(mse)
                        sample_pred_org_x += sample_prediction_ball_local_xy[0] - w / 2
                        sample_pred_org_y += sample_prediction_ball_local_xy[1] - h / 2

                    print('Ball Detection - \t Local stage: \t (x, y) - gt = ({}, {}), prediction = ({}, {})'.format(
                        sample_local_ball_pos_xy[0], sample_local_ball_pos_xy[1], sample_prediction_ball_local_xy[0],
                        sample_prediction_ball_local_xy[1]))

                print('Ball Detection - \t Overall: \t (x, y) - org: ({}, {}), prediction = ({}, {})'.format(
                    sample_org_ball_pos_xy[0], sample_org_ball_pos_xy[1], int(sample_pred_org_x),
                    int(sample_pred_org_y)))
                mse = (sample_org_ball_pos_xy[0] - sample_pred_org_x) ** 2 + (
                        sample_org_ball_pos_xy[1] - sample_pred_org_y) ** 2
                mse_overall.update(mse)

                # Process event stage
                if pred_events is not None:
                    sample_target_events = target_events[sample_idx].numpy()
                    sample_prediction_events = prediction_get_events(pred_events[sample_idx], configs.event_thresh)
                    print(
                        'Event Spotting - \t gt = (is land: {}, is shot: {}), prediction: (is land: {:.4f}, is shot: {:.4f})'.format(
                            sample_target_events[0], sample_target_events[1], pred_events[sample_idx][0],
                            pred_events[sample_idx][1]))
                    # Compute metrics
                    spce.update(SPCE(sample_prediction_events, sample_target_events, thresh=0.5))
                    pce.update(PCE(sample_prediction_events, sample_target_events))

                # Process segmentation stage
                if pred_seg is not None:
                    sample_target_seg = target_seg[sample_idx].transpose(1, 2, 0).astype(np.int)
                    sample_prediction_seg = get_prediction_seg(pred_seg[sample_idx], configs.seg_thresh)

                    # Calculate the IoU
                    iou = 2 * np.sum(sample_target_seg * sample_prediction_seg) / (
                            np.sum(sample_target_seg) + np.sum(sample_prediction_seg) + 1e-9)
                    iou_seg.update(iou)
                    
                    if iou >0 :
                        suuuu=suuuu+1
                    alll=alll+1
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # f_sample_target_seg = f_target_seg[sample_idx].transpose(1, 2, 0).astype(np.int)
                    # f_sample_prediction_seg = get_prediction_seg(pred_seg[sample_idx], configs.seg_thresh)
                    # y_pred = sample_prediction_seg > 0.5
                    # (tp, tn, fp1, fp2, fn) = outcome_track(sample_prediction_seg, sample_target_seg, 1)
                    # TP += tp
                    # TN += tn
                    # FP1 += fp1
                    # FP2 += fp2
                    # FN += fn
                    
                    
                    

                    print('Segmentation - \t \t IoU = {:.4f}'.format(iou))

                    if configs.save_test_output:
                        fig, axes = plt.subplots(nrows=batch_size, ncols=2, figsize=(10, 5))
                        plt.tight_layout()
                        axes.ravel()
                        axes[2 * sample_idx].imshow(sample_target_seg * 255)
                        axes[2 * sample_idx + 1].imshow(sample_prediction_seg * 255)
                        # title
                        target_title = 'target seg'
                        pred_title = 'pred seg'
                        if pred_events is not None:
                            target_title += ', is bounce: {}, is shot: {}'.format(sample_target_events[0],
                                                                                 sample_target_events[1])
                            pred_title += ', is bounce: {}, is shot: {}'.format(sample_prediction_events[0],
                                                                               sample_prediction_events[1])

                        axes[2 * sample_idx].set_title(target_title)
                        axes[2 * sample_idx + 1].set_title(pred_title)

                        plt.savefig(os.path.join(configs.saved_dir,
                                                 'batch_idx_{}_sample_idx_{}.jpg'.format(batch_idx, sample_idx)))

            if ((batch_idx + 1) % configs.print_freq) == 0:
                print(
                    'batch_idx: {} - Average iou_seg: {:.4f}, mse_global: {:.1f}, mse_local: {:.1f}, mse_overall: {:.1f}, pce: {:.4f} spce: {:.4f}'.format(
                        batch_idx, iou_seg.avg, mse_global.avg, mse_local.avg, mse_overall.avg, pce.avg, spce.avg))

            batch_time.update(time.time() - start_time)
            start_time = time.time()
    endtime = datetime.datetime.now()
    print ((endtime - starttime).seconds)
    print("FPS:",alll/(endtime - starttime).seconds)
    print(
        'Average iou_seg: {:.4f}, mse_global: {:.1f}, mse_local: {:.1f}, mse_overall: {:.1f}, pce: {:.4f} spce: {:.4f}'.format(
            iou_seg.avg, mse_global.avg, mse_local.avg, mse_overall.avg, pce.avg, spce.avg))
    print('Done testing')
    
    print("all test num:",alll)
    print("right:",suuuu)
    # print("iou for badimition bigger than 0:",suuuu)
    print("acc for tracking ball:",suuuu/alll)
    display_track(TP, TN, FP1, FP2, FN)
    print('Done testing')
    # import datetime
# starttime = datetime.datetime.now()
#long running
# endtime = datetime.datetime.now()
# print (endtime - starttime).seconds


if __name__ == '__main__':
    main()
    # print(alll)
    # print(suuuu)
