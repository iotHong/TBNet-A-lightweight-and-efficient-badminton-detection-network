"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.05.21
# email: nguyenmaudung93.kstn@gmail.com
# project repo: https://github.com/maudzung/TTNet-Realtime-for-Table-Tennis-Pytorch
-----------------------------------------------------------------------------------
# Description: The configurations of the project will be defined here
"""

import torch
import os
import datetime
import argparse
from easydict import EasyDict as edict
import sys

sys.path.append('../')

from utils.misc import make_folder


def parse_configs():
    parser = argparse.ArgumentParser(description='TTNet Implementation')
    parser.add_argument('--seed', type=int, default=2020,
                        help='re-produce the results with seed random')
    parser.add_argument('--saved_fn', type=str, default='ttnet', metavar='FN',
                        help='The name using for saving logs, models,...')
    ####################################################################
    ##############     Model configs            ###################
    ####################################################################
    parser.add_argument('-a', '--arch', type=str, default='ttnet', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--dropout_p', type=float, default=0.5, metavar='P',
                        help='The dropout probability of the model')
    parser.add_argument('--multitask_learning', action='store_true',
                        help='If true, the weights of different losses will be learnt (train).'
                             'If false, a regular sum of different losses will be applied')
    parser.add_argument('--no_local', action='store_true',
                        help='If true, no local stage for ball detection.')
    parser.add_argument('--no_event', action='store_true',
                        help='If true, no event spotting detection.')
    parser.add_argument('--no_seg', action='store_true',
                        help='If true, no segmentation module.')
    parser.add_argument('--pretrained_path', type=str, default=None, metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--overwrite_global_2_local', action='store_true',
                        help='If true, the weights of the local stage will be overwritten by the global stage.')

    ####################################################################
    ##############     Dataloader and Running configs            #######
    ####################################################################
    parser.add_argument('--working-dir', type=str, default='../../', metavar='PATH',
                        help='the ROOT working directory')
    parser.add_argument('--no-val', action='store_true',
                        help='If true, use all data for training, no validation set')
    parser.add_argument('--no-test', action='store_true',
                        help='If true, dont evaluate the model on the test set')
    parser.add_argument('--val-size', type=float, default=0.2,
                        help='The size of validation set')
    parser.add_argument('--smooth-labelling', action='store_true',
                        help='If true, smoothly make the labels of event spotting')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='mini-batch size (default: 8), this is the total'
                             'batch size of all GPUs on the current node when using'
                             'Data Parallel or Distributed Data Parallel')
    parser.add_argument('--print_freq', type=int, default=50, metavar='N',
                        help='print frequency (default: 50)')
    parser.add_argument('--checkpoint_freq', type=int, default=2, metavar='N',
                        help='frequency of saving checkpoints (default: 2)')
    parser.add_argument('--sigma', type=float, default=1., metavar='SIGMA',
                        help='standard deviation of the 1D Gaussian for the ball position target')
    parser.add_argument('--thresh_ball_pos_mask', type=float, default=0.05, metavar='THRESH',
                        help='the lower thresh for the 1D Gaussian of the ball position target')
    ####################################################################
    ##############     Training strategy            ###################
    ####################################################################

    parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                        help='the starting epoch')
    parser.add_argument('--num_epochs', type=int, default=30, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--minimum_lr', type=float, default=1e-7, metavar='MIN_LR',
                        help='minimum learning rate during training')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0., metavar='WD',
                        help='weight decay (default: 1e-6)')
    parser.add_argument('--optimizer_type', type=str, default='adam', metavar='OPTIMIZER',
                        help='the type of optimizer, it can be sgd or adam')
    parser.add_argument('--lr_type', type=str, default='plateau', metavar='SCHEDULER',
                        help='the type of the learning rate scheduler (steplr or ReduceonPlateau)')
    parser.add_argument('--lr_factor', type=float, default=0.5, metavar='FACTOR',
                        help='reduce the learning rate with this factor')
    parser.add_argument('--lr_step_size', type=int, default=5, metavar='STEP_SIZE',
                        help='step_size of the learning rate when using steplr scheduler')
    parser.add_argument('--lr_patience', type=int, default=3, metavar='N',
                        help='patience of the learning rate when using ReduceoPlateau scheduler')
    parser.add_argument('--earlystop_patience', type=int, default=None, metavar='N',
                        help='Early stopping the training process if performance is not improved within this value')
    parser.add_argument('--freeze_global', action='store_true',
                        help='If true, no update/train weights for the global stage of ball detection.')
    parser.add_argument('--freeze_local', action='store_true',
                        help='If true, no update/train weights for the local stage of ball detection.')
    parser.add_argument('--freeze_event', action='store_true',
                        help='If true, no update/train weights for the event module.')
    parser.add_argument('--freeze_seg', action='store_true',
                        help='If true, no update/train weights for the segmentation module.')

    ####################################################################
    ##############     Loss weight            ###################
    ####################################################################
    parser.add_argument('--bce_weight', type=float, default=0.5,
                        help='The weight of BCE loss in segmentation module, the dice_loss weight = 1- bce_weight')
    parser.add_argument('--global_weight', type=float, default=1.,
                        help='The weight of loss of the global stage for ball detection')
    parser.add_argument('--local_weight', type=float, default=1.,
                        help='The weight of loss of the local stage for ball detection')
    parser.add_argument('--event_weight', type=float, default=1.,
                        help='The weight of loss of the event spotting module')
    parser.add_argument('--seg_weight', type=float, default=1.,
                        help='The weight of BCE loss in segmentation module')

    ####################################################################
    ##############     Distributed Data Parallel            ############
    ####################################################################
    parser.add_argument('--world-size', default=-1, type=int, metavar='N',
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, metavar='N',
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:29500', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--gpu_idx', default=None, type=int,
                        help='GPU index to use.')
    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    ####################################################################
    ##############     Evaluation configurations     ###################
    ####################################################################
    parser.add_argument('--evaluate', action='store_true',
                        help='only evaluate the model, not training')
    parser.add_argument('--resume_path', type=str, default=None, metavar='PATH',
                        help='the path of the resumed checkpoint')
    parser.add_argument('--use_best_checkpoint', action='store_true',
                        help='If true, choose the best model on val set, otherwise choose the last model')
    parser.add_argument('--seg_thresh', type=float, default=0.5,
                        help='threshold of the segmentation output')
    parser.add_argument('--event_thresh', type=float, default=0.5,
                        help='threshold of the event spotting output')
    parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the image of testing phase will be saved')

    ####################################################################
    ##############     Demonstration configurations     ###################
    ####################################################################
    parser.add_argument('--video_path', type=str, default=None, metavar='PATH',
                        help='the path of the video that needs to demo')
    parser.add_argument('--output_format', type=str, default='text', metavar='PATH',
                        help='the type of the demo output')
    parser.add_argument('--show_image', action='store_true',
                        help='If true, show the image during demostration')
    parser.add_argument('--save_demo_output', action='store_true',
                        help='If true, the image of demonstration phase will be saved')

    configs = edict(vars(parser.parse_args()))

    ####################################################################
    ############## Hardware configurations ############################
    ####################################################################
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda')
    configs.ngpus_per_node = torch.cuda.device_count()

    configs.pin_memory = True

    ####################################################################
    ##############     Data configs            ###################
    ####################################################################
    configs.dataset_dir = os.path.join(configs.working_dir, 'dataset')
    configs.train_game_list = ['match1_1_01_00', 'match1_1_02_00', 'match1_1_02_01', 'match1_1_02_02', 'match1_1_02_03', 'match1_1_02_04', 'match1_1_03_04', 'match1_1_03_06', 'match1_1_06_06', 'match1_1_06_08', 'match1_1_06_09', 'match2_1_00_02', 'match2_1_02_03', 'match2_1_04_03', 'match2_1_06_08', 'match2_1_06_09', 'match2_1_08_11', 'match2_1_08_12', 'match2_1_09_12', 'match3_1_01_00', 'match3_1_08_10', 'match3_1_12_17', 'match3_2_04_07', 'match3_2_10_12', 'match3_2_18_15', 'match3_3_11_10', 'match4_1_03_02', 'match4_1_15_10', 'match4_2_02_05', 'match4_2_05_07', 'match4_2_14_17', 'match4_3_02_00', 'match4_3_07_05', 'match4_3_18_17', 'match5_1_01_01', 'match5_1_01_02', 'match5_1_19_18', 'match5_1_21_19', 'match5_2_15_13', 'match5_2_20_17', 'match6_1_02_00', 'match6_1_05_03', 'match6_1_11_04', 'match6_1_15_06', 'match6_1_19_12', 'match7_1_02_01', 'match7_1_12_13', 'match7_2_14_15', 'match7_3_08_05', 'match8_1_01_00', 'match8_1_05_13', 'match8_2_03_06', 'match8_2_10_12', 'match8_3_02_00', 'match8_3_15_08', 'match8_3_17_12', 'match8_3_21_13', 'match9_1_01_03', 'match9_1_02_03', 'match9_1_04_05', 'match9_1_05_06', 'match9_1_06_06', 'match9_1_07_07', 'match9_1_07_10', 'match9_1_07_11', 'match9_1_07_14', 'match10_1_03_01', 'match10_1_03_03', 'match10_1_12_16', 'match10_2_04_02', 'match10_2_14_08', 'match11_1_03_01', 'match11_1_07_06', 'match11_1_13_13', 'match11_2_05_00', 'match11_2_15_04', 'match12_1_01_00', 'match12_1_10_12', 'match12_2_01_01', 'match12_2_05_14', 'match12_3_03_03', 'match13_1_09_10', 'match13_1_17_15', 'match13_2_06_05', 'match13_2_07_05', 'match13_2_09_08', 'match14_1_17_14', 'match14_2_13_06', 'match14_2_15_10', 'match14_2_19_13', 'match14_2_21_17', 'match15_1_21_12', 'match15_2_14_08', 'match15_2_16_12', 'match15_2_18_14', 'match15_2_19_14', 'match16_1_03_06', 'match16_1_13_20', 'match16_3_12_06', 'match16_3_14_09', 'match16_3_17_16', 'match17_1_02_02', 'match17_1_15_13', 'match17_2_08_05', 'match17_2_15_11', 'match17_2_18_11', 'match18_1_06_12', 'match18_2_02_02', 'match18_3_03_05', 'match18_3_12_14', 'match18_3_16_17', 'match18_3_20_19', 'match19_1_01_03', 'match19_1_07_08', 'match19_2_12_06', 'match19_2_14_08', 'match20_1_09_05', 'match20_1_11_10', 'match20_2_00_01', 'match20_2_05_08', 'match20_2_07_08', 'match20_2_19_14', 'match21_1_02_01', 'match21_1_16_17', 'match21_1_19_19', 'match21_2_02_03', 'match21_2_04_04', 'match21_2_09_08', 'match21_2_12_08', 'match22_1_02_01', 'match22_1_07_02', 'match22_2_17_18', 'match22_2_18_18', 'match22_3_15_13', 'match23_1_06_04', 'match23_1_17_13', 'match23_2_02_03', 'match23_2_07_03', 'match23_2_12_08']
    configs.test_game_list = ['test_match1_1_05_02', 'test_match1_1_05_03', 'test_match1_1_06_03', 'test_match1_1_07_03', 'test_match1_1_07_04', 'test_match1_1_07_06', 'test_match1_1_09_06', 'test_match1_2_02_07', 'test_match1_2_03_08', 'test_match1_2_03_10', 'test_match2_1_03_03', 'test_match2_1_04_04', 'test_match2_1_19_15', 'test_match2_2_02_05', 'test_match2_2_08_12', 'test_match3_1_02_00', 'test_match3_1_03_02', 'test_match3_1_05_02', 'test_match3_1_05_03', 'test_match3_1_06_05', 'test_match3_1_06_06', 'test_match3_1_08_08', 'test_match3_1_08_09', 'test_match3_1_09_12', 'test_match3_1_09_15', 'test_match3_1_10_16']
    configs.events_dict = {
        '1': 0,#bounce=>land
        '2': 1,#shot
        "empty":2
    }
    configs.events_weights_loss_dict = {
        '1': 3.,#bounce=>land
        '2': 1.#shot
    }
    configs.events_weights_loss = (configs.events_weights_loss_dict['1'], configs.events_weights_loss_dict['2'])
    configs.num_events = len(configs.events_weights_loss_dict)  # Just "bounce" and "net hits"
    configs.num_frames_sequence = 9

    configs.org_size = (1280, 720)
    configs.input_size = (320, 128)

    configs.tasks = ['global', 'local', 'event', 'seg']
    if configs.no_local:
        if 'local' in configs.tasks:
            configs.tasks.remove('local')
        if 'event' in configs.tasks:
            configs.tasks.remove('event')
    if configs.no_event:
        if 'event' in configs.tasks:
            configs.tasks.remove('event')
    if configs.no_seg:
        if 'seg' in configs.tasks:
            configs.tasks.remove('seg')

    # Compose loss weight for tasks, normalize the weights later
    loss_weight_dict = {
        'global': configs.global_weight,
        'local': configs.local_weight,
        'event': configs.event_weight,
        'seg': configs.seg_weight
    }
    configs.tasks_loss_weight = [loss_weight_dict[task] for task in configs.tasks]

    configs.freeze_modules_list = []
    if configs.freeze_global:
        configs.freeze_modules_list.append('ball_global_stage')
    if configs.freeze_local:
        configs.freeze_modules_list.append('ball_local_stage')
    if configs.freeze_event:
        configs.freeze_modules_list.append('events_spotting')
    if configs.freeze_seg:
        configs.freeze_modules_list.append('segmentation')

    ####################################################################
    ############## logs, Checkpoints, and results dir ########################
    ####################################################################
    configs.checkpoints_dir = os.path.join(configs.working_dir, 'checkpoints', configs.saved_fn)
    configs.logs_dir = os.path.join(configs.working_dir, 'logs', configs.saved_fn)
    configs.use_best_checkpoint = True

    if configs.use_best_checkpoint:
        configs.saved_weight_name = os.path.join(configs.checkpoints_dir, '{}_best.pth'.format(configs.saved_fn))
    else:
        configs.saved_weight_name = os.path.join(configs.checkpoints_dir, '{}.pth'.format(configs.saved_fn))

    configs.results_dir = os.path.join(configs.working_dir, 'results')

    make_folder(configs.checkpoints_dir)
    make_folder(configs.logs_dir)
    make_folder(configs.results_dir)

    if configs.save_test_output:
        configs.saved_dir = os.path.join(configs.results_dir, configs.saved_fn)
        make_folder(configs.saved_dir)

    if configs.save_demo_output:
        configs.save_demo_dir = os.path.join(configs.results_dir, 'demo', configs.saved_fn)
        make_folder(configs.save_demo_dir)

    return configs


if __name__ == "__main__":
    configs = parse_configs()
    print(configs)

    print(datetime.date.today())
    print(datetime.datetime.now().year)
