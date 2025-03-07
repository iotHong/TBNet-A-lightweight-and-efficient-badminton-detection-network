#!/bin/bash

python demo.py \
  --working-dir '../' \
  --saved_fn 'deme_4_16' \
  --arch 'ttnet' \
  --gpu_idx 0 \
  --pretrained_path ../checkpoints/ttnet_1st_phase/ttnet_1st_phase_epoch_30.pth \
  --seg_thresh 0.5 \
  --event_thresh 0.5 \
  --thresh_ball_pos_mask 0.05 \
  --output_format 'video'\
  --video_path ../dataset/test/videos/test_match1_1_05_02.mp4 \
  --show_image \
  --save_demo_output