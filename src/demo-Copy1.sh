#!/bin/bash

python demo-Copy1.py \
  --working-dir '../' \
  --saved_fn 'deme_4_17' \
  --arch 'ttnet' \
  --gpu_idx 0 \
  --pretrained_path ../checkpoints/TBnet_1st_phase_418_retry/TBnet_1st_phase_418_retry_epoch_30.pth \
  --seg_thresh 0.5 \
  --event_thresh 0.5 \
  --thresh_ball_pos_mask 0.05 \
  --output_format 'video'\
  --video_path ../dataset/test/videos/test_match1_1_05_03.mp4 \
  --show_image \
  --save_demo_output