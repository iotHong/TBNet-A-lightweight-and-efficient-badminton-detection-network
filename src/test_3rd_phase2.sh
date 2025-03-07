#!/bin/bash

python test-Copy1.py \
  --working-dir '../' \
  --saved_fn 'ttnet_1td_phase' \
  --gpu_idx 0 \
  --batch_size 1 \
  --pretrained_path ../checkpoints/ttnet_2st_phase/ttnet_2st_phase_epoch_30.pth \
  --seg_thresh 0.5 \
  --no_local \
  --no_event \
  --smooth-labelling