#!/bin/bash

python test.py \
  --working-dir '../' \
  --saved_fn 'TBnet_1st' \
  --gpu_idx 0 \
  --batch_size 1 \
  --pretrained_path ../checkpoints/ttnet_1st_phase/ttnet_1st_phase_epoch_30.pth \
  --seg_thresh 0.5 \
  --event_thresh 0.5 \
  --smooth-labelling