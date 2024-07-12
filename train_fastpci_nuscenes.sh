#!/bin/bash
python train_fastpci.py  \
--batch_size 4 \
--epochs 200 \
--gpu 0 \
--data_root data/NL-Drive/train/ \
--scene_list data/NL-Drive/train_scene02_list.txt \
--npoints 8192 \
--save_dir experiments/nuscenes/ \
> .log_fastpci_nuscenes_train 2>&1
