#!/bin/bash
python train_fastpci.py  \
--batch_size 4 \
--epochs 100 \
--gpu 0 \
--data_root data/NL-Drive/train/ \
--scene_list data/NL-Drive/train_scene_list.txt \
--npoints 8192 \
--save_dir experiments/ko/ \
> .log_fastpci_ko_train 2>&1
