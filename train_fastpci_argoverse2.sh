#!/bin/bash
python train_fastpci.py  \
--batch_size 4 \
--epochs 200 \
--gpu 0 \
--data_root data/NL-Drive/train/ \
--scene_list data/NL-Drive/train_scene01_list.txt \
--npoints 8192 \
--save_dir experiments/argoverse2/ \
> .log_fastpci_argoverse2_train 2>&1
