#!/bin/bash
python test_fastpci.py  \
--batch_size 4 \
--gpu 0 \
--data_root data/NL-Drive/test/ \
--scene_list data/NL-Drive/test_scene01_list.txt \
--npoints 8192 \
--pretrain_model experiments/argoverse2/ckpt_best_***.pth \
> .log_fastpci_argoverse2_test 2>&1
