# coding=gbk

import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from data.no_norm_datasets import NLDriveDataset
from models.afmf_pcit_prelu import SceneFlowPWC
from models.utils import chamfer_loss, EMD

import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    # dataset setting
    parser.add_argument('--data_root', type=str, default='./dataset/NL-Drive/', help='Dataset path.')
    parser.add_argument('--scene_list', type=str, default='./data/scene_list.txt', help='Path of the scene list to be used in the dataset.')
    parser.add_argument('--interval', type=int, default=4, help='Interval frames between point cloud sequence.')
    parser.add_argument('--npoints', type=int, default=8192, help='Point number [default: 8192 for NL_Drive].')
    parser.add_argument('--num_frames', type=int, default=4, help='Number of input point cloud frames.')
    parser.add_argument('--t_begin', type=float, default=0., help='Time stamp of the first input frame.')
    parser.add_argument('--t_end', type=float, default=1., help='Time stamp of the last input frame.')
    # test setting
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--pretrain_model', type=str, default='./pretrain_model/interp_kitti.pth')

    return parser.parse_args()

def get_timestamp(args):
    time_seq = [t for t in np.linspace(args.t_begin, args.t_end, args.num_frames)] 
    t_left = time_seq[args.num_frames//2 - 1]
    t_right = time_seq[args.num_frames//2]
    time_intp = [t for t in np.linspace(t_left, t_right, args.interval+1)]
    time_intp = time_intp[1:-1]  
    return time_seq, time_intp

def test(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    test_dataset = NLDriveDataset(data_root = args.data_root, 
                                       scene_list = args.scene_list, 
                                       interval = args.interval,     # 4
                                       num_points = args.npoints, #8192
                                       num_frames = args.num_frames) # 4:1,5,9,13
    
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             num_workers=16,
                             pin_memory=True,
                             drop_last=False)
    
    net = SceneFlowPWC().cuda()
    checkpoint = torch.load(args.pretrain_model)
    net.load_state_dict(checkpoint['net'])

    net.eval()
    _, time_inp = get_timestamp(args)


    with torch.no_grad():
        chamfer_loss_list = [[],[],[]]
        emd_loss_list = [[],[],[]]

        pbar = tqdm(enumerate(test_loader))
        for i, (input, gt) in pbar:
            for idx in range(len(input)):
                input[idx] = input[idx].permute(0,2,1).cuda().contiguous().float()
            for idxx in range(len(gt)):
                gt[idxx] = gt[idxx].permute(0,2,1).cuda().contiguous().float()
            
            for j in range(len(time_inp)):
                t = time_inp[j]
                pred_pc = net(input[1], input[2], t)

                cd = chamfer_loss(pred_pc, gt[j])
                emd = EMD(pred_pc, gt[j])

                cd = cd.squeeze().cpu().numpy()
                emd = emd.squeeze().cpu().numpy()

                chamfer_loss_list[j].append(cd)
                emd_loss_list[j].append(emd)

                pbar.set_description('Frame {}||[{}/{}]: CD:{:.3} EMD:{:.3}'.format(i, j, len(test_loader), cd, emd))


        chamfer_loss_array0 = np.array(chamfer_loss_list[0])
        emd_loss_array0 = np.array(emd_loss_list[0])
        mean_chamfer_loss0 = np.mean(chamfer_loss_array0)
        mean_emd_loss0 = np.mean(emd_loss_array0)

        chamfer_loss_array1 = np.array(chamfer_loss_list[1])
        emd_loss_array1 = np.array(emd_loss_list[1])
        mean_chamfer_loss1 = np.mean(chamfer_loss_array1)
        mean_emd_loss1 = np.mean(emd_loss_array1)

        chamfer_loss_array2 = np.array(chamfer_loss_list[2])
        emd_loss_array2 = np.array(emd_loss_list[2])
        mean_chamfer_loss2 = np.mean(chamfer_loss_array2)
        mean_emd_loss2 = np.mean(emd_loss_array2)
        
        chamfer_loss_array = np.array(chamfer_loss_list)
        emd_loss_array = np.array(emd_loss_list)
        mean_chamfer_loss = np.mean(chamfer_loss_array)
        mean_emd_loss = np.mean(emd_loss_array)

    print("Frame1: Mean chamfer distance: ", mean_chamfer_loss0)
    print("Frame1: Mean earth mover's distance: ", mean_emd_loss0)
    print("Frame2: Mean chamfer distance: ", mean_chamfer_loss1)
    print("Frame2: Mean earth mover's distance: ", mean_emd_loss1)
    print("Frame3: Mean chamfer distance: ", mean_chamfer_loss2)
    print("Frame3: Mean earth mover's distance: ", mean_emd_loss2)
    print("-------------------------------------------")
    print("Average: Mean chamfer distance: ", mean_chamfer_loss)
    print("Average: Mean earth mover's distance: ", mean_emd_loss)

if __name__ == '__main__':
    args = parse_args()
    test(args)