# coding:utf-8

import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import numpy as np

from data.no_norm_datasets import NLDriveDataset
from models.afmf_pcit_prelu import SceneFlowPWC
from models.utils import chamfer_loss, EMD
import time
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='FastPCI')
    # training setting
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay.')
    parser.add_argument('--gpu', type=str, default='0')  
    parser.add_argument('--resume', type=bool, default=False, help='whether continue the training')
    parser.add_argument('--save_dir', type=str, default='')
    # dataset setting
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--scene_list', type=str, default='')
    parser.add_argument('--interval', type=int, default=4)
    parser.add_argument('--num_frames', type=int, default=4)
    parser.add_argument('--npoints', type=int, default=8192)
    parser.add_argument('--t_begin', type=float, default=0., help='Time stamp of the first input frame.')
    parser.add_argument('--t_end', type=float, default=1., help='Time stamp of the last input frame.')

    return parser.parse_args()

def init_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.fill_(0.0)

def get_timestamp(args):
    time_seq = [t for t in np.linspace(args.t_begin, args.t_end, args.num_frames)] 
    t_left = time_seq[args.num_frames//2 - 1]
    t_right = time_seq[args.num_frames//2]
    time_intp = [t for t in np.linspace(t_left, t_right, args.interval+1)]
    time_intp = time_intp[1:-1]  
        
    return time_seq, time_intp

def train(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    train_dataset = NLDriveDataset(args.data_root, args.scene_list, args.npoints, args.interval, args.num_frames)

    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size,
                              num_workers=8,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)
    
    net = SceneFlowPWC().cuda()
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('the number of network parameters: {}'.format(total_params))

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=args.weight_decay)
    
    if args.resume:
        experiments = 'experiments/ko/ckpt_best_95.pth'
        checkpoint = torch.load(experiments)
        net = checkpoint['net']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']
        start_epoch = checkpoint['epoch']
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5, last_epoch = start_epoch - 1)
    else:
        start_epoch = 0
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5, last_epoch = start_epoch - 1)

    best_train_loss = float('inf')
    _, time_inp = get_timestamp(args)

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        count = 0
        total_loss = 0
        l2 = 0
        l22 = 0

        pbar = tqdm(enumerate(train_loader))

        for i, (input, gt) in pbar:
            for i in range(len(input)):
                input[i] = input[i].permute(0,2,1).cuda().contiguous().float()
            for i in range(len(gt)):
                gt[i] = gt[i].permute(0,2,1).cuda().contiguous().float()

            j = random.randint(0,2)
            t = time_inp[j]
            gtgt = gt[j]

            optimizer.zero_grad()
            _, pc_pred, warped_list, gt_list, warped_pc2t = net(input[1], input[2], t, gtgt, train=True)


            loss = chamfer_loss(pc_pred, gtgt)
            loss1 = chamfer_loss(warped_list[0], gtgt)
            loss2 = chamfer_loss(warped_pc2t, gtgt)
            multiscaleloss = 0
            alpha = [1.0, 0.8, 0.4]  #, 0.2
            for l in range(len(alpha)-1):
                temp = chamfer_loss(warped_list[l+1], gt_list[l])
                multiscaleloss += alpha[l+1] * temp

    
            losssum = loss + loss1 + 0.25*multiscaleloss + loss2 


            losssum.backward()
            optimizer.step()

            count += 1
            total_loss += loss.item()
            l2 += loss1.item()
            l22 += loss2.item()
            if i % 10 == 0:
                print('Train Epoch:{}[{}/{}({:.0f}%)]\tLoss: {:.6f}\tloss1: {:.6f}\tmultiscaleloss: {:.6f}\tLoss2: {:.6f}'.format(
                    epoch+1, i, len(train_loader), 100. * i/len(train_loader), loss.item(), loss1.item(), multiscaleloss.item(), loss2.item()
                ))
            
        scheduler.step()
        total_loss = total_loss/count
        l2 = l2/count
        l22 = l22/count
        
        print('Epoch ', epoch+1, 'finished ', 'loss = ', total_loss, 'loss1 = ', l2, 'loss2=', l22)

        if total_loss < best_train_loss:
            best_train_loss = total_loss
            checkpoint = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch
                }
            if not os.path.isdir(args.save_dir):
                os.mkdir(args.save_dir)
            torch.save(checkpoint, args.save_dir + 'ckpt_best_'+str(epoch)+'.pth')
            

        print('Best train loss: {:.4f}'.format(best_train_loss))
        one_epoch_time = time.time() - start_time
        print('epoch:',epoch, 'one_epoch_time:', one_epoch_time)

if __name__ == '__main__':
    args = parse_args()
    
    train(args)