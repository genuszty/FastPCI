import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# We utilize kaolin to implement layers for FlowNet3D
# website: https://github.com/NVIDIAGameWorks/kaolin
# import kaolin as kal
#from kaolin.models.PointNet2 import furthest_point_sampling
#from kaolin.models.PointNet2 import fps_gather_by_index
#from kaolin.models.PointNet2 import ball_query
#from kaolin.models.PointNet2 import group_gather_by_index
#from kaolin.models.PointNet2 import three_nn
#from kaolin.models.PointNet2 import three_interpolate
from .common import fps, gather_points, ball_query, three_nn, three_interpolate
# We utilize pytorch3d to implement k-nearest-neighbor search
# website: https://github.com/facebookresearch/pytorch3d
from pytorch3d.ops import knn_points, knn_gather

from .utils import pdist2squared

'''
Layers for FlowNet3D
'''
class Sample(nn.Module):
    '''
    Furthest point sample
    '''
    def __init__(self, num_points):
        super(Sample, self).__init__()
        
        self.num_points = num_points
        
    def forward(self, points):
        new_points_ind = fps(points.permute(0, 2, 1).contiguous(), self.num_points)  # 获得质心点的索引
        #print('sample_new_points_ind:', new_points_ind.shape)
        new_points = gather_points(points.permute(0, 2, 1).contiguous(), new_points_ind)  # 根据索引从raw pc中采样M个点作为质点
        return new_points  # 局部区域质心的坐标

class Group(nn.Module):
    '''
    kNN group for FlowNet3D
    '''
    def __init__(self, radius, num_samples, knn=False):
        super(Group, self).__init__()
        
        self.radius = radius
        self.num_samples = num_samples
        self.knn = knn
        
    def forward(self, points, new_points, features):
        points = points.permute(0, 2, 1).contiguous()
        features = features.permute(0, 2, 1).contiguous()
        #print('points:',points.shape)
        #print('new_points:',new_points.shape)
        #print('features:', features.shape)
        if self.knn:
            dist = pdist2squared(points.permute(0, 2, 1).contiguous(), new_points)   # 计算点与最远点样本之间的欧式距离
            #print('dist:', dist.shape)
            ind = dist.topk(self.num_samples, dim=1, largest=False)[1].int().permute(0, 2, 1).contiguous()  # 选出前num_samples个距离最近的index
        else:
            ind = ball_query(points, new_points, self.radius, self.num_samples)   # 以每个质心为球心，并行找出半径radius内的K个点的索引
        #print('ind:',ind.shape)  
        grouped_points = gather_points(points, ind)   # 根据索引在raw pc中找到以质心为中心的局部点集。
        #print('grouped_points:', grouped_points.shape)
        if self.knn:
            grouped_points_new = grouped_points - new_points.unsqueeze(3).permute(0,2,3,1)
        else:  
            grouped_points_new = grouped_points - new_points.unsqueeze(2)         # 再从聚类好的组里去掉最远点。
        #print('grouped_points_new:', grouped_points_new.shape)
        grouped_features = gather_points(features, ind)   # 根据索引在feature中找到以质心为中心的local feature tensor
        new_features = torch.cat([grouped_points_new, grouped_features], dim=3)
        new_features = new_features.permute(0, 3, 1, 2)
        return new_features

class SetConv(nn.Module):
    def __init__(self, num_points, radius, num_samples, in_channels, out_channels):
        super(SetConv, self).__init__()
        
        self.sample = Sample(num_points)   # 采样质心点的个数
        self.group = Group(radius, num_samples)  # 以radius为半径进行球查询聚类
        
        layers = []
        out_channels = [in_channels+3, *out_channels]
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i - 1], out_channels[i], 1, bias=True), nn.BatchNorm2d(out_channels[i], eps=0.001), nn.ReLU()]
        self.conv = nn.Sequential(*layers)
        
    def forward(self, points, features):
        new_points = self.sample(points)   # 得到所有局部区域的质心
        new_features = self.group(points, new_points, features)  # 以质点为中心，在raw pc中找到邻点，建立局部点集
        #print('_____new_features______:', new_features.shape)
        new_features = self.conv(new_features) # 从聚类好的组中进行特征抽取，共3层conv2d
        new_features = new_features.max(dim=3)[0]   # 将局部区域模式编码成特征向量。
        new_points = new_points.permute(0,2,1)
        return new_points, new_features  # 

class FlowEmbedding(nn.Module):
    def __init__(self, num_samples, in_channels, out_channels):
        super(FlowEmbedding, self).__init__()
        
        self.num_samples = num_samples
        
        self.group = Group(None, self.num_samples, knn=True)
        
        layers = []
        out_channels = [2*in_channels+3, *out_channels]
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i - 1], out_channels[i], 1, bias=True), nn.BatchNorm2d(out_channels[i], eps=0.001), nn.ReLU()]
        self.conv = nn.Sequential(*layers)
        
    def forward(self, points1, points2, features1, features2):
        new_features = self.group(points2, points1, features2)   # 获得第二帧中所有qj
        new_features = torch.cat([new_features, features1.unsqueeze(3).expand(-1, -1, -1, self.num_samples)], dim=1)  # 聚合所有邻qj
        new_features = self.conv(new_features)
        new_features = new_features.max(dim=3)[0]
        return new_features

class SetUpConv(nn.Module):
    def __init__(self, num_samples, in_channels1, in_channels2, out_channels1, out_channels2):
        super(SetUpConv, self).__init__()
        
        self.group = Group(None, num_samples, knn=True)
        
        layers = []
        out_channels1 = [in_channels1+3, *out_channels1]
        for i in range(1, len(out_channels1)):
            layers += [nn.Conv2d(out_channels1[i - 1], out_channels1[i], 1, bias=True), nn.BatchNorm2d(out_channels1[i], eps=0.001), nn.ReLU()]
        self.conv1 = nn.Sequential(*layers)
        
        layers = []
        if len(out_channels1) == 1:
            out_channels2 = [in_channels1+in_channels2+3, *out_channels2]
        else:
            out_channels2 = [out_channels1[-1]+in_channels2, *out_channels2]
        for i in range(1, len(out_channels2)):
            layers += [nn.Conv2d(out_channels2[i - 1], out_channels2[i], 1, bias=True), nn.BatchNorm2d(out_channels2[i], eps=0.001), nn.ReLU()]
        self.conv2 = nn.Sequential(*layers)
        
    def forward(self, points1, points2, features1, features2):
        new_features = self.group(points1, points2, features1)
        new_features = self.conv1(new_features)  # 1层conv
        new_features = new_features.max(dim=3)[0]
        new_features = torch.cat([new_features, features2], dim=1)
        new_features = new_features.unsqueeze(3)
        new_features = self.conv2(new_features)
        new_features = new_features.squeeze(3)
        return new_features

class FeaturePropagation(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels):
        super(FeaturePropagation, self).__init__()
        
        layers = []
        out_channels = [in_channels1+in_channels2, *out_channels]
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i - 1], out_channels[i], 1, bias=True), nn.BatchNorm2d(out_channels[i], eps=0.001), nn.ReLU()]
        self.conv = nn.Sequential(*layers)
        
    def forward(self, points1, points2, features1, features2):
        #print('!!!!!!!feature:',features1.shape)
        dist, ind = three_nn(points2.permute(0, 2, 1).contiguous(), points1.permute(0, 2, 1).contiguous())
        #print('!!!!!!!ind:', ind.shape, 'distdist:', dist.shape)
        dist = dist * dist
        #print('___distdist___:', dist.shape)
        dist[dist < 1e-10] = 1e-10
        inverse_dist = 1.0 / dist
        norm = torch.sum(inverse_dist, dim=2, keepdim=True)
        weights = inverse_dist / norm
        a = gather_points(features1.permute(0,2,1).contiguous(), ind).permute(0,3,1,2)
        #print('$$$$$$$$$$________new_features:',a.shape)
        new_features = torch.sum(a * weights.unsqueeze(1), dim = 3)
        #print('________new_features________:',new_features.shape)
        new_features = torch.cat([new_features, features2], dim=1)
        #print('________new_features:',new_features.shape)
        new_features = self.conv(new_features.unsqueeze(3)).squeeze(3)
        #print('________new_features?????????:', new_features.shape)
        return new_features

class PointsFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointsFusion, self).__init__()

        layers = []
        out_channels = [in_channels, *out_channels]
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i - 1], out_channels[i], 1, bias=True), nn.BatchNorm2d(out_channels[i], eps=0.001), nn.ReLU()]
        
        self.conv = nn.Sequential(*layers)
    
    def knn_group(self, points1, points2, features2, k):
        '''
        For each point in points1, query kNN points/features in points2/features2
        Input:
            points1: [B,3,N]
            points2: [B,3,N]
            features2: [B,C,N]
        Output:
            new_features: [B,4,N]
            nn: [B,3,N]
            grouped_features: [B,C,N]
        '''
        points1 = points1.permute(0,2,1).contiguous()
        points2 = points2.permute(0,2,1).contiguous()
        _, nn_idx, nn = knn_points(points1, points2, K=k, return_nn=True)
        points_resi = nn - points1.unsqueeze(2).repeat(1,1,k,1)
        grouped_dist = torch.norm(points_resi, dim=-1, keepdim=True)
        grouped_features = knn_gather(features2.permute(0,2,1), nn_idx)
        new_features = torch.cat([points_resi, grouped_dist], dim=-1)

        return new_features.permute(0,3,1,2).contiguous(),\
            nn.permute(0,3,1,2).contiguous(),\
            grouped_features.permute(0,3,1,2).contiguous()
    
    def forward(self, points1, points2, features1, features2, k, t):
        '''
        Input:
            points1: [B,3,N]
            points2: [B,3,N]
            features1: [B,C,N] (only for inference of additional features)
            features2: [B,C,N] (only for inference of additional features)
            k: int, number of kNN cluster
            t: [B], time step in (0,1)
        Output:
            fused_points: [B,3+C,N]
        '''
        N = points1.shape[-1]   # 点数
        B = points1.shape[0]    # batch size

        new_features_list = []
        new_grouped_points_list = []
        new_grouped_features_list = []

        for i in range(B):
            t1 = t[i]
            new_points1 = points1[i:i+1,:,:]
            new_points2 = points2[i:i+1,:,:]
            new_features1 = features1[i:i+1,:,:]
            new_features2 = features2[i:i+1,:,:]

            N2 = int(N*t1)   # 设置从warped帧中采样点的个数
            N1 = N - N2

            k2 = int(k*t1)
            k1 = k - k2

            randidx1 = torch.randperm(N)[:N1]   # 把N个数打散，取前N1个数
            randidx2 = torch.randperm(N)[:N2]   # 把N个数打散，取前N2个数
            # 从warped_pc1中取N1个点，从warped_pc2中取N2个点，cat起来
            new_points = torch.cat((new_points1[:,:,randidx1], new_points2[:,:,randidx2]), dim=-1)   # [B,3,N]

            new_features1, grouped_points1, grouped_features1 = self.knn_group(new_points, new_points1, new_features1, k1)
            new_features2, grouped_points2, grouped_features2 = self.knn_group(new_points, new_points2, new_features2, k2)

            new_features = torch.cat((new_features1, new_features2), dim=-1)
            new_grouped_points = torch.cat((grouped_points1, grouped_points2), dim=-1)
            new_grouped_features = torch.cat((grouped_features1, grouped_features2), dim=-1)

            new_features_list.append(new_features)
            new_grouped_points_list.append(new_grouped_points)
            new_grouped_features_list.append(new_grouped_features)

        new_features = torch.cat(new_features_list, dim=0)                  # [B,4,N,K]
        new_grouped_points = torch.cat(new_grouped_points_list, dim=0)      # [B,3,N,K]
        new_grouped_features = torch.cat(new_grouped_features_list, dim=0)  # [B,1,N,K]

        new_features = self.conv(new_features)  
        new_features = torch.max(new_features, dim=1, keepdim=False)[0]  # [B,N,K]
        weights = F.softmax(new_features, dim=-1)

        C = features1.shape[1]
        weights = weights.unsqueeze(1).repeat(1,3+C,1,1)
        fused_points = torch.cat([new_grouped_points, new_grouped_features], dim=1)
        fused_points = torch.sum(torch.mul(weights, fused_points), dim=-1, keepdim=False)  # [B,4,N]

        return fused_points

def knn_group_withI(points1, points2, intensity2, k):
    '''
    Input:
        points1: [B,3,N]
        points2: [B,3,N]
        intensity2: [B,1,N]
    '''
    points1 = points1.permute(0,2,1).contiguous()
    points2 = points2.permute(0,2,1).contiguous()
    _, nn_idx, nn = knn_points(points1, points2, K=k, return_nn=True)
    points_resi = nn - points1.unsqueeze(2).repeat(1,1,k,1) # [B,M,k,3]
    grouped_dist = torch.norm(points_resi, dim=-1, keepdim=True)
    grouped_features = knn_gather(intensity2.permute(0,2,1), nn_idx) # [B,M,k,1]
    new_features = torch.cat([points_resi, grouped_dist], dim=-1)
    
    # [B,5,M,k], [B,3,M,k], [B,1,M,k]
    return new_features.permute(0,3,1,2).contiguous(), \
        nn.permute(0,3,1,2).contiguous(), \
        grouped_features.permute(0,3,1,2).contiguous()