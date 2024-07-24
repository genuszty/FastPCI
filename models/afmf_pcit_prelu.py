import torch.nn as nn
import torch
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from pytorch3d.ops import knn_points
from models.pointnet2.pointnet2_utils import gather_operation, grouping_operation, furthest_point_sample
from models.pointT_layer2 import FlowRefineNet, TransformerBlock, square_distance  #, index_points
#from models.amformer import MotionFormer
#from models.common import gather_points
#from models.utils import chamfer_loss

scale = 1.0

class SceneFlowPWC(nn.Module):
    def __init__(self):
        super(SceneFlowPWC, self).__init__()

        flow_nei = 32
        feat_nei = 16
        self.scale = scale
        
        self.feature_bone = MotionFormer()

        #l0: 8192
        self.level0 = Conv1d(3, 32)
        self.level0_1 = Conv1d(32, 32)
        #self.shape0 = Shapenet(32, 32)
        self.cost0 = PointConvFlow(flow_nei, 32 + 32 + 32 + 32 + 3, [32, 32])
        self.flow0 = SceneFlowEstimatorPointConv(32 + 64, 32)
        self.level0_2 = Conv1d(32, 64)
        c = 32
        self.rf_block0 = FlowRefineNet(32, 32, c=c)

        #l1: 2048
        self.level1 = PointConvD(2048, feat_nei, 64 + 3, 64)
        self.shape1 = TransformerBlock(64, 64)
        self.cost1 = PointConvFlow(flow_nei, 64 + 32 + 64 + 32 + 3, [64, 64])
        self.flow1 = SceneFlowEstimatorPointConv(64 + 64+32*4, 64)
        self.rf_block1 = FlowRefineNet(64, 64, c=2*c)
        self.level1_0 = Conv1d(64, 64)
        self.level1_1 = Conv1d(64, 128)

        #l2: 512
        self.level2 = PointConvD(512, feat_nei, 128 + 3, 128)
        self.shape2 = TransformerBlock(128, 128)
        self.cost2 = PointConvFlow(flow_nei, 128 + 64 + 128 + 64 + 3, [128, 128])
        self.flow2 = SceneFlowEstimatorPointConv(128+64*4, 128, flow_ch=0)
        self.rf_block2 = FlowRefineNet(128, 128, c=4*c)
        self.level2_0 = Conv1d(128, 128)
        self.level2_1 = Conv1d(128, 256)

        #l3: 256
        self.level3 = PointConvD(256, feat_nei, 256 + 3, 256)
        self.shape3 = TransformerBlock(256, 256)
        
        #self.level4 = PointConvD(64, feat_nei, 512 + 3, 512)
        #self.mlp = nn.Sequential(nn.Linear(512,512), nn.ReLU(), nn.Linear(512,512))
        #self.shape4 = TransformerBlock(512, 512)
        #deconv
        self.conv4_3 = Conv1d(512, 256)
        self.conv3_2 = Conv1d(256, 128)
        self.conv2_1 = Conv1d(128, 64)
        self.pred = nn.Sequential(nn.Linear(64, 32),nn.ReLU(),nn.Linear(32,3))
        self.deconv3_2 = Conv1d(256, 64)
        self.deconv2_1 = Conv1d(128, 32)
        self.deconv1_0 = Conv1d(64, 32)

        #warping
        self.warping = PointWarping()

        #upsample
        self.upsample = UpsampleFlow()

        self.fusion = PointsFusion(4, [64,64,128])

    def forward(self, xyz1, xyz2, t, gt=None, train=False):
       
        #xyz1, xyz2: B, 3, N
        #color1, color2: B, 3, N

        #l0
        #pc1_l0 = xyz1.permute(0, 2, 1)
        #pc2_l0 = xyz2.permute(0, 2, 1)
        B = xyz1.shape[0]
        pc1_l0 = xyz1
        pc2_l0 = xyz2
        #color1 = color1.permute(0, 2, 1) # B 3 N
        #color2 = color2.permute(0, 2, 1) # B 3 N
        feat1_l0 = self.level0(pc1_l0)
        feat1_l0 = self.level0_1(feat1_l0)  # B,32,8192 (point1)
        feat1_l0_1 = self.level0_2(feat1_l0) # B,64,8192

        feat2_l0 = self.level0(pc2_l0)
        feat2_l0 = self.level0_1(feat2_l0)  # B,32,8192 (point2)
        feat2_l0_1 = self.level0_2(feat2_l0)
        #el01 = self.pointtlayer1(feat1_l0.permute(0,2,1), xyz1.permute(0,2,1))
        #el02 = self.pointtlayer1(feat2_l0.permute(0,2,1), xyz2.permute(0,2,1))
        #print('el01:', el01.shape, 'el02:', el02.shape)
        af, mf = self.feature_bone(xyz1, xyz2)
        #print('af:', len(af), 'mf:', len(mf))
        #print('af[0]:', af[0].shape, 'af[1]:', af[1].shape, 'af[2]', af[2].shape, 'af[3]', af[3].shape)
        #print('mf[0]:', type(mf[0]), len(mf[0])) 
        #print('mf[1]:', mf[1].shape, 'mf[2]', mf[2].shape)
        '''
        af[0]: torch.Size([12, 32, 8192]) af[1]: torch.Size([12, 32, 2048]) af[2] torch.Size([12, 64, 512]) af[3] torch.Size([12, 128, 256])
        mf[0]: <class 'list'> 0
        mf[1]: torch.Size([12, 32, 2048]) mf[2] torch.Size([12, 64, 512]) mf[3] torch.Size([12, 512, 256])
        '''

        #l1
        #feat1_l0_1 = torch.cat([feat1_l0_1, af[0][:]])
        pc1_l1, feat1_l1 = self.level1(pc1_l0, feat1_l0_1)  # [B,64,N/4]
        #el11 = self.pointtlayer2(feat1_l1.permute(0,2,1), pc1_l1.permute(0,2,1))
        feat1_l1_2 = self.level1_0(feat1_l1)
        feat1_l1_2 = self.level1_1(feat1_l1_2)

        pc2_l1, feat2_l1 = self.level1(pc2_l0, feat2_l0_1)  # [B,64,N/4]
        #el12 = self.pointtlayer2(feat2_l.permute(0,2,1), pc2_l1.permute(0,2,1))
        #print('el11:', el11.shape, 'el12:', el12.shape)
        feat2_l1_2 = self.level1_0(feat2_l1)
        feat2_l1_2 = self.level1_1(feat2_l1_2)

        #l2
        pc1_l2, feat1_l2 = self.level2(pc1_l1, feat1_l1_2)  # [B,128,N/16]
        #el21 = self.pointtlayer3(feat1_l2.permute(0,2,1), pc1_l2.permute(0,2,1))  # !!!!!!
        feat1_l2_3 = self.level2_0(feat1_l2)
        feat1_l2_3 = self.level2_1(feat1_l2_3)

        pc2_l2, feat2_l2 = self.level2(pc2_l1, feat2_l1_2)  # [B,128,N/16]
        #el22 = self.pointtlayer3(feat2_l2.permute(0,2,1), pc2_l2.permute(0,2,1))  # !!!!!!
        #print('el21:', el21.shape, 'el22:', el22.shape)
        feat2_l2_3 = self.level2_0(feat2_l2)
        feat2_l2_3 = self.level2_1(feat2_l2_3)

        #l3
        pc1_l3, feat1_l3 = self.level3(pc1_l2, feat1_l2_3)  # [B,256,N/32]
        #el31 = self.pointtlayer4(feat1_l3.permute(0,2,1), pc1_l3.permute(0,2,1))  # !!!!!!
        #feat1_l3_4 = self.level3_0(feat1_l3)
        #feat1_l3_4 = self.level3_1(feat1_l3_4)

        pc2_l3, feat2_l3 = self.level3(pc2_l2, feat2_l2_3)  # [B,256,N/32]
        #el32 = self.pointtlayer4(feat2_l3.permute(0,2,1), pc2_l3.permute(0,2,1))  # !!!!!!
        

        feat1_l3_2 = self.upsample(pc1_l2, pc1_l3, feat1_l3)
        feat1_l3_2 = self.deconv3_2(feat1_l3_2)

        feat2_l3_2 = self.upsample(pc2_l2, pc2_l3, feat2_l3)
        feat2_l3_2 = self.deconv3_2(feat2_l3_2)

        c_feat1_l2 = torch.cat([feat1_l2, feat1_l3_2], dim = 1)
        c_feat2_l2 = torch.cat([feat2_l2, feat2_l3_2], dim = 1)

        feat1_l2_1 = self.upsample(pc1_l1, pc1_l2, feat1_l2)
        feat1_l2_1 = self.deconv2_1(feat1_l2_1)

        feat2_l2_1 = self.upsample(pc2_l1, pc2_l2, feat2_l2)
        feat2_l2_1 = self.deconv2_1(feat2_l2_1)

        c_feat1_l1 = torch.cat([feat1_l1, feat1_l2_1], dim = 1)
        c_feat2_l1 = torch.cat([feat2_l1, feat2_l2_1], dim = 1)

        feat1_l1_0 = self.upsample(pc1_l0, pc1_l1, feat1_l1)
        feat1_l1_0 = self.deconv1_0(feat1_l1_0)

        feat2_l1_0 = self.upsample(pc2_l0, pc2_l1, feat2_l1)
        feat2_l1_0 = self.deconv1_0(feat2_l1_0)

        #l2
        c_feat1_l0 = torch.cat([feat1_l0, feat1_l1_0], dim = 1)
        c_feat2_l0 = torch.cat([feat2_l0, feat2_l1_0], dim = 1)
        cost2 = self.cost2(pc1_l2, pc2_l2, c_feat1_l2, c_feat2_l2)
        feat2, flow2 = self.flow2(pc1_l2, torch.cat([feat1_l2, t*mf[2][:B], (1-t)*mf[2][B:], af[2][:B], af[2][B:]],1), cost2)
        flow2 = self.rf_block2(feat1_l2, feat2_l2, cost2, flow2)

        #l1
        up_flow1 = self.upsample(pc1_l1, pc1_l2, self.scale * flow2)
        pc2_l1_warp = self.warping(pc1_l1, pc2_l1, up_flow1)
        cost1 = self.cost1(pc1_l1, pc2_l1_warp, c_feat1_l1, c_feat2_l1)

        feat2_up = self.upsample(pc1_l1, pc1_l2, feat2)
        new_feat1_l1 = torch.cat([feat1_l1, feat2_up], dim = 1)
        feat1, flow1 = self.flow1(pc1_l1, torch.cat([new_feat1_l1,t*mf[1][:B], (1-t)*mf[1][B:], af[1][:B], af[1][B:]],1), cost1, up_flow1)
        flow1 = self.rf_block1(feat1_l1, feat2_l1, cost1, flow1)
        #print('cost1:', cost1.shape, 'new_feat1_l1:', new_feat1_l1.shape)

        #l0
        up_flow0 = self.upsample(pc1_l0, pc1_l1, self.scale * flow1)
        pc2_l0_warp = self.warping(pc1_l0, pc2_l0, up_flow0)
        cost0 = self.cost0(pc1_l0, pc2_l0_warp, c_feat1_l0, c_feat2_l0)

        feat1_up = self.upsample(pc1_l0, pc1_l1, feat1)
        new_feat1_l0 = torch.cat([feat1_l0, feat1_up], dim = 1)
        feat0, flow0 = self.flow0(pc1_l0, new_feat1_l0, cost0, up_flow0)
        #print('cost0:', cost0.shape, 'new_feat1_l0:', new_feat1_l0.shape)
        #print('flow0:', flow0.shape, 'flow1:', flow1.shape, 'flow2:', flow2.shape, 'flow3:', flow3.shape)
        #print('pc1_l0:', pc1_l0.shape, 'pc1_l1:', pc1_l1.shape, 'pc1_l2:', pc1_l2.shape, 'pc1_l3:', pc1_l3.shape)
        #print('pc2_l0:', pc2_l0.shape, 'pc2_l1:', pc2_l1.shape, 'pc2_l2:', pc2_l2.shape, 'pc2_l3:', pc2_l3.shape)
        #flows = [flow0, flow1, flow2]  # [B,3,N], [B,3,N/4], [B,3,N/16]
        
        ### flow refinenet
        flow0 = self.rf_block0(feat1_l0, feat2_l0, cost0, flow0)
        warped_pc1t_l2 = pc1_l2 + flow2*t
        #warped_pc2t_l2 = pc2_l2 + flow2*(1-t)
        warped_pc1t_l1 = pc1_l1 + flow1*t
        #warped_pc2t_l1 = pc2_l1 + flow1*(1-t)
        warped_pc1t = pc1_l0 + flow0*t
        warped_pc2t = pc2_l0 + flow0*(1-t)
        k=32
        #fused_initial = self.fusion0(warped_pc1t, warped_pc2t, k, t)
        #print('flow0:', flow0.shape, 'flow1:', flow1.shape, 'flow2:', flow2.shape)
        #print('feat1_l0:', feat1_l0.shape, 'feat1_l1:', feat1_l1.shape, 'feat1_l2:', feat1_l2.shape)
        _, _, N = flow0.shape
        fps_idx = furthest_point_sample(flow2.permute(0,2,1).contiguous(), int(N/32))
        flow3 = (index_points_gather(flow2.permute(0,2,1).contiguous(), fps_idx)).permute(0,2,1)
        #print('pc1_l3:', pc1_l3.shape, 'flow3:', flow3.shape, 'fps_idx:', fps_idx.shape, 'N:', N)
        warped_pc1t_l3 = pc1_l3 + flow3*t

        warped_feat1t_l0 = feat1_l0_1 + (F.interpolate(flow0.permute(0,2,1), size=feat1_l0_1.size(1), mode="area")).permute(0,2,1)*t
        warped_feat1t_l1 = feat1_l1 + (F.interpolate(flow1.permute(0,2,1), size=feat1_l1.size(1), mode="area")).permute(0,2,1)*t
        warped_feat1t_l2 = feat1_l2 + (F.interpolate(flow2.permute(0,2,1), size=feat1_l2.size(1), mode="area")).permute(0,2,1)*t
        warped_feat1t_l3 = feat1_l3 + (F.interpolate(flow3.permute(0,2,1), size=feat1_l3.size(1), mode="area")).permute(0,2,1)*t
        #warped_feat2t_l0 = feat2_l0 + (F.interpolate(flow0.permute(0,2,1), size=feat1_l0.size(1), mode="area")).permute(0,2,1)*(1-t)
        #warped_feat2t_l1 = feat2_l1 + (F.interpolate(flow1.permute(0,2,1), size=feat1_l1.size(1), mode="area")).permute(0,2,1)*(1-t)
        #warped_feat2t_l2 = feat2_l2 + (F.interpolate(flow2.permute(0,2,1), size=feat1_l2.size(1), mode="area")).permute(0,2,1)*(1-t)
        #warped_feat2t_l3 = feat2_l3 + (F.interpolate(flow3.permute(0,2,1), size=feat1_l3.size(1), mode="area")).permute(0,2,1)*(1-t)
        #out1 = (warped_feat1t_l0, warped_feat1t_l1, warped_feat1t_l2)
        #out2 = (warped_feat2t_l0, warped_feat2t_l1, warped_feat2t_l2)

        # shape encode
        fused_down1, fused_feat1 = self.level1(warped_pc1t, warped_feat1t_l0)
        #print('fused_initial:', fused_initial.shape, 'fused_down1:', fused_down1.shape, 'fused_feat1:', fused_feat1.shape)
        #print('warped_feat1t_l0:', warped_feat1t_l0.shape)
        fea_shape1 = self.shape1(fused_feat1.permute(0,2,1), fused_down1.permute(0,2,1)) #[B,64,2048]
        fused_down2, fused_feat2 = self.level2(fused_down1, torch.cat([warped_feat1t_l1, fea_shape1], dim=1)) 
        fea_shape2 = self.shape2(fused_feat2.permute(0,2,1), fused_down2.permute(0,2,1)) #[B,128,512]
        fused_down3, fused_feat3 = self.level3(fused_down2, torch.cat([warped_feat1t_l2, fea_shape2], dim=1)) 
        fea_shape3 = self.shape3(fused_feat3.permute(0,2,1), fused_down3.permute(0,2,1)) #[B,256,256]
        #fused_down4, fused_feat4 = self.level4(fused_down3, torch.cat([warped_feat1t_l3, fea_shape3], dim=1))
        #fea_shape4 =self.shape4((self.mlp(fused_feat4.permute(0,2,1))), fused_down4.permute(0,2,1)) #[B,512,64]
        #print('fused_down3:', fused_down3.shape, 'fused_down4:', fused_down4.shape,'warped_pc1t_l3:', warped_pc1t_l3.shape)
        
        #up_feat3 = self.upsample(fused_down3, fused_down4, fea_shape4)
        #print('up_feat3:', up_feat3.shape)
        #up_feat3 = self.shape3((self.conv4_3(up_feat3)).permute(0,2,1), fused_down3.permute(0,2,1))
        up_pc2 = self.upsample(fused_down2, fused_down3, warped_pc1t_l3)
        up_feat2 = self.upsample(fused_down2, fused_down3, fea_shape3)
        #print('up_pc2:',up_pc2.shape, 'up_feat2:',up_feat2.shape)
        up_feat2 = self.shape2((self.conv3_2(up_feat2)).permute(0,2,1), up_pc2.permute(0,2,1))
        up_pc1 = self.upsample(fused_down1, fused_down2, warped_pc1t_l2)
        up_feat1 = self.upsample(fused_down1, fused_down2, fea_shape2)
        up_feat1 = self.shape1((self.conv2_1(up_feat1)).permute(0,2,1), up_pc1.permute(0,2,1))
        up_feat0 = self.upsample(warped_pc1t, fused_down1, fea_shape1)
        refine_out = (self.pred(up_feat0.permute(0,2,1))).permute(0,2,1)
        #print('refine_out:', refine_out.shape)
        #mask = F.softmax(refine_out, dim=-1)
        #refine_out = warped_pc1t*mask + warped_pc2t*(1-mask)
        #print('mask:', mask.shape)
        #pred = torch.cat([warped_pc1t[:,:,mask<t], warped_pc2t[:,:,mask>t]], dim=1)
        #print('pred:', pred.shape)
        if t > 0.5:
            warped_pc = warped_pc2t
        else:
            warped_pc = warped_pc1t

        fused_points = self.fusion(warped_pc, refine_out, k, t)
        if train:
            gt1 = downsampling(gt, int(N/4))
            gt2 = downsampling(gt, int(N/16))
            gt3 = downsampling(gt, int(N/32))
            gt_list = [gt1, gt2, gt3]
            warped_list = [warped_pc1t, warped_pc1t_l1, warped_pc1t_l2, warped_pc1t_l3]
        
            return flow0, fused_points, warped_list, gt_list, warped_pc2t
        else:
            return fused_points

def downsampling(pc, num):
    # return [B,3,N], [B,3,N/4], [B,3,N/16], [B,3,N/32]
    _, _, N = pc.shape
    idx = furthest_point_sample(pc.permute(0,2,1).contiguous(), num)
    gt = (index_points_gather(pc.permute(0,2,1).contiguous(), idx)).permute(0,2,1)
    return gt



LEAKY_RATE = 0.1
use_bn = False # True

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=use_bn):
        super(Conv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        self.composed_module = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm1d(out_channels) if bn else nn.Identity(),
            relu
        )

    def forward(self, x):
        x = self.composed_module(x)
        return x

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    #device = src.device
    #print('src:', src.shape)
    #print('dst:', dst.shape)
    dst = dst.to(src.device)
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def gather_point(points, inds):
    '''

    :param points: shape=(B, N, C)
    :param inds: shape=(B, M) or shape=(B, M, K)
    :return: sampling points: shape=(B, M, C) or shape=(B, M, K, C)
    '''
    device = points.device
    B, N, C = points.shape
    inds_shape = list(inds.shape)
    inds_shape[1:] = [1] * len(inds_shape[1:])
    repeat_shape = list(inds.shape)
    repeat_shape[0] = 1
    batchlists = torch.arange(0, B, dtype=torch.long).to(device).reshape(inds_shape).repeat(repeat_shape)
    inds = inds.type(torch.long)
    return points[batchlists, inds, :]

def index_points_gather(points, fps_idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """

    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = gather_operation(points_flipped, fps_idx)
    return new_points.permute(0, 2, 1).contiguous()

def index_points_group(points, knn_idx):
    """
    Input:
        points: input points data, [B, N, C]
        knn_idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = grouping_operation(points_flipped, knn_idx.int()).permute(0, 2, 3, 1)

    return new_points

def group(nsample, xyz, points):
    """
    Input:
        nsample: scalar
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = N
    new_xyz = xyz
    idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points_group(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points_group(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm

def group_query(nsample, s_xyz, xyz, s_points):
    """
    Input:
        nsample: scalar
        s_xyz: input points position data, [B, N, C]
        s_points: input points data, [B, N, D]
        xyz: input points position data, [B, S, C]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = s_xyz.shape
    S = xyz.shape[1]
    new_xyz = xyz
    idx = knn_point(nsample, s_xyz, new_xyz)
    grouped_xyz = index_points_group(s_xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if s_points is not None:
        grouped_points = index_points_group(s_points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm

class WeightNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit = [8, 8], bn = use_bn):
        super(WeightNet, self).__init__()

        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        
    def forward(self, localized_xyz):
        #xyz : BxCxKxN

        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                weights =  F.relu(bn(conv(weights)))
            else:
                weights = F.relu(conv(weights))

        return weights

class PointConv(nn.Module):
    def __init__(self, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(PointConv, self).__init__()
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def forward(self, xyz, points):
        """
        PointConv without strides size, i.e., the input and output have the same number of points.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        new_points, grouped_xyz_norm = group(self.nsample, xyz, points)

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other = weights.permute(0, 3, 2, 1)).view(B, N, -1)
        #print('new_points:', new_points.shape, new_points.device)
        new_points = self.linear(new_points)
        #print('---new_points---:', new_points.shape, new_points.device)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_points

class PointConvD(nn.Module):
    def __init__(self, npoint, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(PointConvD, self).__init__()
        self.npoint = npoint
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz, points):
        """
        PointConv with downsampling.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        #import ipdb; ipdb.set_trace()
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1).contiguous()
        points = points.permute(0, 2, 1)

        fps_idx = furthest_point_sample(xyz, self.npoint)
        new_xyz = index_points_gather(xyz, fps_idx)

        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, points)

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other = weights.permute(0, 3, 2, 1)).view(B, self.npoint, -1)
        new_points = self.linear(new_points)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_xyz.permute(0, 2, 1), new_points

class PointConvFlow(nn.Module):
    def __init__(self, nsample, in_channel, mlp, bn = use_bn, use_leaky = True):
        super(PointConvFlow, self).__init__()
        self.nsample = nsample
        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        if bn:
            self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet1 = WeightNet(3, last_channel)
        self.weightnet2 = WeightNet(3, last_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def forward(self, xyz1, xyz2, points1, points2):
        """
        Cost Volume layer for Flow Estimation
        Input:
            xyz1: input points position data, [B, C, N1]
            xyz2: input points position data, [B, C, N2]
            points1: input points data, [B, D, N1]
            points2: input points data, [B, D, N2]
        Return:
            new_points: upsample points feature data, [B, D', N1]
        """
        # import ipdb; ipdb.set_trace()
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        # point-to-patch Volume
        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        grouped_points2 = index_points_group(points2, knn_idx) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1)
        new_points = torch.cat([grouped_points1, grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3
        new_points = new_points.permute(0, 3, 2, 1) # [B, D1+D2+3, nsample, N1]
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                new_points =  self.relu(bn(conv(new_points)))
            else:
                new_points =  self.relu(conv(new_points))

        # weighted sum
        weights = self.weightnet1(direction_xyz.permute(0, 3, 2, 1)) # B C nsample N1 

        point_to_patch_cost = torch.sum(weights * new_points, dim = 2) # B C N

        # Patch to Patch Cost
        knn_idx = knn_point(self.nsample, xyz1, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz1, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        # weights for group cost
        weights = self.weightnet2(direction_xyz.permute(0, 3, 2, 1)) # B C nsample N1 
        grouped_point_to_patch_cost = index_points_group(point_to_patch_cost.permute(0, 2, 1), knn_idx) # B, N1, nsample, C
        patch_to_patch_cost = torch.sum(weights * grouped_point_to_patch_cost.permute(0, 3, 2, 1), dim = 2) # B C N

        return patch_to_patch_cost

class PointWarping(nn.Module):

    def forward(self, xyz1, xyz2, flow1 = None):
        if flow1 is None:
            return xyz2

        # move xyz1 to xyz2'
        xyz1_to_2 = xyz1 + flow1 

        # interpolate flow
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        xyz1_to_2 = xyz1_to_2.permute(0, 2, 1) # B 3 N1
        xyz2 = xyz2.permute(0, 2, 1) # B 3 N2
        flow1 = flow1.permute(0, 2, 1)

        knn_idx = knn_point(3, xyz1_to_2, xyz2)
        grouped_xyz_norm = index_points_group(xyz1_to_2, knn_idx) - xyz2.view(B, N2, 1, C) # B N2 3 C
        dist = torch.norm(grouped_xyz_norm, dim = 3).clamp(min = 1e-10)
        norm = torch.sum(1.0 / dist, dim = 2, keepdim = True)
        weight = (1.0 / dist) / norm 

        grouped_flow1 = index_points_group(flow1, knn_idx)
        flow2 = torch.sum(weight.view(B, N2, 3, 1) * grouped_flow1, dim = 2)
        warped_xyz2 = (xyz2 - flow2).permute(0, 2, 1) # B 3 N2

        return warped_xyz2

class UpsampleFlow(nn.Module):
    def forward(self, xyz, sparse_xyz, sparse_flow):
        #import ipdb; ipdb.set_trace()
        B, C, N = xyz.shape
        _, _, S = sparse_xyz.shape

        xyz = xyz.permute(0, 2, 1) # B N 3
        sparse_xyz = sparse_xyz.permute(0, 2, 1) # B S 3
        sparse_flow = sparse_flow.permute(0, 2, 1) # B S 3
        knn_idx = knn_point(3, sparse_xyz, xyz)
        grouped_xyz_norm = index_points_group(sparse_xyz, knn_idx) - xyz.view(B, N, 1, C)
        dist = torch.norm(grouped_xyz_norm, dim = 3).clamp(min = 1e-10)
        norm = torch.sum(1.0 / dist, dim = 2, keepdim = True)
        weight = (1.0 / dist) / norm 

        grouped_flow = index_points_group(sparse_flow, knn_idx)
        dense_flow = torch.sum(weight.view(B, N, 3, 1) * grouped_flow, dim = 2).permute(0, 2, 1)
        return dense_flow 

class SceneFlowEstimatorPointConv(nn.Module):

    def __init__(self, feat_ch, cost_ch, flow_ch = 3, channels = [128, 128], mlp = [128, 64], neighbors = 9, clamp = [-200, 200], use_leaky = True):
        super(SceneFlowEstimatorPointConv, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch + cost_ch + flow_ch

        for _, ch_out in enumerate(channels):
            pointconv = PointConv(neighbors, last_channel + 3, ch_out, bn = True, use_leaky = True)
            #pointconv = PointTransformerLayer(dim = last_channel, out_c = ch_out, pos_mlp_hidden_dim = 64, attn_mlp_hidden_mult = 4)
            self.pointconv_list.append(pointconv)
            last_channel = ch_out 
        
        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 3, 1)

    def forward(self, xyz, feats, cost_volume, flow = None):
        '''
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        '''
        if flow is None:
            new_points = torch.cat([feats, cost_volume], dim = 1)
        else:
            new_points = torch.cat([feats, cost_volume, flow], dim = 1)

        for _, pointconv in enumerate(self.pointconv_list):
            #print('new_points000:', new_points.shape)
            #print('xyz:', xyz.shape)
            new_points = pointconv(xyz, new_points)
            #new_points = pointconv(new_points.permute(0,2,1), xyz.permute(0,2,1))
            #print('new_points111:', new_points.shape)

        for conv in self.mlp_convs:
            new_points = conv(new_points)

        flow = self.fc(new_points)
        return new_points, flow.clamp(self.clamp[0], self.clamp[1])


class PointsFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointsFusion, self).__init__()

        layers = []
        out_channels = [in_channels, *out_channels]
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i - 1], out_channels[i], 1, bias=True), nn.BatchNorm2d(out_channels[i], eps=0.001), nn.ReLU()]
        
        self.conv = nn.Sequential(*layers)
        #self.sample = Sample(N)

    def knn_group(self, points1, points2, k):
        '''
        For each point in points1, query kNN points in points2
        Input:
            points1: [B,3,N]
            points2: [B,3,N]
        Output:
            new_features: [B,4,N]
            nn: [B,3,N]
        '''
        points1 = points1.permute(0,2,1).contiguous()
        points2 = points2.permute(0,2,1).contiguous()
        _, nn_idx, nn = knn_points(points1, points2, K=k, return_nn=True)
        points_resi = nn - points1.unsqueeze(2).repeat(1,1,k,1)
        grouped_dist = torch.norm(points_resi, dim=-1, keepdim=True)
        new_features = torch.cat([points_resi, grouped_dist], dim=-1)

        return new_features.permute(0,3,1,2).contiguous(),\
            nn.permute(0,3,1,2).contiguous()
    
    def forward(self, points1, points2, k, t):
        '''
        Input:
            points1: [B,3,N]
            points2: [B,3,N]
            features1: [B,C,N] (only for inference of additional features)
            features2: [B,C,N] (only for inference of additional features)
            k: int, number of kNN cluster
            t: [B], time step in (0,1)
            pc: [B,4,N]
        Output:
            fused_points: [B,3+C,N]
        '''
        N = points1.shape[-1]   # 点数
        B = points1.shape[0]    # batch size

        new_features_list = []
        new_grouped_points_list = []
        new_grouped_features_list = []
        

        for i in range(B):
            new_points1 = points1[i:i+1,:,:]
            new_points2 = points2[i:i+1,:,:]

            new_features1, grouped_points1 = self.knn_group(new_points1, new_points1, k)
            new_features2, grouped_points2 = self.knn_group(new_points1, new_points2, k)

            new_features = torch.cat((new_features1, new_features2), dim=-1)
            new_grouped_points = torch.cat((grouped_points1, grouped_points2), dim=-1)

            new_features_list.append(new_features)
            new_grouped_points_list.append(new_grouped_points)

        new_features = torch.cat(new_features_list, dim=0)
        new_grouped_points = torch.cat(new_grouped_points_list, dim=0)

        new_features = self.conv(new_features)  # [B,128,N,32+16]
        new_features = torch.max(new_features, dim=1, keepdim=False)[0]
        weights = F.softmax(new_features, dim=-1)

        weights = weights.unsqueeze(1).repeat(1,3,1,1)
        fused_points = torch.sum(torch.mul(weights, new_grouped_points), dim=-1, keepdim=False)

        return fused_points

class PointsFusion2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointsFusion2, self).__init__()
        
        self.conv = SelfAttention(in_channels, out_channels)
    
    def knn_group(self, points1, points2, k):
        '''
        For each point in points1, query kNN points in points2
        Input:
            points1: [B,3,N]
            points2: [B,3,N]
        Output:
            new_features: [B,4,N,k]
            nn: [B,3,N,k]
        '''
        points1 = points1.permute(0,2,1).contiguous()
        points2 = points2.permute(0,2,1).contiguous()
        _, nn_idx, nn = knn_points(points1, points2, K=k, return_nn=True)
        points_resi = nn - points1.unsqueeze(2).repeat(1,1,k,1)
        grouped_dist = torch.norm(points_resi, dim=-1, keepdim=True)
        new_features = torch.cat([points_resi, grouped_dist], dim=-1)

        return new_features.permute(0,3,1,2).contiguous(), nn.permute(0,3,1,2).contiguous()
    
    def forward(self, points1, points2, pc, k, t):
        '''
        Input:
            points1: [B,3,N]
            points2: [B,3,N]
            features1: [B,C,N] (only for inference of additional features)
            features2: [B,C,N] (only for inference of additional features)
            k: int, number of kNN cluster
        Output:
            fused_points: [B,3,N]
        '''
        N = points1.shape[-1]   # 点数
        B = points1.shape[0]    # batch size

        new_features_list = []
        new_grouped_points_list = []

        for i in range(B):
            new_points1 = points1[i:i+1,:,:]
            new_points2 = points2[i:i+1,:,:]
            new_points3 = pc[i:i+1,:,:]

            N2 = int(N*t)   # 设置从warped帧中采样点的个数
            N1 = N - N2

            k2 = int(k*t)
            k1 = k - k2

            randidx1 = torch.randperm(N)[:N1]   # 把N个数打散，取前N1个数
            randidx2 = torch.randperm(N)[:N2]   # 把N个数打散，取前N2个数
            # 从warped_pc1中取N1个点，从warped_pc2中取N2个点，cat起来
            new_points = torch.cat((new_points1[:,:,randidx1], new_points2[:,:,randidx2]), dim=-1)   # [B,3,N]

            new_features1, grouped_points1 = self.knn_group(new_points, new_points1, k1)
            new_features2, grouped_points2 = self.knn_group(new_points, new_points2, k2)
            new_features3, grouped_points3 = self.knn_group(new_points, new_points3, k)

            new_features = torch.cat((new_features1, new_features2, new_features3), dim=-1)
            new_grouped_points = torch.cat((grouped_points1, grouped_points2, grouped_points3), dim=-1)

            new_features_list.append(new_features)
            new_grouped_points_list.append(new_grouped_points)

        new_features = torch.cat(new_features_list, dim=0)                  # [B,4,N,k*2]
        new_grouped_points = torch.cat(new_grouped_points_list, dim=0)      # [B,3,N,k*2]

        new_features = self.conv(new_features.permute(0,2,3,1)) 
        #print('new_features:', new_features.shape) 
        new_features = torch.max(new_features.permute(0,3,1,2), dim=1, keepdim=False)[0]  # [B,N,K]
        weights = F.softmax(new_features, dim=-1)
        
        weights = weights.unsqueeze(1).repeat(1,3,1,1)
        fused_points = torch.sum(torch.mul(weights, new_grouped_points), dim=-1, keepdim=False)  # [B,3,N]

        return fused_points


class DWConv(nn.Module):
    def __init__(self, dim):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv1d(dim, dim, 1, 1, 0, bias=True, groups=dim)

    def forward(self, x, N):
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, N)
        x = self.dwconv(x)
        x = x.reshape(B, C, -1).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        #elif isinstance(m, nn.Conv1d):
        #    fan_out = m.kernel_size * m.out_channels
        #    fan_out //= m.groups
        #    m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        #    if m.bias is not None:
        #        m.bias.data.zero_()

    def forward(self, x, N):
        x = self.fc1(x)
        #print('x:', x.shape)
        x = self.dwconv(x, N)
        #print('x:', x.shape)
        x = self.act(x)
        #print('x:', x.shape)
        x = self.drop(x)
        #print('x:', x.shape)
        x = self.fc2(x)
        #print('x:', x.shape)
        x = self.drop(x)
        #print('x:', x.shape)
        return x


class InterFrameAttention(nn.Module):
    def __init__(self, dim, motion_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.motion_dim = motion_dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.cor_embed = nn.Linear(3, motion_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.motion_proj = nn.Linear(motion_dim, motion_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        

    def forward(self, x1, x2, cor, N, mask=None):
        B, N, C = x1.shape
        B, N, C_c = cor.shape
        q = self.q(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        cor_embed_ = self.cor_embed(cor)
        cor_embed = cor_embed_.reshape(B, N, self.num_heads, self.motion_dim // self.num_heads).permute(0, 2, 1, 3)
        k, v = kv[0], kv[1]    
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            nW = mask.shape[0] # mask: nW, N, N
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        c_reverse = (attn @ cor_embed).transpose(1, 2).reshape(B, N, -1)
        motion = self.motion_proj(c_reverse-cor_embed_)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, motion

class MotionFormerBlock(nn.Module):
    def __init__(self, dim, motion_dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.PReLU, norm_layer=nn.BatchNorm1d):
        super().__init__()

        #self.shift_size = shift_size
        #if not isinstance(self.shift_size, (tuple, list)):
        #    self.shift_size = to_2tuple(shift_size)
        #self.bidirectional = bidirectional
        self.norm1 = norm_layer(dim)
        self.attn = InterFrameAttention(
            dim,
            motion_dim, 
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        #elif isinstance(m, nn.Conv1d):
            

    def forward(self, x, cor, N, B):
        #x = x.view(2*B, -1, N)    

        #print('x:', x.shape)  
        nwB = x.shape[0]
        x_norm = (self.norm1(x)).permute(0,2,1)  # B,C,N

        x_reverse = torch.cat([x_norm[nwB//2:], x_norm[:nwB//2]])
        #print('x_norm:', x_norm.shape, 'x_reverse:', x_reverse.shape, 'cor:', cor.shape)
        x_appearence, x_motion = self.attn(x_norm, x_reverse, cor, N)  # B,C,N
        x_norm = x_norm + self.drop_path(x_appearence)

        x_back = x_norm #.view(2*B, N, -1)
        #print('x_back:', x_back.shape, 'x:', x.permute(0,2,1).shape)
        x_back = self.norm2(x_back.permute(0,2,1))
        #print('x_back:', x_back.shape)
        x_back = self.drop_path(self.mlp(x_back.permute(0,2,1), N))
        #print('x_back:', x_back.shape)
        x = x + x_back.permute(0,2,1) # self.drop_path(self.mlp(self.norm2(x_back.permute(0,2,1)), N))
        return x, x_motion


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, depths=2,act_layer=nn.PReLU):
        super().__init__()
        layers = []
        for i in range(depths):
            if i == 0:
                layers.append(nn.Conv1d(in_dim, out_dim, 1,1,0,bias=True))
            else:
                layers.append(nn.Conv1d(out_dim, out_dim, 1,1,0,bias=True))
            layers.extend([
                act_layer(out_dim),
            ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        return x


class MotionFormer(nn.Module):
    def __init__(self, in_chans=3, npoints=[8192,2048,512], embed_dims=[32, 32, 64], motion_dims=[0,16,32],
                 mlp_ratios=[4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.BatchNorm1d,
                 depths=[2, 2, 2, 4], feat_nei = 16, **kwarg):    # window_sizes=[11, 11],
        super().__init__()
        self.depths = depths
        self.num_stages = len(embed_dims)  # 5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        #self.conv_stages = self.num_stages - len(num_heads) # 3

        for i in range(self.num_stages): # 0,1,2,3
            if i == 0:                   # 0
                block = ConvBlock(in_chans,embed_dims[i],depths[i])   # inputchannel=3-->outchannel=32, 2 layers
            else: # 1,2,3
                patch_embed = PointConvD(npoints[i], feat_nei, embed_dims[i-1]+3, embed_dims[i])  # downsampling x2
                
                block = nn.ModuleList([MotionFormerBlock(
                    dim=embed_dims[i], motion_dim=motion_dims[i],
                    mlp_ratio=4., qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer)
                    for j in range(depths[i])])

                norm = norm_layer(embed_dims[i])
                setattr(self, f"norm{i + 1}", norm)
                setattr(self, f"patch_embed{i + 1}", patch_embed)
            cur += depths[i]

            setattr(self, f"block{i + 1}", block)

        self.cor = {}

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        

    def get_cor(self, shape, device):
        k = (str(shape), str(device))
        if k not in self.cor:
            tenHorizontal = torch.linspace(-1.0, 1.0, shape[1], device=device).view(
                1, 1, shape[1]).expand(shape[0], -1, -1).permute(0, 2, 1)
            tenVertical = torch.linspace(-1.0, 1.0, shape[1], device=device).view(
                1, 1, shape[1]).expand(shape[0], -1, -1).permute(0, 2, 1)
            tenZaxis = torch.linspace(-1.0, 1.0, shape[1], device=device).view(
                1, 1, shape[1]).expand(shape[0], -1, -1).permute(0, 2, 1)
            self.cor[k] = torch.cat([tenHorizontal, tenVertical, tenZaxis], -1).to(device)
        return self.cor[k]

    def forward(self, x1, x2):
        B = x1.shape[0] 
        x = torch.cat([x1, x2], 0)  # 2B,3,N
        motion_features = []
        appearence_features = []
        xs = []
        for i in range(self.num_stages):
            motion_features.append([])  
            patch_embed = getattr(self, f"patch_embed{i + 1}",None)
            block = getattr(self, f"block{i + 1}",None)
            norm = getattr(self, f"norm{i + 1}",None)
            if i == 0:
                fea = block(x)
                xs.append(fea)
            else:
                #print('i:', i, 'x:', x.shape)
                xyz, fea = patch_embed(x, xs[i-1])
                xs.append(fea)
                N = xyz.shape[2]
                cor = self.get_cor((xyz.shape[0], N), xyz.device)
                for blk in block:
                    x_, x_motion = blk(fea, cor, N, B)
                    #print('x_motion:', x_motion.shape)
                    motion_features[i].append(x_motion.permute(0, 2, 1).contiguous())  # B,C,N
                    x_ = x_.permute(0,2,1)
                #print('x:', x.shape)
                x_ = norm(x_.permute(0,2,1))  # B,C,N
                fea = x_.reshape(2*B, -1, N).contiguous()
                motion_features[i] = torch.cat(motion_features[i], 1)
                #print('motion:', motion_features[i].shape)
            #print('fea:', fea.shape)
            appearence_features.append(fea)
            #print('motion_features:', type(motion_features))
        return appearence_features, motion_features
