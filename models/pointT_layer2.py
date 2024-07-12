import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k=16):
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k
        
    # xyz: b x n x 3, features: b x n x f
    def forward(self, features, xyz):
        #print('xyz:', xyz.shape, 'features:', features.shape)
        #xyz = xyz.permute(0,2,1)
        #features = features.permute(0,2,1)
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)
        
        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f
        
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f
        
        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre
        return res.permute(0,2,1)  # , attn

class FlowRefineNet(nn.Module):
    def __init__(self, context_dim, corr_dim, c=24):
        super(FlowRefineNet, self).__init__()
        flow_dim = 3
        motion_dim = c
        hidden_dim = c

        self.occl_convs = nn.Sequential(nn.Conv1d(2 * context_dim, hidden_dim, 1, 1, 0),
                                        nn.LeakyReLU(0.1, inplace=True),
                                        nn.Conv1d(hidden_dim, hidden_dim, 1, 1, 0),
                                        nn.LeakyReLU(0.1, inplace=True),
                                        nn.Conv1d(hidden_dim, 1, 1, 1, 0),
                                        nn.Sigmoid())
        
        self.motion_convs = nn.Sequential(nn.Conv1d(corr_dim + flow_dim, motion_dim, 3, 1, 1),
                                          nn.LeakyReLU(0.1, inplace=True))
        
        self.flow_head = nn.Sequential(nn.Conv1d(corr_dim + motion_dim + flow_dim, hidden_dim, 3, 1, 1),
                                       nn.LeakyReLU(0.1, inplace=True),
                                       nn.Conv1d(hidden_dim, 3, 3, 1, 1))
    
    def forward_once(self, fea0, fea1, cost, flow):
        # get context feature
        occl = self.occl_convs(torch.cat([fea0, fea1], dim=1))
        fea = fea0 * occl + fea1 * (1 - occl)
        
        # merge correlation and flow features, get motion features
        motion = self.motion_convs(torch.cat([cost, flow], dim=1))
        
        # update flows
        inp = torch.cat([fea, motion, flow], dim=1)
        delta_flow = self.flow_head(inp)
        flow = flow + delta_flow
        return flow

    def forward(self, fea0, fea1, cost, flow):

        for i in range(4):
            flow = self.forward_once(fea0, fea1, cost, flow)
            
        
        return flow