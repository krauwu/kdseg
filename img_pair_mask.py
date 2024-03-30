import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['IPMLoss']

# cert_list = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
#              11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 21.0,
#              22.0, 23.0, 24.0]

class IPMLoss(nn.Module):
    def __init__(self,
                 student_channels=512,
                 teacher_channels=2048,
                 pool_size=4,
                 patch_size=(4, 4),
                 mask_ratio=0.75,
                 enhance_projector=False,
                 dataset='citys'
                 ):

        super(IPMLoss, self).__init__()

        self.zeta_fd = mask_ratio
        self.pool_size = pool_size
        self.patch_size = patch_size
        self.dataset = dataset
        self.mask_sit = 1  # 0: spatial  1: channel  2: channel+spatial
        # self.mask_loc = 0  # 0: random  1: neat  2: on_target

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)

        self.generator = None
        if enhance_projector:
            self.projetor = EnhancedProjector(teacher_channels, teacher_channels)
        else:
            self.projetor = nn.Sequential(nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1))

    def forward(self, feat_S, feat_T, target):
        # feat_S = self.align(feat_S)
        dis_loss = self.get_dis_loss(feat_S, feat_T, target)
        return dis_loss

    def get_dis_loss(self, preds_S, preds_T, target):
        loss_mse = nn.MSELoss(reduction='sum')
        BS, C, H, W = preds_S.shape  # 1/8 input size

        target = target.unsqueeze(1).float()
        # print(target.unique(), 'target1')
        target = F.interpolate(target, scale_factor=0.125, mode='bilinear', align_corners=True)
        # print(target.unique(), 'target')

        device = preds_S.device
        # print(target.shape, preds_T.shape, preds_S.shape,'22') # torch.Size([22, 512, 512]) torch.Size([22, 256, 64, 64]) torch.Size([22, 256, 64, 64]) 22

        if self.mask_sit == 0:
            mat_spt = torch.rand((BS, 1, H, W)).to(device)
            mat_spt = torch.where(mat_spt > 1 - self.zeta_fd, 0, 1).to(device)
            # mat_chl = torch.rand((BS, C, 1, 1)).to(device)
            # mat_chl = torch.where(mat_chl > 1 - self.zeta_fd, 0, 1).to(device)
            # mat_sc = torch.rand((BS, C, H, W)).to(device)
            # mat_sc = torch.where(mat_sc > 1 - self.zeta_fd, 0, 1).to(device)
            # print(torch.sum(mat_spt == 1).item(), torch.sum(mat_spt == 0).item(), 'mat')

        elif self.mask_sit == 2:
            mat_spt = torch.zeros((BS, 1, H, W)).to(device)
            # for value in cert_list:
            #     mask_tar = target == value
            #     mat_spt[mask_tar] = 0
            mask_tar = target != -1
            mat_spt[mask_tar] = torch.rand(mat_spt.shape).detach().clone()[mask_tar].to(device).detach()
            mat_spt = torch.where(mat_spt > 1 - self.zeta_fd, 0, 1).to(device)
            mat_spt[~mask_tar] = torch.ones(mat_spt.shape).long().detach().clone()[~mask_tar].to(device)

        elif self.mask_sit == 1:
            mat_spt = torch.zeros((BS, 1, H, W)).to(device)
            mat_spt[:, :, ::2, ::2] = 1

        masked_fea = torch.mul(preds_S, mat_spt)

        masked_fea = self.align(masked_fea)
        new_fea = self.projetor(masked_fea)

        dis_loss = loss_mse(new_fea, preds_T) / BS

        return dis_loss


class EnhancedProjector(nn.Module):
    def __init__(self,
                 in_channels=2048,
                 out_channels=2048,
                 ):
        super(EnhancedProjector, self).__init__()
        self.block_1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True))
        self.block_2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True))
        self.adpator_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.adpator_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x_1 = self.block_1(x)
        x_2 = self.block_2(x_1)

        x_1 = self.adpator_1(x_1)
        x_2 = self.adpator_2(x_2)

        out = (x_1 + x_2) / 2.

        return out
