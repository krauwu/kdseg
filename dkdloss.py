import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CriterionDKD']


class CriterionDKD(nn.Module):
    '''
    decouple knowledge distillation loss for segmentation
    '''

    def __init__(self, dkd_set, bs, temperature=1, ignore_index=-1, **kwargs):
        super(CriterionDKD, self).__init__()
        self.temperature = temperature
        self.bs = bs
        if dkd_set:
            self.target_lamda = 8  # CITYSCAPE: T:N = 8-1
            self.n_target_lamda = 1
        else:
            self.target_lamda = 8
            self.n_target_lamda = 1
        self.ignore = ignore_index

    def decouple_target(self, feat, mask_gt):
        """feat [bs*h*w, cls_num]"""
        ft_1 = feat * mask_gt
        ft_2 = feat * (~mask_gt)

        return torch.cat([ft_1, ft_2], dim=1)

    def forward(self, pred, soft, label):
        ''' pred [bs, cls_num, h, w] '''

        B, C, h, w = soft.size()

        label = label.unsqueeze(1).float().clone()
        label = torch.nn.functional.interpolate(label, [pred.shape[2], pred.shape[3]])

        # label[bs,1,h,w]
        # decouple: mask-gt/other

        scale_pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, C)
        scale_soft = soft.permute(0, 2, 3, 1).contiguous().view(-1, C)
        scale_label = label.permute(0, 2, 3, 1).contiguous().view(-1, 1)

        scale_label += 1
        mask_gt = torch.zeros(scale_pred.shape[0], scale_pred.shape[1] + 1
                              ).cuda().scatter_(1, scale_label.long(), torch.ones_like(scale_label)).bool()
        mask_gt = mask_gt[:, 1:]

        decouple_pred = self.decouple_target(scale_pred, mask_gt)
        decouple_soft = self.decouple_target(scale_soft, mask_gt)

        p_s = F.log_softmax(decouple_pred / self.temperature, dim=1)  # row-wise
        p_t = F.softmax(decouple_soft / self.temperature, dim=1)
        tar_loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.temperature ** 2)

        p_s_n = F.log_softmax(scale_pred / self.temperature - 1000.0 * mask_gt, dim=1)
        p_t_n = F.softmax(scale_soft / self.temperature - 1000.0 * mask_gt, dim=1)
        n_tar_loss = F.kl_div(p_s_n, p_t_n, reduction='batchmean') * (self.temperature ** 2)

        loss = self.target_lamda * tar_loss + self.n_target_lamda * n_tar_loss

        return loss
