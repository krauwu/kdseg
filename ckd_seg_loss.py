"""Custom losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

__all__ = ['CriterionCKD']

class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)

        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class CriterionCKD(nn.Module):
    """
    refound from minibatch_loss, muliti-gpu version can be modified from other loss
    需要设置两个模块:类别间的CKD 和 CRD在feat中的应用\
    [bs, cls_num, 256, pixel_h, pixel_w] -> [bs, cls_num, 256, pixel_num]
    截取一部分像素点进行CKD
    """

    def __init__(self, temperature, s_channels, t_channels, ignore_label, feat_size,
                 queue_size=24):
        super(CriterionCKD, self).__init__()
        self.temperature = temperature
        # self.queue_size = queue_size
        # self.ignore_label = ignore_label
        # self.img_size = feat_size
        # self.counting = 0

        self.project_head = nn.Sequential(
            nn.Conv2d(s_channels, t_channels, 1, bias=False),
            nn.BatchNorm2d(t_channels),
            nn.ReLU(True),
            nn.Conv2d(t_channels, t_channels, 1, bias=False)
        ).cuda()
        # [128, 256, 64, 128]
        # self.ptr_t = 0
        # self.register_buffer("teacher_pixel_ckd_queue", torch.randn(queue_size, t_channels, self.img_size[0], self.img_size[1]))
        # self.teacher_pixel_ckd_queue = nn.functional.normalize(self.teacher_pixel_ckd_queue, p=2, dim=1)

    @torch.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        tensors_gather = [torch.ones_like(tensor)
                          for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output

    def batch_compare(self, feature):
        ''' feature: torch.Size([6, 256, 64, 128])'''

        diff_tensor = []

        for i in range(feature.shape[0]):
            for j in range(i+1, feature.shape[0]):
                diff_tensor.append(feature[i] - feature[j])

        diff_tensor = torch.stack(diff_tensor, dim=0)

        return diff_tensor

    # def dequeue_and_enqueue_wo_label(self, feat_t):
    #     # version only for one gpu, 目前无类别信息
    #     # 废案
    #     pixel_queue_t = self.teacher_pixel_ckd_queue
    #     # [128, 256, 64, 128]
    #     # f_t = [bs, 256, 64, 128]
    #     ptr_t = self.ptr_t
    #
    #     if ptr_t + feat_t.shape[0] < self.queue_size:
    #         pixel_queue_t[ptr_t:ptr_t + feat_t.shape[0], ...] = feat_t
    #
    #     else:
    #         pixel_queue_t[ptr_t:, ...] = feat_t[:self.queue_size - ptr_t, ...]
    #         pixel_queue_t[:feat_t.shape[0] - (self.queue_size - ptr_t), ...] = feat_t[self.queue_size - ptr_t:, ...]
    #
    #     self.ptr_t = (ptr_t + feat_t.shape[0]) % self.queue_size
    #
    # def memorybank_compare(self, feat):
    #     ''' feat: torch.Size([6, 256, 64, 128]) '''
    #
    #     diff_tensor = []
    #
    #     for i in range(feat.shape[0]):
    #         for j in range(self.teacher_pixel_ckd_queue.shape[0]):
    #             diff_tensor.append(feat[i] - self.teacher_pixel_ckd_queue[j])
    #
    #     diff_tensor = torch.stack(diff_tensor, dim=0)
    #
    #     # diff_tensor: torch.Size([bs*queue_size, 256, 64, 128])
    #
    #     return diff_tensor

    def forward(self, feat_S, feat_T):

        B, C, H, W = feat_S.size()

        feat_S = F.normalize(feat_S, p=2, dim=1)
        feat_T = F.normalize(feat_T, p=2, dim=1)

        feat_S = self.project_head(feat_S)
        # size both s, t: [bs, 256(channel_model), 64(input_down), 128(same)]

        diff_teacher = self.batch_compare(feat_T)
        diff_student = self.batch_compare(feat_S)

        diff_teacher = F.log_softmax(diff_teacher/self.temperature, dim=1)
        diff_student = F.softmax(diff_student/self.temperature, dim=1)

        ckd_loss = F.kl_div(diff_teacher, diff_student, reduction='batchmean') * self.temperature**2

        return ckd_loss
