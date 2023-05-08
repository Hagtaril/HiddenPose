import torch
import torch.cuda.comm
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat

import numpy as np


class NMTNORMCritierion(nn.Module):
    def __init__(self, label_smoothing):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.LogSoftmax = nn.LogSoftmax(dim = 1)

        if label_smoothing > 0:
            self.criterion_ = nn.KLDivLoss(reduction='none')
        else:
            self.criterion_ = nn.NLLLoss(reduction='none', ignore_index = 100000)

        self.confidence = 1.0 - label_smoothing


    def _smooth_label(self, num_tokens):
        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.label_smoothing / (num_tokens - 1))
        return one_hot

    def criterion(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        num_tokens = scores.size(-1)

        gtruth = labels.view(-1)
        if self.confidence < 1:
            tdata = gtruth.detach().long().to('cuda:0')
            one_hot = self._smooth_label(num_tokens).to('cuda:0')
            tmp_ = one_hot.repeat(gtruth.size(0), 1)
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
            gtruth = tmp_.detach()

        scores = scores.to('cuda:0')
        gtruth = gtruth.to('cuda:0')
        loss = torch.mean(self.criterion_(scores, gtruth), dim = 1)
        return loss

    def forward(self, output_x, output_y, output_z, target, target_weight):
        batch_size = output_x.size(0)
        num_joints = output_x.size(1)
        loss = 0

        for idx in range(num_joints):
            coord_x_pred = output_x[:, idx].squeeze()
            coord_y_pred = output_y[:, idx].squeeze()
            coord_z_pred = output_z[:, idx].squeeze()
            coord_gt = target[:, idx].squeeze()
            weight = target_weight[:, idx].squeeze()

            # coord_gt = rearrange(coord_gt, 'd -> 1 d')
            # coord_x_pred, coord_y_pred, coord_z_pred = map(lambda t:rearrange(t, 'd -> 1 d'), (coord_x_pred, coord_y_pred, coord_z_pred))
            loss += self.criterion(coord_x_pred, coord_gt[:, 0]).mul(weight).mean()
            loss += self.criterion(coord_y_pred, coord_gt[:, 1]).mul(weight).mean()
            loss += self.criterion(coord_z_pred, coord_gt[:, 2]).mul(weight).mean()
        return loss / num_joints

    
class L2JointLocationLoss(nn.Module):
    def __init__(self, output_3d, size_average=True, reduce=True):
        super(L2JointLocationLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.output_3d = output_3d

    def forward(self, preds, *args):
        gt_joints = args[0]
        gt_joints_vis = args[1]

        num_joints = int(gt_joints_vis.shape[1] / 3)
        hm_width = preds.shape[-1]
        hm_height = preds.shape[-2]
        # hm_depth = preds.shape[-3] // num_joints if self.output_3d else 1
        hm_depth = preds.shape[-3]

        pred_jts = softmax_integral_tensor(preds, num_joints, self.output_3d, hm_width, hm_height, hm_depth,)

        _assert_no_grad(gt_joints)
        _assert_no_grad(gt_joints_vis)
        return weighted_mse_loss(pred_jts, gt_joints, gt_joints_vis, self.size_average)


def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"


def generate_3d_integral_preds_tensor(heatmaps, num_joints, x_dim, y_dim, z_dim):
    assert isinstance(heatmaps, torch.Tensor)
    '''
    Parameter


    heatmaps: probility of location
    -----------------------------------
    Return 
    accu_x: mean location of x label

    '''

    heatmaps = heatmaps.reshape((heatmaps.shape[0], num_joints, z_dim, y_dim, x_dim))

    accu_x = heatmaps.sum(dim=2)  # (1, 24, 5, 5, 5)
    accu_x = accu_x.sum(dim=2)
    accu_y = heatmaps.sum(dim=2)
    accu_y = accu_y.sum(dim=3)
    accu_z = heatmaps.sum(dim=3)
    accu_z = accu_z.sum(dim=3)

    accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(x_dim).type(torch.cuda.FloatTensor), devices=[accu_x.device.index])[0]
    accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(y_dim).type(torch.cuda.FloatTensor), devices=[accu_y.device.index])[0]
    accu_z = accu_z * torch.cuda.comm.broadcast(torch.arange(z_dim).type(torch.cuda.FloatTensor), devices=[accu_z.device.index])[0]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)

    return accu_x, accu_y, accu_z


def softmax_integral_tensor(preds, num_joints, output_3d, hm_width, hm_height, hm_depth):
    # global soft max
    preds = preds.reshape((preds.shape[0], num_joints, -1))
    preds = F.softmax(preds, 2)
    # fn = nn.LogSoftmax(dim=2)
    # preds = fn(preds)
    # print(preds)

    # integrate heatmap into joint location
    if output_3d:
        x, y, z = generate_3d_integral_preds_tensor(preds, num_joints, hm_width, hm_height, hm_depth)
    else:
        assert 0, 'Not Implemented!'  # TODO: Not Implemented
    # x = x / float(hm_width) - 0.5
    # y = y / float(hm_height) - 0.5
    # z = z / float(hm_depth) - 0.5
    # writer.add_scalar('x0 location', x[0, 0, 0], global_step=global_iter_num)
    # writer.add_scalar('x10 location', x[0, 10, 0], global_step=global_iter_num)
    # writer.add_scalar('y0 location', y[0, 0, 0], global_step=global_iter_num)
    # writer.add_scalar('y10 location', y[0, 10, 0], global_step=global_iter_num)
    # writer.add_scalar('z0 location', z[0, 0, 0], global_step=global_iter_num)
    # writer.add_scalar('z10 location', z[0, 10, 0], global_step=global_iter_num)
    preds = torch.cat((x, y, z), dim=2)
    preds = preds.reshape((preds.shape[0], num_joints * 3))
    return preds


def weighted_mse_loss(input, target, weights, size_average):
    out = (input - target) ** 2
    out = out * weights
    if size_average:
        return out.sum() / len(input)
    else:
        return out.sum()



class JointsMSELoss(nn.Module):
    def __init__(self, cfg, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.target_type = cfg.DATASET.TARGET_TYPE
        self.heatmap_size = cfg.DATASET.HEATMAP_SIZE
        self.sigma = cfg.DATASET.SIGMA
        self.use_different_joints_weight = cfg.DATASET.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [self.num_joints, 3]
        :param joints_vis: [self.num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        # self.target_type = 'gaussian'
        # self.heatmap_size = [128, 128]
        # self.sigma = 2 * self.heatmap_size[0] / 64

        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        # target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma

            for joint_id in range(self.num_joints):
                target_weight[joint_id] = \
                    self.adjust_target_weight(joints[joint_id], target_weight[joint_id], tmp_size)

                if target_weight[joint_id] == 0:
                    continue

                mu_x = joints[joint_id][0].numpy()
                mu_y = joints[joint_id][1].numpy()

                # 生成过程与hrnet的heatmap size不一样
                x = np.arange(0, self.heatmap_size[0], 1, np.float32)
                y = np.arange(0, self.heatmap_size[1], 1, np.float32)
                y = y[:, np.newaxis]

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id] = np.exp(- ((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * self.sigma ** 2))

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return torch.Tensor(target), torch.Tensor(target_weight)

    def adjust_target_weight(self, joint, target_weight, tmp_size):
        # feat_stride = self.image_size / self.heatmap_size
        mu_x = joint[0]
        mu_y = joint[1]
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight = 0

        return target_weight

    def forward(self, output, target, target_weight, writer, global_iter_num):
        # [3, 2, 5, 5]   [b, n, 3]
        # output = output.reshape(3,1,5,5)
        batch_size = output.size(0)
        num_joints = output.size(1)

        heatmaps_pred = output.reshape((batch_size, num_joints, -1))


        heatmaps_gt = torch.zeros_like(output)
        for idx_batch in range(batch_size):

            heatmaps_gt[idx_batch], target_weight[idx_batch] = self.generate_target(target[idx_batch], target_weight[idx_batch])
        heatmaps_gt = heatmaps_gt.reshape((batch_size, num_joints, -1))

        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[:,idx].squeeze()  # (batch_size, dim)
            # heatmap_gt, target_weight = self.generate_target(target[:, idx], target_weight[:, idx])
            heatmap_gt = heatmaps_gt[:,idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class JointTarget(object):
    """docstring for JointTarget"""

    def __init__(self, cfg):
        super().__init__()
        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.target_type = cfg.DATASET.TARGET_TYPE
        self.heatmap_size = cfg.DATASET.HEATMAP_SIZE
        self.sigma = cfg.DATASET.SIGMA * cfg.DATASET.HEATMAP_SIZE[0] // cfg.DATASET.IMAGE_SIZE[0]
        self.use_different_joints_weight = cfg.DATASET.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [self.num_joints, 3]
        :param joints_vis: [self.num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        # self.target_type = 'gaussian'
        # self.heatmap_size = [128, 128]
        # self.sigma = 2 * self.heatmap_size[0] / 64

        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                target_weight[joint_id] = \
                    self.adjust_target_weight(joints[joint_id], target_weight[joint_id], tmp_size)

                if target_weight[joint_id] == 0:
                    continue

                mu_x = joints[joint_id][0].numpy()
                mu_y = joints[joint_id][1].numpy()

                # 生成过程与hrnet的heatmap size不一样
                x = np.arange(0, self.heatmap_size[0], 1, np.float32)
                y = np.arange(0, self.heatmap_size[1], 1, np.float32)
                y = y[:, np.newaxis]

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id] = np.exp(- ((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * self.sigma ** 2))

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return torch.Tensor(target), torch.Tensor(target_weight)

    def adjust_target_weight(self, joint, target_weight, tmp_size):
        # feat_stride = self.image_size / self.heatmap_size
        mu_x = joint[0]
        mu_y = joint[1]
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight = 0

        return target_weight


class DiceLoss(nn.Module):
    """Calculate dice loss."""
    def __init__(self, eps: float = 1e-9):
        super(DiceLoss, self).__init__()
        self.eps = eps
        
    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        
        num = targets.size(0)
        probability = torch.sigmoid(logits)
        probability = probability.view(num, -1)
        targets = targets.view(num, -1)
        assert(probability.shape == targets.shape)
        
        intersection = 2.0 * (probability * targets).sum()
        union = probability.sum() + targets.sum()
        dice_score = (intersection + self.eps) / union
        #print("intersection", intersection, union, dice_score)
        return 1.0 - dice_score
        
        
class BCEDiceLoss(nn.Module):
    """Compute objective loss: BCE loss + DICE loss."""
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        
    def forward(self, 
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        assert(logits.shape == targets.shape)
        dice_loss = self.dice(logits, targets)
        bce_loss = self.bce(logits, targets)
        
        return bce_loss + dice_loss



if __name__ == '__main__':
    test_NMTNORMCritierion = True # SimDR
    test_L2JointLocationLoss = False # 3D Heatmap

    test_JointMESLoss = False # 2D Heatmap

    if test_NMTNORMCritierion:
        criterion = nn.KLDivLoss(reduction='mean')
        output = torch.Tensor([[0.1132, 0.5477, 0.3390]])
        target = torch.Tensor([[0.8541, 0.0511, 0.0947]])
        # output = F.log_softmax(output, dim = 1)
        # target = F.log_softmax(target, dim = 1)
        loss = criterion(output, target)
        print(loss.item())
        import numpy as np

        output = output[0].detach().numpy()
        output_1 = output[0]
        target_1 = target[0][0].numpy()

        loss_1 = target_1 * (np.log(target_1) - output_1)
        print(loss_1)

    if test_L2JointLocationLoss:

        batch_size = 1
        num_joints = 24
        z_dim = 5
        y_dim = 5
        x_dim = 5

        input = torch.zeros(batch_size, num_joints, z_dim, y_dim, x_dim).cuda() - 1000
        # input[0, 0, 2, 2, 2] = 1.
        for idx in range(num_joints):
            input[0,idx,0,0,0] = 1 if idx != 0 else -1000
        input[0, 0, 1, 1, 1] = 1.

        print(input.shape)
        # test_a = generate_3d_integral_preds_tensor(input, num_joints, z_dim, y_dim, x_dim)
        # test_b = softmax_integral_tensor(input, 24, True, 5, 5, 5)

        Loss = L2JointLocationLoss(output_3d=True)
        gt_joints = torch.zeros(batch_size, num_joints, 3).cuda()
        # gt_joints[0, 1, 0] = 1
        gt_joints[0,0] = torch.Tensor([1,1,1])
        gt_joints_vis = torch.ones_like(gt_joints).cuda()
        # gt_joints = rearrange(gt_joints, 'b n d -> b (n d)')
        gt_joints = gt_joints.reshape((batch_size, num_joints * 3))
        gt_joints_vis = rearrange(gt_joints_vis, 'b n d -> b (n d)')
        print(gt_joints[0, 0])
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter('./log')

        loss = Loss(input, gt_joints, gt_joints_vis, writer, 1)
        print(f'loss is {loss.item()}')
        # intput : [batch_size, num_joints, z_length, y_length, x_length]
        # gt_joints : [batch_size, num_joints * 3]  belong to [-0.5, 0.5]
        # gt_joints_vis : [batch_size, num_joints_vis]
        # writer : tensorboard SummaryWriter
        # global_iter_num : int

        print('finished')

    if test_JointMESLoss:
        def get_cfg():
            from yacs.config import CfgNode as CN
            _C = CN()
            _C.DATASET = CN()

            _C.DATASET.NUM_JOINTS = 4
            _C.DATASET.TARGET_TYPE = 'gaussian'
            _C.DATASET.HEATMAP_SIZE = [5, 5]
            # _C.DATASET.IMAGE_SIZE = [5, 5]
            _C.DATASET.SIGMA = 1
            _C.DATASET.USE_DIFFERENT_JOINTS_WEIGHT = True

            return _C


        cfg = get_cfg()
        criterion = JointsMSELoss(cfg, use_target_weight=False)

        output = torch.ones(6, 4, 5, 5)
        gt = torch.Tensor([[ [2,2] ], [[1,1]], [[1,1]]])
        gt = repeat(gt, 'a b d -> (2 a) (4 b) d')
        gt_vis = torch.ones_like(gt)
        loss = criterion(output, gt, gt_vis, 1, 2)
        print(f'loss is {loss.item()}')

        # heatmap = JointTarget(cfg)
        #
        # joints = torch.Tensor([[10, 10, 0], [100, 100, 0]])
        # joints_vis = torch.ones(2, 3)
        # target, target_weight = heatmap.generate_target(joints, joints_vis)
        #
        # from einops import repeat
        #
        # target = repeat(target, 'n h w -> b n h w', b=3)
        # target_weight = repeat(target_weight, 'n 1 -> b n 1', b=3)
        #
        # # image = target[0]
        # #
        # # print(target_weight)
        # #
        # # import matplotlib.pyplot as plt
        # # plt.imshow(image, 'hot')
        # # plt.show()
        # criterion = JointsMSELoss(use_target_weight=True)
        # target = torch.ones(3, 2, 5, 5)
        # # output = torch.rand((3, 2, cfg.DATASET.HEATMAP_SIZE[0], cfg.DATASET.HEATMAP_SIZE[1]))
        # output = torch.ones(3, 2, 5, 5)
        # output[:, :, 1, 1] = 0
        # print(output.shape)
        # loss = criterion(output, target, target_weight, 1, 2)
        # print(f'loss is {loss}')