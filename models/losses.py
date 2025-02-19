import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

import torch
import torch.nn.functional as F

def alignloss(feature1, feature2, label):
    '''feature1, feature2: 多通道特征张量, shape=(B, C, H, W)
    label: 分类标签张量, shape=(B, H, W)
    输出未变化区域特征相似度损失值
    '''
    # 找到标签为0的像素位置
    zero_label_indices = torch.where(label == 0)

    # 筛选出对应位置的特征向量
    feature1_selected = feature1[zero_label_indices[0], :, zero_label_indices[1], zero_label_indices[2]]
    feature2_selected = feature2[zero_label_indices[0], :, zero_label_indices[1], zero_label_indices[2]]

    # 计算相似度（这里用欧几里得距离作为示例）
    def euclidean_distance(x, y):
        return torch.norm(x - y, dim=1)  # 计算欧几里得距离

    similarity = euclidean_distance(feature1_selected, feature2_selected)

    # 求所有相关像素的平均相似度
    mean_similarity = torch.mean(similarity)
    return mean_similarity

def kl_divergence(pred1, pred2, weight=None):
    # Flatten predictions to (B * H * W, C)
    pred1 = pred1.view(pred1.size(0), pred1.size(1), -1)  # (B, C, H*W)
    pred2 = pred2.view(pred2.size(0), pred2.size(1), -1)  # (B, C, H*W)
    
    # Compute softmax along channel dimension
    pred1 = F.softmax(pred1, dim=1)
    pred2 = F.softmax(pred2, dim=1)
    pred1 = pred1.transpose(1,2)
    pred2 = pred2.transpose(1,2)
    # Add a small epsilon to avoid taking log of zero
    eps = 1e-8
    pred1 = pred1.clamp(min=eps)
    pred2 = pred2.clamp(min=eps)
    
    # KL divergence
    kl = F.kl_div(pred1.log(), pred2, reduction='none')
    kl[:,:,0] = kl[:,:,0] * weight[0]
    kl[:,:,1] = kl[:,:,1] * weight[1]
    kl = torch.mean(kl)
    return kl

def cross_entropy(input, target, weight=None, reduction='mean', ignore_index=255):
    """
    计算交叉熵损失函数
    :param input: torch.Tensor, 输入数据张量，形状为 N*C*H*W
    :param target: torch.Tensor, 目标标签张量，形状为 N*1*H*W 或者 N*H*W
    :param weight: torch.Tensor, 类别权重张量，形状为 C
    :param reduction: str, 损失的减少方法，可选值为 'mean', 'sum' 或者 'none'
    :param ignore_index: int, 忽略索引，指定的类别索引将不参与损失计算，默认为255
    :return: torch.Tensor, 交叉熵损失值，形状为 [0]
    """
    target = target.long()  # 将目标张量转换为长整型，确保与输入张量匹配
    if target.dim() == 4:  # 如果目标张量的维度是4
        target = torch.squeeze(target, dim=1)  # 压缩维度为1，假设是N*1*H*W形式的，变为N*H*W形式

    # 如果输入张量的最后一个维度与目标张量的最后一个维度不匹配
    if input.shape[-1] != target.shape[-1]:
        # 利用双线性插值对输入张量进行调整，使其大小与目标张量的大小相匹配
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear', align_corners=True)

    # 调用 PyTorch 中的交叉熵损失函数 F.cross_entropy 进行损失计算
    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)


#Focal Loss
def get_alpha(supervised_loader):
    # get number of classes
    num_labels = 0
    for batch in supervised_loader:
        label_batch = batch['L']
        label_batch.data[label_batch.data==255] = 0 # pixels of ignore class added to background
        l_unique = torch.unique(label_batch.data)
        list_unique = [element.item() for element in l_unique.flatten()]
        num_labels = max(max(list_unique),num_labels)
    num_classes = num_labels + 1
    # count class occurrences
    alpha = [0 for i in range(num_classes)]
    for batch in supervised_loader:
        label_batch = batch['L']
        label_batch.data[label_batch.data==255] = 0 # pixels of ignore class added to background
        l_unique = torch.unique(label_batch.data)
        list_unique = [element.item() for element in l_unique.flatten()]
        l_unique_count = torch.stack([(label_batch.data==x_u).sum() for x_u in l_unique]) # tensor([65920, 36480])
        list_count = [count.item() for count in l_unique_count.flatten()]
        for index in list_unique:
            alpha[index] += list_count[list_unique.index(index)]
    return alpha

# for FocalLoss
def softmax_helper(x):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/nd_softmax.py
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)

class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=1, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
	
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
            alpha = 1/alpha # inverse of class frequency
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')
        
        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
	
        # to resolve error in idx in scatter_
        idx[idx==225]=0
        
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth/(num_class-1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


#miou loss
from torch.autograd import Variable
def to_one_hot_var(tensor, nClasses, requires_grad=False):

    n, h, w = torch.squeeze(tensor, dim=1).size()
    one_hot = tensor.new(n, nClasses, h, w).fill_(0)
    one_hot = one_hot.scatter_(1, tensor.type(torch.int64).view(n, 1, h, w), 1)
    return Variable(one_hot, requires_grad=requires_grad)

class mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=2):
        super(mIoULoss, self).__init__()
        self.classes = n_classes
        self.weights = Variable(weight)

    def forward(self, inputs, target, is_target_variable=False):
        # inputs => N x Classes x H x W
        # target => N x H x W
        # target_oneHot => N x Classes x H x W

        N = inputs.size()[0]
        if is_target_variable:
            target_oneHot = to_one_hot_var(target.data, self.classes).float()
        else:
            target_oneHot = to_one_hot_var(target, self.classes).float()

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=1)

        # Numerator Product
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2)

        # Denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.classes, -1).sum(2)

        loss = (self.weights * inter) / (union + 1e-8)

        ## Return average loss over classes and batch
        return -torch.mean(loss)

#Minimax iou
class mmIoULoss(nn.Module):
    def __init__(self, n_classes=2):
        super(mmIoULoss, self).__init__()
        self.classes = n_classes

    def forward(self, inputs, target, is_target_variable=False):
        # inputs => N x Classes x H x W
        # target => N x H x W
        # target_oneHot => N x Classes x H x W

        N = inputs.size()[0]
        if is_target_variable:
            target_oneHot = to_one_hot_var(target.data, self.classes).float()
        else:
            target_oneHot = to_one_hot_var(target, self.classes).float()

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=1)

        # Numerator Product
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2)

        # Denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.classes, -1).sum(2)

        iou = inter/ (union + 1e-8)

        #minimum iou of two classes
        min_iou = torch.min(iou)

        #loss
        loss = -min_iou-torch.mean(iou)
        return loss
