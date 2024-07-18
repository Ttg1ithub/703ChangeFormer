import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
import os

class AdaptiveInstanceNormalization(nn.Module):
    """
    风格迁移（有随机比例扰动
    初始化：
    show=False是否展示迁移图像，
    static_ratio=0.5迁移比例
    """
    counter = 0
    cts=[]

    def __init__(self, show=False, static_ratio=0.5):
        super(AdaptiveInstanceNormalization, self).__init__()
        self.static_ratio = static_ratio     
        self.ratio=rand(self.static_ratio,0,1)
        self.show=show
    def forward(self, x_cont, x_style=None):
        """
        输入：
        x_cont 源图像
        x_style=None 风格图像
        输出：
        迁移后的图像
        """
        if x_style is not None:
            x_style = torch.narrow(x_style, 0, 0, x_cont.size()[0])
            assert (x_cont.size()[:2] == x_style.size()[:2])
            size = x_cont.size()
            style_mean, style_std = calc_mean_std(x_style)
            content_mean, content_std = calc_mean_std(x_cont)

            normalized_x_cont = (x_cont - content_mean.expand(size))/content_std.expand(size)
            denormalized_x_cont = normalized_x_cont * style_std.expand(size) + style_mean.expand(size)

            k1=torch.full(size,self.ratio).to('cuda')
            k2=torch.full(size,1-self.ratio).to('cuda')
            self.ratio=rand(self.static_ratio,0,1)          
            denormalized_x_cont = k1*denormalized_x_cont + k2*x_cont
            del k1,k2
            #将张量沿高度方向拼接
            ct = torch.cat((x_cont[0], denormalized_x_cont[0]), dim=1).to('cuda:1')  # 拼接在高度（垂直方向），dim=1 表示沿着通道的方向拼接
            AdaptiveInstanceNormalization.cts.append(ct)
            #保存拼接后的张量为图像文件
            if self.show and len(AdaptiveInstanceNormalization.cts)==64:
                concatenated_tensor=torch.cat(tuple(AdaptiveInstanceNormalization.cts),dim=2)
                AdaptiveInstanceNormalization.counter+=1
                AdaptiveInstanceNormalization.cts=[]
                # if torch.equal(denormalized_x_cont, x_cont):
                #     print("张量相等")
                # else:
                #     print("张量不相等!!!")

                save_image(concatenated_tensor, os.path.join('/mnt/backup/gcw-yhj/ChangeFormer/Adain-effect',
                                                             str(AdaptiveInstanceNormalization.counter)+'.png'))

            return denormalized_x_cont

        else:
            return x_cont

def rand(mean, low, high, std=0.2,):
    np.random.seed(42)
    while True:
        random_number = np.random.normal(mean, std)
        if low <= random_number <= high:
            break
    return random_number


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

