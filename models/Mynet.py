import torch
import torch.nn as nn
import torch.nn.functional as F
from models.T3SAWBaseNetworks import *
import torchvision
from models.help_funcs import Transformer as SelfAttn

class Mynet(nn.Module):
    def __init__(self, backbone:str='resnet18', output_nc = 2, *args, **kwargs) -> None:
        """
        backbone:骨干名字
        """
        super().__init__(*args, **kwargs)
        self.backbone = self._getBackbone(backbone)
        self.embed_dims = [64, 128, 256, 512]
        self.embedding_dim = 64
        self._mixing_mask = nn.ModuleList(
            [
                MixingMaskAttentionBlock(6, 3, [3, 5, 10], [5, 10, 1]),
                MixingMaskAttentionBlock(128, 32, [32, 16, 8], [16, 8, 1]),
                MixingMaskAttentionBlock(256, 64, [64, 16, 8], [16, 8, 1]),
                MixingMaskAttentionBlock(512, 128, [128, 32, 16], [32, 16, 1]),
                MixingBlock(1024, 256),
            ]
        )
        self._up = nn.ModuleList(
            [
                UpMask(2, 256, 128),
                UpMask(2, 128, 64),
                UpMask(2, 64, 64),
                UpMask(4, 64, 32),
            ]
        )
        # #Final predction head
        # self.convd2x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        # # self.convd2x2    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        # self.dense_2x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        # self.convd1x    = UpsampleConvLayer(self.embedding_dim, 32, kernel_size=4, stride=2)
        # # self.convd1x2    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        # self.dense_1x   = nn.Sequential( ResidualBlock(32))
        # # self._classify = PixelwiseLinear([32, 16, 8], [16, 8, 2], nn.Sigmoid())
        self._classify = nn.Sequential(
            PixelwiseLinear([32, 16, 8, 4], [16, 8, 4, 2], nn.Softmax(dim=-3))
        )
        self.decoder = DecoderT3(input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=False, 
                    in_channels = self.embed_dims, embedding_dim= self.embedding_dim, output_nc=output_nc, 
                    decoder_softmax = False, feature_strides=[2, 4, 8, 16])
    def _getBackbone(self, bkbn_name:str, pretrained:bool=True, output_layer_bkbn:str=None, freeze_backbone:bool=False):
        # The whole model:
        entire_model = getattr(torchvision.models, bkbn_name)(
            pretrained=pretrained
        )

        # # Slicing it:
        # derived_model = nn.ModuleList([])
        # for name, layer in entire_model.named_children():
        #     derived_model.append(layer)
        #     if name == output_layer_bkbn:
        #         break

        # Freezing the backbone weights:
        if freeze_backbone:
            for param in entire_model.parameters():
                param.requires_grad = False
        return entire_model
    
    def _res_forward(self,x:torch.Tensor)->list:
        y = [x]
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        y.append(x)
        x = self.backbone.layer2(x)
        y.append(x)
        x = self.backbone.layer3(x)
        y.append(x)
        x = self.backbone.layer4(x)
        y.append(x)
        assert type(y[0])==torch.Tensor
        return y

    def _encode(self, y1:List[torch.Tensor], y2:List[torch.Tensor]) -> List[torch.Tensor]:
        features = []
        for num, item in enumerate(y1):
            features.append(self._mixing_mask[num](item, y2[num]))
        return features

    def _decode(self, features) -> torch.Tensor:
        upping = features[-1]
        for i, j in enumerate(range(-2, -2-len(self._up), -1)):
            upping = self._up[i](upping, features[j])
        return upping

    def forward(self,x1,x2,xw=None):
        y1, y2 = [], []
        if type(self.backbone)==torchvision.models.resnet.ResNet:
            y1, y2 = self._res_forward(x1), self._res_forward(x2) 
        features = self._encode(y1, y2)
        latent = self._decode(features)
        '''#Upsampling x2 (x1/2 scale)
        x = self.convd2x(latent)
        #Residual block
        x = self.dense_2x(x)
        #Upsampling x2 (x1 scale)
        x = self.convd1x(x)
        # x_abs = self.convd1x2(x_abs)
        #Residual block
        x = self.dense_1x(x)'''
        x = self._classify(latent)
        return [x]