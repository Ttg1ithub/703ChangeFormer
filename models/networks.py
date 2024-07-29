import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler
from functools import partial

import functools
from einops import rearrange

import models
from models.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d
import models.T3SAWBaseNetworks as T3
from models.ChangeFormer import ChangeFormerV1, ChangeFormerV2, ChangeFormerV3, ChangeFormerV4, ChangeFormerV5, ChangeFormerV6
from models.SiamUnet_diff import SiamUnet_diff
from models.SiamUnet_conc import SiamUnet_conc
from models.Unet import Unet
from models.DTCDSCN import CDNet34
from datasets.adain import AdaptiveInstanceNormalization as adain
from memory_profiler import profile
import objgraph as graph

###############################################################################
# Helper Functions
###############################################################################

def get_scheduler(optimizer, args):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs//3
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(args, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if args.net_G == 'base_resnet18':
        net = ResNet(input_nc=3, output_nc=2, output_sigmoid=False)

    elif args.net_G == 'T3SAW':
        net = T3SAW(input_nc=3, output_nc=2, embed_dim=args.embed_dim)

    elif args.net_G == 'base_transformer_pos_s4':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned')

    elif args.net_G == 'base_transformer_pos_s4_dd8':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=1, dec_depth=8)

    elif args.net_G == 'base_transformer_pos_s4_dd8_dedim8':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=1, dec_depth=8, decoder_dim_head=8)

    elif args.net_G == 'ChangeFormerV1':
        net = ChangeFormerV1() #ChangeFormer with Transformer Encoder and Convolutional Decoder
    
    elif args.net_G == 'ChangeFormerV2':
        net = ChangeFormerV2() #ChangeFormer with Transformer Encoder and Convolutional Decoder

    elif args.net_G == 'ChangeFormerV3':
        net = ChangeFormerV3() #ChangeFormer with Transformer Encoder and Convolutional Decoder (Fuse)

    elif args.net_G == 'ChangeFormerV4':
        net = ChangeFormerV4() #ChangeFormer with Transformer Encoder and Convolutional Decoder (Fuse)
    
    elif args.net_G == 'ChangeFormerV5':
        net = ChangeFormerV5(embed_dim=args.embed_dim) #ChangeFormer with Transformer Encoder and Convolutional Decoder (Fuse)

    elif args.net_G == 'ChangeFormerV6':
        net = ChangeFormerV6(embed_dim=args.embed_dim) #ChangeFormer with Transformer Encoder and Convolutional Decoder (Fuse)
    
    elif args.net_G == "SiamUnet_diff":
        #Implementation of ``Fully convolutional siamese networks for change detection''
        #Code copied from: https://github.com/rcdaudt/fully_convolutional_change_detection
        net = SiamUnet_diff(input_nbr=3, label_nbr=2)

    elif args.net_G == "SiamUnet_conc":
        #Implementation of ``Fully convolutional siamese networks for change detection''
        #Code copied from: https://github.com/rcdaudt/fully_convolutional_change_detection
        net = SiamUnet_conc(input_nbr=3, label_nbr=2)

    elif args.net_G == "Unet":
        #Usually abbreviated as FC-EF = Image Level Concatenation
        #Implementation of ``Fully convolutional siamese networks for change detection''
        #Code copied from: https://github.com/rcdaudt/fully_convolutional_change_detection
        net = Unet(input_nbr=3, label_nbr=2)
    
    elif args.net_G == "DTCDSCN":
        #The implementation of the paper"Building Change Detection for Remote Sensing Images Using a Dual Task Constrained Deep Siamese Convolutional Network Model "
        #Code copied from: https://github.com/fitzpchao/DTCDSCN
        net = CDNet34(in_channels=3)

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)
    return init_net(net, init_type, init_gain, gpu_ids)


###############################################################################
# main Functions
###############################################################################


class ResNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=True):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = models.resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False,False,False])
        elif backbone == 'resnet34':
            self.resnet = models.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet50':
            self.resnet = models.resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False,False,False])
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        # self.tmp_seq = torch.nn.Sequential(self.resnet.bn1,self.resnet.relu,
        #                         self.resnet.maxpool,self.resnet.layer1)

        self.adain = adain(show=False)
        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)

        self.resnet_stages_num = resnet_stages_num

        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

    def _single_first(self, x):
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x_4 = self.resnet.layer1(x) # 1/4, in=64, out=64
        return x_4

    def _single_next(self, x_8, x_ls:list=[]):
        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8) # 1/8, in=128, out=256
            x_ls.append(x_8)

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8) # 1/32, in=256, out=512
            x_ls.append(x_8)
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            x = self.upsamplex2(x_8)
        else:
            x = x_8
        # output layers
        x = self.conv_pred(x)
        return x
    
    def forward_single(self, x, xw=None, x_ls:list=[], xw_ls:list=[]):
        # resnet layers
        x = self.resnet.conv1(x)
        if xw is not None:
            xw = self.resnet.conv1(xw)
            x_sw = self.adain(x,xw)
        
        x_4 = self._single_first(x)
        x_ls.append(x_4)
        if xw is not None:
            x_sw = self._single_first(x_sw)
            xw = self._single_first(xw)
            x_sw = self.adain(x_sw, xw)
            xw_ls.append(x_sw)
        x_8 = self.resnet.layer2(x_4) # 1/8, in=64, out=128
        x_ls.append(x_8)
        if xw is not None:
            x_sw = self.resnet.layer2(x_sw)
            xw = self.resnet.layer2(xw)
            x_sw = self.adain(x_sw, xw)
            xw_ls.append(x_sw)
        x = self._single_next(x_8, x_ls)
        if xw is None:
            x_sw = torch.tensor([-1.0])
        else:
            x_sw = self._single_next(x_sw, xw_ls)
        return x, x_sw

class BASE_Transformer(ResNet):
    """
    Resnet of 8 downsampling + BIT + bitemporal feature Differencing + a small CNN
    """
    def __init__(self, input_nc, output_nc, with_pos, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 tokenizer=True, if_upsample_2x=True,
                 pool_mode='max', pool_size=2,
                 backbone='resnet18',
                 decoder_softmax=True, with_decoder_pos=None,
                 with_decoder=True):
        super(BASE_Transformer, self).__init__(input_nc, output_nc,backbone=backbone,
                                             resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x,
                                               )
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,
                                padding=0, bias=False)
        self.tokenizer = tokenizer
        if not self.tokenizer:
            #  if not use tokenzier，then downsample the feature map into a certain size
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size

        self.token_trans = token_trans
        self.with_decoder = with_decoder
        dim = 32
        mlp_dim = 2*dim

        self.with_pos = with_pos
        if with_pos == 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, 32))
        decoder_pos_size = 256//4
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder =nn.Parameter(torch.randn(1, 32,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax)

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)

        return tokens

    def _forward_reshape_tokens(self, x):
        # b,c,h,w = x.shape
        if self.pool_mode == 'max':
            x = F.adaptive_max_pool2d(x, [self.pooling_size, self.pooling_size])
        elif self.pool_mode == 'ave':
            x = F.adaptive_avg_pool2d(x, [self.pooling_size, self.pooling_size])
        else:
            x = x
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        return tokens

    def _forward_transformer(self, x):
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h,w,b,l,c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def _forward_next(self, x1, x2):
        #  forward tokenzier
        if self.tokenizer:
            token1 = self._forward_semantic_tokens(x1)
            token2 = self._forward_semantic_tokens(x2)
        else:
            token1 = self._forward_reshape_tokens(x1)
            token2 = self._forward_reshape_tokens(x2)
        # forward transformer encoder
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)
        # forward transformer decoder
        if self.with_decoder:
            x1 = self._forward_transformer_decoder(x1, token1)
            x2 = self._forward_transformer_decoder(x2, token2)
        else:
            x1 = self._forward_simple_decoder(x1, token1)
            x2 = self._forward_simple_decoder(x2, token2)
        # feature differencing
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x1 = self.upsamplex4(x1)
        x2 = self.upsamplex4(x2)
        x = self.upsamplex4(x)
        # forward small cnn
        x = self.classifier(x)
        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x, x1, x2
    
    def forward(self, x1, x2, imgs_wild=None):
        # forward backbone resnet
        if imgs_wild is None:
            x1, x1_sw = self.forward_single(x1)
            x2, x2_sw = self.forward_single(x2)
        else:
            x1, x1_sw = self.forward_single(x1,imgs_wild[0])
            x2, x2_sw = self.forward_single(x2,imgs_wild[1])
        outputs = []
        if imgs_wild is not None:
            outputs.extend([self._forward_next(x1, x1_sw)[0],
                            self._forward_next(x2, x2_sw)[0]])
            outputs.append(self._forward_next(x1_sw,x2_sw)[0])
        ans, z1, z2 = self._forward_next(x1,x2)
        outputs.extend([z1, z2, ans])
        return outputs

class T3SAW(ResNet):   
    '''
    Timporary Symmetry Siamese Swap + Adain Wild
    '''    
    def __init__(self, input_nc, output_nc, embed_dim, resnet_stages_num=5, backbone='resnet18', output_sigmoid=False, if_upsample_2x=True):
        super().__init__(input_nc, output_nc, resnet_stages_num, backbone, output_sigmoid, if_upsample_2x)
        self.embed_dims = [64,128,256,512]
        self.PCM = T3.PCM(self.embed_dims, iter=1)

        self.depths = [3, 6, 16, 3] #[3, 3, 6, 18, 3]
        self.drop_rate = 0.0
        self.attn_drop = 0.0
        self.drop_path_rate = 0.1 
        self.encoder = T3.EncoderT3(embed_dims=self.embed_dims, num_heads = [1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=self.drop_rate,
                 attn_drop_rate = self.attn_drop, drop_path_rate=self.drop_path_rate, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=self.depths, sr_ratios=[8, 4, 2, 1])
        
        self.embedding_dim = embed_dim
        self.decoder = T3.DecoderT3(input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=False, 
                    in_channels = self.embed_dims, embedding_dim= self.embedding_dim, output_nc=output_nc, 
                    decoder_softmax = False, feature_strides=[2, 4, 8, 16])

    def forward(self, x1, x2, imgs_wild=None):
        # forward backbone resnet
        x1_ls, x2_ls, x1w_ls, x2w_ls = [], [], [], []
        if imgs_wild is None:
            x1, x1_sw = self.forward_single(x1, x_ls=x1_ls)
            x2, x2_sw = self.forward_single(x2, x_ls=x2_ls)
        else:
            x1, x1_sw = self.forward_single(x1,imgs_wild[0], x1_ls, x1w_ls)
            x2, x2_sw = self.forward_single(x2,imgs_wild[1], x2_ls, x2w_ls)
            x1w_ls, x2w_ls  = self.PCM(x1w_ls), self.PCM(x2w_ls)
            x1w_ls, x2w_ls  = self.encoder(x1w_ls), self.encoder(x2w_ls)
            x1w_ls, x2w_ls  = self.decoder(x1w_ls,x2w_ls)
        x1_res, x2_res = self.PCM(x1_ls), self.PCM(x2_ls)
        x1_trans, x2_trans  = self.encoder(x1_ls), self.encoder(x2_ls)
        output_res = self.decoder1(x1_res, x2_res)[-1]
        output_trans = self.decoder2(x1_trans,x2_trans)[-1]
        B, C, H, W = output_res.shape
        output = torch.zeros(B, 2*C, H, W).to('cuda')
        output[:, 0::2, :, :] = output_res
        output[:, 1::2, :, :] = output_trans
        output = self.fuse(output)
        output = F.softmax(input=output, dim=1)
        outputs = [output]
        return outputs
