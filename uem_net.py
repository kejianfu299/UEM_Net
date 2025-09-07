
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from timm.models.layers import trunc_normal_
import math

# 导入SID模块
from SID_module import SIDModule


from wtconv2d import WTConv2d

from UE_Mamba_encoder_bi import UE_Mamba_encoder_Gated
from UE_Mamaba_decoder_bi import UE_Mamba_decoder_Gated


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))



class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class DepthwiseSeparableConvWithWTConv2d(nn.Module):
    """深度可分离小波卷积"""

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 wt_levels=1, wt_type='db1'):
        super().__init__()
        self.depthwise = WTConv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=1,
            bias=True,
            wt_levels=wt_levels,
            wt_type=wt_type
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=False
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class HWF_level1(nn.Module):
    """浅层桥接：专注于细节和边缘"""

    def __init__(self, dim_xh, dim_xl):
        super().__init__()
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
        group_size = dim_xl // 2

        # g0: 精细边缘检测
        self.g0 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            DepthwiseSeparableConvWithWTConv2d(
                group_size + 1, group_size + 1,
                kernel_size=3,  # 小核，保持细节
                wt_levels=1,  # 单层，避免过度平滑
                wt_type='db1'  # 最适合边缘
            )
        )

        # g1: 细节纹理
        self.g1 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            DepthwiseSeparableConvWithWTConv2d(
                group_size + 1, group_size + 1,
                kernel_size=3,  # 小核
                wt_levels=1,
                wt_type='db1'  # 对细节敏感
            )
        )

        # g2: 局部模式
        self.g2 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            DepthwiseSeparableConvWithWTConv2d(
                group_size + 1, group_size + 1,
                kernel_size=3,  # 稍大的核
                wt_levels=1,
                wt_type='db1'
            )
        )

        # g3: 初步上下文
        self.g3 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            DepthwiseSeparableConvWithWTConv2d(
                group_size + 1, group_size + 1,
                kernel_size=3,
                wt_levels=1,
                wt_type='db1'
            )
        )

        self.residual_conv = nn.Conv2d(dim_xl + 1, dim_xl, 1)
        self.tail_conv = nn.Sequential(
            LayerNorm(normalized_shape=dim_xl * 2 + 4, data_format='channels_first'),
            nn.Conv2d(dim_xl * 2 + 4, dim_xl, 1),
            nn.GELU()
        )

    def forward(self, xh, xl, mask):
        xl_residual = xl
        mask_residual = mask

        xh = self.pre_project(xh)
        xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode='bilinear', align_corners=True)
        xh = torch.chunk(xh, 4, dim=1)
        xl = torch.chunk(xl, 4, dim=1)

        x0 = self.g0(torch.cat((xh[0], xl[0], mask), dim=1))
        x1 = self.g1(torch.cat((xh[1], xl[1], mask), dim=1))
        x2 = self.g2(torch.cat((xh[2], xl[2], mask), dim=1))
        x3 = self.g3(torch.cat((xh[3], xl[3], mask), dim=1))

        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.tail_conv(x)

        residual = self.residual_conv(torch.cat([xl_residual, mask_residual], dim=1))
        x = x + residual

        return x


class HWF_level2(nn.Module):
    """中间层桥接：平衡细节和语义"""

    def __init__(self, dim_xh, dim_xl):
        super().__init__()
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
        group_size = dim_xl // 2

        # g0: 边缘保持
        self.g0 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            DepthwiseSeparableConvWithWTConv2d(
                group_size + 1, group_size + 1,
                kernel_size=3,
                wt_levels=1,
                wt_type='db2'
            )
        )

        # g1: 纹理分析
        self.g1 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            DepthwiseSeparableConvWithWTConv2d(
                group_size + 1, group_size + 1,
                kernel_size=3,  # 增大核
                wt_levels=1,
                wt_type='db2'
            )
        )

        # g2: 区域特征
        self.g2 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            DepthwiseSeparableConvWithWTConv2d(
                group_size + 1, group_size + 1,
                kernel_size=3,
                wt_levels=1,  # 增加小波层级
                wt_type='db2'
            )
        )

        # g3: 语义信息
        self.g3 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            DepthwiseSeparableConvWithWTConv2d(
                group_size + 1, group_size + 1,
                kernel_size=3,
                wt_levels=1,
                wt_type='db2'
            )
        )

        self.residual_conv = nn.Conv2d(dim_xl + 1, dim_xl, 1)
        self.tail_conv = nn.Sequential(
            LayerNorm(normalized_shape=dim_xl * 2 + 4, data_format='channels_first'),
            nn.Conv2d(dim_xl * 2 + 4, dim_xl, 1),
            nn.GELU()
        )

    def forward(self, xh, xl, mask):
        # 与level1相同的forward实现
        xl_residual = xl
        mask_residual = mask

        xh = self.pre_project(xh)
        xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode='bilinear', align_corners=True)
        xh = torch.chunk(xh, 4, dim=1)
        xl = torch.chunk(xl, 4, dim=1)

        x0 = self.g0(torch.cat((xh[0], xl[0], mask), dim=1))
        x1 = self.g1(torch.cat((xh[1], xl[1], mask), dim=1))
        x2 = self.g2(torch.cat((xh[2], xl[2], mask), dim=1))
        x3 = self.g3(torch.cat((xh[3], xl[3], mask), dim=1))

        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.tail_conv(x)

        residual = self.residual_conv(torch.cat([xl_residual, mask_residual], dim=1))
        x = x + residual

        return x



class HWF_level3(nn.Module):
    """深层桥接：专注于语义和全局信息"""

    def __init__(self, dim_xh, dim_xl):
        super().__init__()
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
        group_size = dim_xl // 2

        # g0: 结构信息
        self.g0 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            DepthwiseSeparableConvWithWTConv2d(
                group_size + 1, group_size + 1,
                kernel_size=3,
                wt_levels=2,  # 多层小波
                wt_type='sym3'
            )
        )

        # g1: 语义特征
        self.g1 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            DepthwiseSeparableConvWithWTConv2d(
                group_size + 1, group_size + 1,
                kernel_size=3,
                wt_levels=2,
                wt_type='sym3'  # 更高阶小波
            )
        )

        # g2: 全局模式
        self.g2 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            DepthwiseSeparableConvWithWTConv2d(
                group_size + 1, group_size + 1,
                kernel_size=3,
                wt_levels=2,  # 3层小波
                wt_type='sym3'  # 适合全局特征
            )
        )

        # g3: 上下文聚合
        self.g3 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            DepthwiseSeparableConvWithWTConv2d(
                group_size + 1, group_size + 1,
                kernel_size=3,
                wt_levels=2,
                wt_type='sym3'  # 高阶Daubechies
            )
        )

        self.residual_conv = nn.Conv2d(dim_xl + 1, dim_xl, 1)
        self.tail_conv = nn.Sequential(
            LayerNorm(normalized_shape=dim_xl * 2 + 4, data_format='channels_first'),
            nn.Conv2d(dim_xl * 2 + 4, dim_xl, 1),
            nn.GELU()
        )

    def forward(self, xh, xl, mask):
        # 与level1相同的forward实现
        xl_residual = xl
        mask_residual = mask

        xh = self.pre_project(xh)
        xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode='bilinear', align_corners=True)
        xh = torch.chunk(xh, 4, dim=1)
        xl = torch.chunk(xl, 4, dim=1)

        x0 = self.g0(torch.cat((xh[0], xl[0], mask), dim=1))
        x1 = self.g1(torch.cat((xh[1], xl[1], mask), dim=1))
        x2 = self.g2(torch.cat((xh[2], xl[2], mask), dim=1))
        x3 = self.g3(torch.cat((xh[3], xl[3], mask), dim=1))

        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.tail_conv(x)

        residual = self.residual_conv(torch.cat([xl_residual, mask_residual], dim=1))
        x = x + residual

        return x

class uem_net(nn.Module):

    def __init__(self, num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], bridge=True, gt_ds=True,
                 use_sid=True):
        super().__init__()

        self.bridge = bridge
        self.gt_ds = gt_ds
        self.use_sid = use_sid

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )
        self.encoder4 = nn.Sequential(
            UE_Mamba_encoder_Gated(c_list[2], c_list[3]),
        )
        # self.tokenizedKANLayer1 = TokenizedKANBlock(c_list[3], patch_size=8)
        self.encoder5 = nn.Sequential(
            UE_Mamba_encoder_Gated(c_list[3], c_list[4]),
        )
        # self.tokenizedKANLayer2 = TokenizedKANBlock(c_list[4], patch_size=4)
        self.encoder6 = nn.Sequential(
            UE_Mamba_encoder_Gated(c_list[4], c_list[5]),
        )

        # 添加SID模块
        if self.use_sid:
            self.sid_module = SIDModule(c_list[5], scale_factor=32)
            print('SID module was used')

        if bridge:
            self.HWF1 = HWF_level1(c_list[1], c_list[0])
            self.HWF2 = HWF_level2(c_list[2], c_list[1])
            self.HWF3 = HWF_level2(c_list[3], c_list[2])
            self.HWF4 = HWF_level3(c_list[4], c_list[3])
            self.HWF5 = HWF_level3(c_list[5], c_list[4])
            print('group_aggregation_bridge was used')
        if gt_ds:
            self.gt_conv1 = nn.Sequential(nn.Conv2d(c_list[4], 1, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(c_list[3], 1, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(c_list[2], 1, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(c_list[1], 1, 1))
            self.gt_conv5 = nn.Sequential(nn.Conv2d(c_list[0], 1, 1))
            print('gt deep supervision was used')

        # 修改decoder1-3，使其能接收SID masks
        self.decoder1 = UE_Mamba_decoder_Gated(c_list[5], c_list[4])
        # self.decoder_dropout1 = nn.Dropout2d(0.3)
        # self.tokenizedKANLayer4 = TokenizedKANBlock(c_list[4], patch_size=4)

        self.decoder2 = UE_Mamba_decoder_Gated(c_list[4], c_list[3])
        # self.decoder_dropout2 = nn.Dropout2d(0.3)
        # self.tokenizedKANLayer5 = TokenizedKANBlock(c_list[3], patch_size=4)

        self.decoder3 = UE_Mamba_decoder_Gated(c_list[3], c_list[2])
        # self.decoder_dropout3 = nn.Dropout2d(0.3)
        # self.tokenizedKANLayer6 = TokenizedKANBlock(c_list[2], patch_size=4)

        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()



    def forward(self, x):

        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out  # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # b, c1, H/4, W/4

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out  # b, c2, H/8, W/8

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out  # b, c3, H/16, W/16

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out  # b, c4, H/32, W/32

        out = F.gelu(self.encoder6(out))  # b, c5, H/32, W/32
        t6 = out

        # 使用SID模块
        if self.use_sid:
            (f_fg, f_bg, f_uc), (mask_fg, mask_bg, mask_uc) = self.sid_module(out)
            # 为decoder准备masks（需要下采样到相应尺寸）
            mask_fg_32 = F.interpolate(mask_fg, size=(out.shape[2], out.shape[3]), mode='bilinear', align_corners=True)
            mask_bg_32 = F.interpolate(mask_bg, size=(out.shape[2], out.shape[3]), mode='bilinear', align_corners=True)
            mask_uc_32 = F.interpolate(mask_uc, size=(out.shape[2], out.shape[3]), mode='bilinear', align_corners=True)
        else:
            mask_fg = mask_bg = mask_uc = None
            mask_fg_32 = mask_bg_32 = mask_uc_32 = None

        # Decoder with SID masks
        out5 = F.gelu(self.dbn1(self.decoder1(out, mask_fg_32, mask_bg_32,mask_uc_32)))  # b, c4, H/32, W/32
        if self.gt_ds:
            gt_pre5 = self.gt_conv1(out5)
            t5 = self.HWF5(t6, t5, gt_pre5)
            gt_pre5 = F.interpolate(gt_pre5, scale_factor=32, mode='bilinear', align_corners=True)
        else:
            t5 = self.HWF5(t6, t5)
        out5 = torch.add(out5, t5)  # b, c4, H/32, W/32

        # 准备16x16的masks
        if self.use_sid:
            mask_fg_16 = F.interpolate(mask_fg, size=(out.shape[2] * 2, out.shape[3] * 2), mode='bilinear',
                                       align_corners=True)
            mask_bg_16 = F.interpolate(mask_bg, size=(out.shape[2] * 2, out.shape[3] * 2), mode='bilinear',
                                       align_corners=True)
            mask_uc_16 = F.interpolate(mask_uc, size=(out.shape[2] * 2, out.shape[3] * 2), mode='bilinear',
                                       align_corners=True)
        else:
            mask_fg_16 = mask_bg_16 = mask_uc_16 = None

        out4 = F.gelu(
            F.interpolate(self.dbn2(self.decoder2(out5, mask_fg_16, mask_bg_16, mask_uc_16)), scale_factor=(2, 2),
                          mode='bilinear', align_corners=True))  # b, c3, H/16, W/16
        if self.gt_ds:
            gt_pre4 = self.gt_conv2(out4)
            t4 = self.HWF4(t5, t4, gt_pre4)
            gt_pre4 = F.interpolate(gt_pre4, scale_factor=16, mode='bilinear', align_corners=True)
        else:
            t4 = self.HWF4(t5, t4)
        out4 = torch.add(out4, t4)  # b, c3, H/16, W/16

        # 准备8x8的masks
        if self.use_sid:
            mask_fg_8 = F.interpolate(mask_fg, size=(out.shape[2] * 4, out.shape[3] * 4), mode='bilinear',
                                      align_corners=True)
            mask_bg_8 = F.interpolate(mask_bg, size=(out.shape[2] * 4, out.shape[3] * 4), mode='bilinear',
                                      align_corners=True)
            mask_uc_8 = F.interpolate(mask_uc, size=(out.shape[2] * 4, out.shape[3] * 4), mode='bilinear',
                                      align_corners=True)
        else:
            mask_fg_8 = mask_bg_8 = mask_uc_8 = None

        out3 = F.gelu(
            F.interpolate(self.dbn3(self.decoder3(out4, mask_fg_8, mask_bg_8, mask_uc_8)), scale_factor=(2, 2),
                          mode='bilinear', align_corners=True))  # b, c2, H/8, W/8
        if self.gt_ds:
            gt_pre3 = self.gt_conv3(out3)
            t3 = self.HWF3(t4, t3, gt_pre3)
            gt_pre3 = F.interpolate(gt_pre3, scale_factor=8, mode='bilinear', align_corners=True)
        else:
            t3 = self.HWF3(t4, t3)
        out3 = torch.add(out3, t3)  # b, c2, H/8, W/8

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c1, H/4, W/4
        if self.gt_ds:
            gt_pre2 = self.gt_conv4(out2)
            t2 = self.HWF2(t3, t2, gt_pre2)
            gt_pre2 = F.interpolate(gt_pre2, scale_factor=4, mode='bilinear', align_corners=True)
        else:
            t2 = self.HWF2(t3, t2)
        out2 = torch.add(out2, t2)  # b, c1, H/4, W/4

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c0, H/2, W/2
        if self.gt_ds:
            gt_pre1 = self.gt_conv5(out1)
            t1 = self.HWF1(t2, t1, gt_pre1)
            gt_pre1 = F.interpolate(gt_pre1, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            t1 = self.HWF1(t2, t1)
        out1 = torch.add(out1, t1)  # b, c0, H/2, W/2

        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear',
                             align_corners=True)  # b, num_class, H, W
        if self.gt_ds and self.use_sid:
            return (torch.sigmoid(gt_pre5), torch.sigmoid(gt_pre4), torch.sigmoid(gt_pre3), torch.sigmoid(gt_pre2),
                    torch.sigmoid(gt_pre1)), torch.sigmoid(out0), mask_fg, mask_bg, mask_uc
        elif self.gt_ds:
            return (torch.sigmoid(gt_pre5), torch.sigmoid(gt_pre4), torch.sigmoid(gt_pre3), torch.sigmoid(gt_pre2),
                    torch.sigmoid(gt_pre1)), torch.sigmoid(out0)
        elif self.use_sid:
            return torch.sigmoid(out0), mask_fg, mask_bg, mask_uc
        else:
            return torch.sigmoid(out0)


