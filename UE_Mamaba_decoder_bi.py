import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


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


class GatedFusion(nn.Module):
    """门控机制融合模块"""

    def __init__(self, channels):
        super().__init__()
        # 修正：输入是channels + channels = channels*2
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, concatenated_features, sid_features):
        # concatenated_features: [B, C, H, W]
        # sid_features: [B, C, H, W]
        gate_input = torch.cat([concatenated_features, sid_features], dim=1)
        gate_weight = self.gate(gate_input)
        output = gate_weight * sid_features + (1 - gate_weight) * concatenated_features
        return output


class UE_Mamba_decoder_Gated(nn.Module):
    """使用门控机制的UE_Mamba_decoder - 去除vertical，使用通道分割"""

    def __init__(self, in_channels, out_channels, d_state=None, dt_rank=None, kernel_size=3,
                 scan_order='fg_bg_uc', dt_min=0.001, dt_max=0.1,
                 dt_init="random", dt_scale=1.0, dt_init_floor=1e-4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.d_state = d_state if d_state is not None else 16
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(in_channels / 32)
        self.scan_order = scan_order

        # 修改为输出 2*C 通道 (mamba分支 + 残差分支)
        self.in_proj = nn.Conv2d(in_channels, in_channels * 2, kernel_size=1)

        # 共享的归一化和卷积（用于mamba分支）
        self.ln_mamba = LayerNorm(in_channels, data_format="channels_first")
        self.conv2d_mamba = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                      padding=kernel_size // 2, groups=in_channels)

        # 残差分支的卷积
        self.conv2d_residual = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                         padding=kernel_size // 2, groups=in_channels)

        # 处理后的归一化
        self.ln_concatenated = LayerNorm(in_channels, data_format="channels_first")  # 修正：是in_channels而不是in_channels*2

        # 门控融合模块
        self.gated_fusion = GatedFusion(in_channels)

        # 水平正向扫描的参数 - 处理一半通道
        half_channels = in_channels // 2
        self.dt_proj_forward = nn.Linear(self.dt_rank, half_channels)
        self.x_proj_forward = nn.Linear(half_channels, self.dt_rank + self.d_state * 2)

        # 水平反向扫描的参数 - 处理另一半通道
        self.dt_proj_backward = nn.Linear(self.dt_rank, half_channels)
        self.x_proj_backward = nn.Linear(half_channels, self.dt_rank + self.d_state * 2)

        # SID引导扫描的参数 - 处理全部通道
        self.dt_proj_sid = nn.Linear(self.dt_rank * 2, in_channels)
        self.x_proj_sid = nn.Linear(in_channels, self.dt_rank * 2 + self.d_state * 2)

        # 备用扫描顺序的参数
        self.dt_proj_sid_alt = nn.Linear(self.dt_rank * 2, in_channels)
        self.x_proj_sid_alt = nn.Linear(in_channels, self.dt_rank * 2 + self.d_state * 2)

        # 输出投影：2*C -> out_channels
        self.out_proj = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)

        # LayerNorm
        self.norm_in = LayerNorm(in_channels, data_format="channels_first")
        self.norm_out = LayerNorm(out_channels, data_format="channels_first")

        # 初始化dt projectors
        self._init_dt_projectors(dt_init, dt_scale, dt_min, dt_max, dt_init_floor)

        # 初始化SSM参数
        self._init_ssm_parameters()

    def _init_dt_projectors(self, dt_init, dt_scale, dt_min, dt_max, dt_init_floor):
        """初始化所有dt投影器的权重和偏置"""
        dt_init_std = self.dt_rank ** -0.5 * dt_scale

        dt_proj_names = ['dt_proj_forward', 'dt_proj_backward',
                         'dt_proj_sid', 'dt_proj_sid_alt']

        for name in dt_proj_names:
            dt_proj = getattr(self, name)

            # 初始化权重
            if dt_init == "constant":
                nn.init.constant_(dt_proj.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
            else:
                raise NotImplementedError

            # 初始化偏置 - 根据处理的通道数调整
            if 'sid' in name:
                channels = self.in_channels
            else:
                channels = self.in_channels // 2

            dt = torch.exp(
                torch.rand(channels) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)

            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            dt_proj.bias._no_reinit = True

    def _init_ssm_parameters(self):
        """初始化SSM相关参数"""
        half_channels = self.in_channels // 2

        # Forward和Backward参数 - 半通道
        for suffix in ['forward', 'backward']:
            A = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32),
                "n -> d n",
                d=half_channels,
            ).contiguous()
            A_log = torch.log(A)
            A_log_param = nn.Parameter(A_log)
            A_log_param._no_weight_decay = True
            setattr(self, f'A_log_{suffix}', A_log_param)

            D = nn.Parameter(torch.ones(half_channels))
            D._no_weight_decay = True
            setattr(self, f'D_{suffix}', D)

        # SID参数 - 全通道
        for suffix in ['sid', 'sid_alt']:
            A = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32),
                "n -> d n",
                d=self.in_channels,
            ).contiguous()
            A_log = torch.log(A)
            A_log_param = nn.Parameter(A_log)
            A_log_param._no_weight_decay = True
            setattr(self, f'A_log_{suffix}', A_log_param)

            D = nn.Parameter(torch.ones(self.in_channels))
            D._no_weight_decay = True
            setattr(self, f'D_{suffix}', D)

    def forward(self, x, mask_fg=None, mask_bg=None, mask_uc=None):
        """
        Args:
            x: 输入特征 [B, C, H, W]
            mask_fg: 前景mask [B, 1, H', W']
            mask_bg: 背景mask [B, 1, H', W']
            mask_uc: 不确定性mask [B, 1, H', W']
        """
        x = self.norm_in(x)

        B, C, H, W = x.shape
        L = H * W

        # 投影到 2*C 通道
        x_proj = self.in_proj(x)

        # 分割为mamba分支和残差分支
        x_mamba, x_residual = torch.chunk(x_proj, 2, dim=1)

        # mamba分支的处理
        x_mamba = self.ln_mamba(x_mamba)
        x_mamba = F.silu(self.conv2d_mamba(x_mamba))

        # 残差分支
        x_residual = F.silu(self.conv2d_residual(x_residual))

        # 将x_mamba在通道维度分成两部分
        x_mamba_forward, x_mamba_backward = torch.chunk(x_mamba, 2, dim=1)

        # 两个方向的扫描（使用不同的输入）
        x_forward_ssm = self.forward_scan(x_mamba_forward, B, C // 2, H, W, L)
        x_backward_ssm = self.backward_scan(x_mamba_backward, B, C // 2, H, W, L)

        # 通道拼接替代均值
        concatenated_features = torch.cat([x_forward_ssm, x_backward_ssm], dim=1)  # [B, C, H, W]
        concatenated_features = self.ln_concatenated(concatenated_features)

        # 使用SID masks处理
        if mask_fg is not None and mask_bg is not None and mask_uc is not None:
            if self.scan_order == 'fg_bg_uc':
                x_sid_processed = self.sid_guided_process_fg_bg_uc(
                    x_mamba, mask_fg, mask_bg, mask_uc, B, C, H, W, L)
            else:
                x_sid_processed = self.sid_guided_process_uc_fg_bg(
                    x_mamba, mask_fg, mask_bg, mask_uc, B, C, H, W, L)

            # 使用门控机制融合concatenated特征和SID处理后的特征
            fused_features = self.gated_fusion(concatenated_features, x_sid_processed)
        else:
            # 如果没有mask，直接使用concatenated特征
            fused_features = concatenated_features

        # 与残差分支拼接
        out = torch.cat([fused_features, x_residual], dim=1)  # [B, 2*C, H, W]
        out = self.out_proj(out)
        out = self.norm_out(out)

        return out

    def sid_guided_process_fg_bg_uc(self, x_mamba, mask_fg, mask_bg, mask_uc, B, C, H, W, L):
        """基于SID masks的引导处理，顺序：前景、背景、不确定性"""
        # 确保mask尺寸匹配
        if mask_fg.shape[2:] != (H, W):
            mask_fg = F.interpolate(mask_fg, size=(H, W), mode='bilinear', align_corners=True)
            mask_bg = F.interpolate(mask_bg, size=(H, W), mode='bilinear', align_corners=True)
            mask_uc = F.interpolate(mask_uc, size=(H, W), mode='bilinear', align_corners=True)

        # 二值化
        threshold = 0.5
        mask_fg = (mask_fg > threshold).float()
        mask_bg = (mask_bg > threshold).float()
        mask_uc = (mask_uc > threshold).float()

        # 确保互斥
        mask_bg = mask_bg * (1 - mask_fg)
        mask_uc = mask_uc * (1 - mask_fg) * (1 - mask_bg)

        # 展平
        mask_fg_flat = mask_fg.view(B, -1).squeeze(1)
        mask_bg_flat = mask_bg.view(B, -1).squeeze(1)
        mask_uc_flat = mask_uc.view(B, -1).squeeze(1)

        # 创建优先级map
        priority_map = mask_fg_flat * 3 + mask_bg_flat * 2 + mask_uc_flat * 1
        priority_map = priority_map + torch.arange(L, device=x_mamba.device).unsqueeze(0) * 0.0001

        # 排序
        sorted_indices = torch.argsort(priority_map, dim=1, descending=True)

        return self._perform_sid_scan(x_mamba, sorted_indices, B, C, H, W, L, use_alt=False)


    def _perform_sid_scan(self, features, sorted_indices, B, C, H, W, L, use_alt=False):
        """执行基于排序索引的扫描处理"""
        # 准备逆索引
        batch_indices = torch.arange(B, device=features.device).unsqueeze(1).expand(-1, L)
        inverse_indices = torch.empty_like(sorted_indices)
        inverse_indices[batch_indices, sorted_indices] = torch.arange(L, device=features.device).expand(B, -1)

        # 重排特征
        x_flat = features.view(B, C, -1)
        x_reordered = torch.gather(x_flat, 2, sorted_indices.unsqueeze(1).expand(-1, C, -1))

        # 选择参数
        if use_alt:
            x_proj = self.x_proj_sid_alt
            dt_proj = self.dt_proj_sid_alt
            A_log = self.A_log_sid_alt
            D = self.D_sid_alt
            dt_bias = self.dt_proj_sid_alt.bias
        else:
            x_proj = self.x_proj_sid
            dt_proj = self.dt_proj_sid
            A_log = self.A_log_sid
            D = self.D_sid
            dt_bias = self.dt_proj_sid.bias

        # SSM扫描
        x_reordered_2d = x_reordered.view(B, C, H, W)
        x_flat_reordered = rearrange(x_reordered_2d, 'b c h w -> (b h w) c')
        x_dbl = x_proj(x_flat_reordered)
        dt, B_, C_ = torch.split(x_dbl, [self.dt_rank * 2, self.d_state, self.d_state], dim=-1)
        dt = dt_proj(dt)

        dt = dt.reshape(B, C, L)
        B_ = B_.reshape(B, self.d_state, L)
        C_ = C_.reshape(B, self.d_state, L)

        # 使用负指数A
        A = -torch.exp(A_log.float())

        # 执行选择性扫描
        x_ssm = selective_scan_fn(
            x_reordered, dt, A, B_, C_, D.float(),
            z=None,
            delta_bias=dt_bias.float(),
            delta_softplus=True,
            return_last_state=False
        )

        # 恢复原始顺序
        x_ssm_restored = torch.gather(x_ssm, 2, inverse_indices.unsqueeze(1).expand(-1, C, -1))
        x_ssm_restored = x_ssm_restored.reshape(B, C, H, W)

        return x_ssm_restored

    def forward_scan(self, x, B, C, H, W, L):
        """水平正向扫描 - 处理一半通道"""
        x_flat = rearrange(x, 'b c h w -> (b h w) c')
        x_dbl = self.x_proj_forward(x_flat)
        dt, B_, C_ = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj_forward(dt)

        x_seq = x.reshape(B, C, L)
        dt = dt.reshape(B, C, L)
        B_ = B_.reshape(B, self.d_state, L)
        C_ = C_.reshape(B, self.d_state, L)

        # 使用负指数A
        A = -torch.exp(self.A_log_forward.float())

        x_ssm = selective_scan_fn(
            x_seq, dt, A, B_, C_, self.D_forward.float(),
            z=None,
            delta_bias=self.dt_proj_forward.bias.float(),
            delta_softplus=True,
            return_last_state=False
        )

        return x_ssm.reshape(B, C, H, W)

    def backward_scan(self, x, B, C, H, W, L):
        """水平反向扫描 - 处理另一半通道"""
        x_flipped = torch.flip(x, dims=[2, 3])

        x_flat = rearrange(x_flipped, 'b c h w -> (b h w) c')
        x_dbl = self.x_proj_backward(x_flat)
        dt, B_, C_ = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj_backward(dt)

        x_seq = x_flipped.reshape(B, C, L)
        dt = dt.reshape(B, C, L)
        B_ = B_.reshape(B, self.d_state, L)
        C_ = C_.reshape(B, self.d_state, L)

        # 使用负指数A
        A = -torch.exp(self.A_log_backward.float())

        x_ssm = selective_scan_fn(
            x_seq, dt, A, B_, C_, self.D_backward.float(),
            z=None,
            delta_bias=self.dt_proj_backward.bias.float(),
            delta_softplus=True,
            return_last_state=False
        )

        x_ssm = x_ssm.reshape(B, C, H, W)
        return torch.flip(x_ssm, dims=[2, 3])