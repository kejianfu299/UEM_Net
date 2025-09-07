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
        # 修正：输入是channels + channels (concatenated + uncertainty) = channels*2
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, concatenated_features, uncertainty_features):
        # concatenated_features: [B, C, H, W]
        # uncertainty_features: [B, C, H, W]
        gate_input = torch.cat([concatenated_features, uncertainty_features], dim=1)
        gate_weight = self.gate(gate_input)
        # 自适应融合
        output = gate_weight * uncertainty_features + (1 - gate_weight) * concatenated_features
        return output


class UE_Mamba_encoder_Gated(nn.Module):
    """使用门控机制的不确定性增强Mamba编码器 - 去除vertical，使用通道分割"""

    def __init__(self, in_channels, out_channels, d_state=None, dt_rank=None, kernel_size=3,
                 dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 如果没有提供d_state，使用默认值16
        self.d_state = d_state if d_state is not None else 16

        # 如果没有提供dt_rank，使用默认计算方式
        # 注意：由于通道被分割，每个方向只处理一半通道
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(in_channels / 32)

        # 输入归一化
        self.norm_in = LayerNorm(in_channels, data_format="channels_first")

        # 输入投影：将输入分为mamba分支和残差分支
        self.in_proj = nn.Conv2d(in_channels, in_channels * 2, kernel_size=1)

        # Mamba分支的归一化和卷积
        self.ln_mamba = LayerNorm(in_channels, data_format="channels_first")
        self.conv2d_mamba = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                      padding=kernel_size // 2, groups=in_channels)

        # 残差分支的卷积
        self.conv2d_residual = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                         padding=kernel_size // 2, groups=in_channels)

        # 处理后的归一化
        self.ln_concatenated = LayerNorm(in_channels, data_format="channels_first")  # 修正：是in_channels而不是in_channels*2
        self.out_ln = LayerNorm(out_channels, data_format="channels_first")

        # 门控融合模块
        self.gated_fusion = GatedFusion(in_channels)

        # 水平正向扫描的参数 - 处理一半通道
        half_channels = in_channels // 2
        self.dt_proj_forward = nn.Linear(self.dt_rank, half_channels)
        self.x_proj_forward = nn.Linear(half_channels, self.dt_rank + self.d_state * 2)

        # 水平反向扫描的参数 - 处理另一半通道
        self.dt_proj_backward = nn.Linear(self.dt_rank, half_channels)
        self.x_proj_backward = nn.Linear(half_channels, self.dt_rank + self.d_state * 2)

        # 不确定性引导扫描的参数 - 仍处理全部通道
        self.dt_proj_uncertainty = nn.Linear(self.dt_rank * 2, in_channels)
        self.x_proj_uncertainty = nn.Linear(in_channels, self.dt_rank * 2 + self.d_state * 2)

        # 输出投影：2*C (C from fused + C from residual) -> out_channels
        self.out_proj = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)

        # 初始化dt projectors
        self._init_dt_projectors(dt_init, dt_scale, dt_min, dt_max, dt_init_floor)

        # 初始化SSM参数（A和D）
        self._init_ssm_parameters()

    def _init_dt_projectors(self, dt_init, dt_scale, dt_min, dt_max, dt_init_floor):
        """初始化所有dt投影器的权重和偏置"""
        dt_init_std = self.dt_rank ** -0.5 * dt_scale

        # 只初始化forward, backward和uncertainty
        for name in ['dt_proj_forward', 'dt_proj_backward', 'dt_proj_uncertainty']:
            dt_proj = getattr(self, name)

            # 初始化权重
            if dt_init == "constant":
                nn.init.constant_(dt_proj.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
            else:
                raise NotImplementedError

            # 初始化偏置 - 注意通道数不同
            if 'uncertainty' in name:
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
        """初始化SSM相关参数（A和D）"""
        # forward和backward使用一半通道
        half_channels = self.in_channels // 2

        # Forward参数 - 半通道
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=half_channels,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log_forward = nn.Parameter(A_log)
        self.A_log_forward._no_weight_decay = True
        self.D_forward = nn.Parameter(torch.ones(half_channels))
        self.D_forward._no_weight_decay = True

        # Backward参数 - 半通道
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=half_channels,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log_backward = nn.Parameter(A_log)
        self.A_log_backward._no_weight_decay = True
        self.D_backward = nn.Parameter(torch.ones(half_channels))
        self.D_backward._no_weight_decay = True

        # Uncertainty参数 - 全通道
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.in_channels,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log_uncertainty = nn.Parameter(A_log)
        self.A_log_uncertainty._no_weight_decay = True
        self.D_uncertainty = nn.Parameter(torch.ones(self.in_channels))
        self.D_uncertainty._no_weight_decay = True

    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W

        # 输入归一化
        x = self.norm_in(x)

        # 投影到 2*C 通道
        x_proj = self.in_proj(x)

        # 分割为mamba分支和残差分支
        x_mamba, x_residual = torch.chunk(x_proj, 2, dim=1)

        # mamba分支的处理
        x_mamba = self.ln_mamba(x_mamba)
        x_mamba = F.silu(self.conv2d_mamba(x_mamba))

        # 残差分支的处理
        x_residual = F.silu(self.conv2d_residual(x_residual))

        # 将x_mamba在通道维度分成两部分
        x_mamba_forward, x_mamba_backward = torch.chunk(x_mamba, 2, dim=1)

        # 两个方向的扫描（使用不同的输入）
        x_forward_ssm = self.forward_scan(x_mamba_forward, B, C // 2, H, W, L)
        x_backward_ssm = self.backward_scan(x_mamba_backward, B, C // 2, H, W, L)

        # 通道拼接替代均值
        concatenated_features = torch.cat([x_forward_ssm, x_backward_ssm], dim=1)  # [B, C, H, W]
        concatenated_features = self.ln_concatenated(concatenated_features)

        # 对不确定性分支进行选择性扫描（使用完整的x_mamba）
        x_uncertainty_ssm = self.selective_scan_uncertainty(x_mamba, B, C, H, W, L)

        # 使用门控机制融合concatenated特征和不确定性特征
        fused_features = self.gated_fusion(concatenated_features, x_uncertainty_ssm)

        # 与残差分支进行通道拼接
        out = torch.cat([fused_features, x_residual], dim=1)  # [B, 2*C, H, W]
        out = self.out_proj(out)
        out = self.out_ln(out)

        return out

    def selective_scan_uncertainty(self, x, B, C, H, W, L):
        """基于不确定性的选择性扫描"""
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x_mean_normalized = torch.sigmoid(x_mean)

        uncertainty = -(x_mean_normalized * torch.log(x_mean_normalized + 1e-6))
        uncertainty_flat = uncertainty.view(B, -1)

        sorted_indices = torch.argsort(uncertainty_flat, dim=1, descending=True)

        batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand_as(sorted_indices)
        restore_indices = torch.zeros_like(sorted_indices)
        restore_indices[batch_indices, sorted_indices] = torch.arange(L, device=x.device).unsqueeze(0).expand_as(
            sorted_indices)

        x_seq = x.reshape(B, C, L)
        x_sorted = torch.gather(x_seq, 2, sorted_indices.unsqueeze(1).expand(-1, C, -1))

        x_flat = rearrange(x_sorted, 'b c l -> (b l) c')
        x_dbl = self.x_proj_uncertainty(x_flat)
        dt, B_, C_ = torch.split(x_dbl, [self.dt_rank * 2, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj_uncertainty(dt)

        dt = dt.reshape(B, C, L)
        B_ = B_.reshape(B, self.d_state, L)
        C_ = C_.reshape(B, self.d_state, L)

        A = -torch.exp(self.A_log_uncertainty.float())

        x_ssm = selective_scan_fn(
            x_sorted, dt, A, B_, C_, self.D_uncertainty.float(),
            z=None,
            delta_bias=self.dt_proj_uncertainty.bias.float(),
            delta_softplus=True,
            return_last_state=False
        )

        x_restored = torch.gather(x_ssm, 2, restore_indices.unsqueeze(1).expand(-1, C, -1))

        return x_restored.reshape(B, C, H, W)

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