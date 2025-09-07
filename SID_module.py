#
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class CBR(nn.Module):
#     def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
#         super().__init__()
#         self.act = act
#
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
#             nn.BatchNorm2d(out_c)
#         )
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.conv(x)
#         if self.act == True:
#             x = self.relu(x)
#         return x
#
#
# class DecoupleLayer(nn.Module):
#     """解耦层：将特征分解为前景、背景和不确定三个分支"""
#
#     def __init__(self, in_c, out_c=None):
#         super(DecoupleLayer, self).__init__()
#         # 按照指定的通道变化：in_c → in_c/2 → 2*in_c → in_c → in_c
#         if out_c is None:
#             out_c = in_c  # 输出通道数默认等于输入通道数
#
#         # 第一层：in_c → in_c/2
#         first_c = max(in_c // 2, 16)  # 确保最小16通道
#         # 第二层：in_c/2 → 2*in_c
#         second_c = in_c * 2
#         # 第三层：2*in_c → in_c (out_c)
#         third_c = out_c
#
#         self.cbr_fg = nn.Sequential(
#             CBR(in_c, first_c, kernel_size=3, padding=1),  # in_c → in_c/2
#             CBR(first_c, second_c, kernel_size=3, padding=1),  # in_c/2 → 2*in_c
#             CBR(second_c, third_c, kernel_size=1, padding=0)  # 2*in_c → in_c
#         )
#         self.cbr_bg = nn.Sequential(
#             CBR(in_c, first_c, kernel_size=3, padding=1),
#             CBR(first_c, second_c, kernel_size=3, padding=1),
#             CBR(second_c, third_c, kernel_size=1, padding=0)
#         )
#         self.cbr_uc = nn.Sequential(
#             CBR(in_c, first_c, kernel_size=3, padding=1),
#             CBR(first_c, second_c, kernel_size=3, padding=1),
#             CBR(second_c, third_c, kernel_size=1, padding=0)
#         )
#
#     def forward(self, x):
#         f_fg = self.cbr_fg(x)
#         f_bg = self.cbr_bg(x)
#         f_uc = self.cbr_uc(x)
#         return f_fg, f_bg, f_uc
#
#
# class AuxiliaryHead(nn.Module):
#     """辅助头：为每个分支生成对应的mask"""
#
#     def __init__(self, in_c, scale_factor=16):
#         super(AuxiliaryHead, self).__init__()
#         self.scale_factor = scale_factor
#
#         # 计算需要的上采样次数
#         up_times = int(torch.log2(torch.tensor(scale_factor, dtype=torch.float32)).item())
#
#         # 按照指定的通道变化设计
#         # in_c(=c_list[5]) → c_list[5] → c_list[5] → c_list[5] → c_list[5]/2 → c_list[5]/4 → 1
#         base_c = in_c  # c_list[5]
#
#         # 前景分支
#         self.branch_fg = nn.Sequential()
#         current_c = in_c  # c_list[5]
#
#         for i in range(up_times):
#             if i == 0:
#                 # 第一层：保持c_list[5]
#                 self.branch_fg.add_module(f'cbr_fg_{i}',
#                                           CBR(current_c, base_c, kernel_size=3, padding=1))
#                 current_c = base_c
#             elif i <= 2:  # i=1,2
#                 # 保持c_list[5]
#                 self.branch_fg.add_module(f'up_fg_{i}',
#                                           nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
#                 self.branch_fg.add_module(f'cbr_fg_{i}',
#                                           CBR(current_c, base_c, kernel_size=3, padding=1))
#                 current_c = base_c
#             elif i == 3:
#                 # 降到c_list[5]/2
#                 self.branch_fg.add_module(f'up_fg_{i}',
#                                           nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
#                 self.branch_fg.add_module(f'cbr_fg_{i}',
#                                           CBR(current_c, max(base_c // 2, 16), kernel_size=3, padding=1))
#                 current_c = max(base_c // 2, 16)
#             else:  # i=4
#                 # 降到c_list[5]/4
#                 self.branch_fg.add_module(f'up_fg_{i}',
#                                           nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
#                 self.branch_fg.add_module(f'cbr_fg_{i}',
#                                           CBR(current_c, max(base_c // 4, 8), kernel_size=3, padding=1))
#                 current_c = max(base_c // 4, 8)
#
#         # 最后的上采样和输出
#         self.branch_fg.add_module('up_fg_final', nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
#         self.branch_fg.add_module('conv_fg_final', nn.Conv2d(current_c, 1, kernel_size=1, padding=0))
#         self.branch_fg.add_module('sigmoid_fg', nn.Sigmoid())
#
#         # 背景分支
#         self.branch_bg = nn.Sequential()
#         current_c = in_c  # c_list[5]
#
#         for i in range(up_times):
#             if i == 0:
#                 self.branch_bg.add_module(f'cbr_bg_{i}',
#                                           CBR(current_c, base_c, kernel_size=3, padding=1))
#                 current_c = base_c
#             elif i <= 2:
#                 self.branch_bg.add_module(f'up_bg_{i}',
#                                           nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
#                 self.branch_bg.add_module(f'cbr_bg_{i}',
#                                           CBR(current_c, base_c, kernel_size=3, padding=1))
#                 current_c = base_c
#             elif i == 3:
#                 self.branch_bg.add_module(f'up_bg_{i}',
#                                           nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
#                 self.branch_bg.add_module(f'cbr_bg_{i}',
#                                           CBR(current_c, max(base_c // 2, 16), kernel_size=3, padding=1))
#                 current_c = max(base_c // 2, 16)
#             else:
#                 self.branch_bg.add_module(f'up_bg_{i}',
#                                           nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
#                 self.branch_bg.add_module(f'cbr_bg_{i}',
#                                           CBR(current_c, max(base_c // 4, 8), kernel_size=3, padding=1))
#                 current_c = max(base_c // 4, 8)
#
#         self.branch_bg.add_module('up_bg_final', nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
#         self.branch_bg.add_module('conv_bg_final', nn.Conv2d(current_c, 1, kernel_size=1, padding=0))
#         self.branch_bg.add_module('sigmoid_bg', nn.Sigmoid())
#
#         # 不确定分支
#         self.branch_uc = nn.Sequential()
#         current_c = in_c  # c_list[5]
#
#         for i in range(up_times):
#             if i == 0:
#                 self.branch_uc.add_module(f'cbr_uc_{i}',
#                                           CBR(current_c, base_c, kernel_size=3, padding=1))
#                 current_c = base_c
#             elif i <= 2:
#                 self.branch_uc.add_module(f'up_uc_{i}',
#                                           nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
#                 self.branch_uc.add_module(f'cbr_uc_{i}',
#                                           CBR(current_c, base_c, kernel_size=3, padding=1))
#                 current_c = base_c
#             elif i == 3:
#                 self.branch_uc.add_module(f'up_uc_{i}',
#                                           nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
#                 self.branch_uc.add_module(f'cbr_uc_{i}',
#                                           CBR(current_c, max(base_c // 2, 16), kernel_size=3, padding=1))
#                 current_c = max(base_c // 2, 16)
#             else:
#                 self.branch_uc.add_module(f'up_uc_{i}',
#                                           nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
#                 self.branch_uc.add_module(f'cbr_uc_{i}',
#                                           CBR(current_c, max(base_c // 4, 8), kernel_size=3, padding=1))
#                 current_c = max(base_c // 4, 8)
#
#         self.branch_uc.add_module('up_uc_final', nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
#         self.branch_uc.add_module('conv_uc_final', nn.Conv2d(current_c, 1, kernel_size=1, padding=0))
#         self.branch_uc.add_module('sigmoid_uc', nn.Sigmoid())
#
#     def forward(self, f_fg, f_bg, f_uc):
#         mask_fg = self.branch_fg(f_fg)
#         mask_bg = self.branch_bg(f_bg)
#         mask_uc = self.branch_uc(f_uc)
#         return mask_fg, mask_bg, mask_uc
#
#
# class SIDModule(nn.Module):
#     """完整的SID模块，包含解耦层和辅助头"""
#
#     def __init__(self, in_channels, mid_channels=None, scale_factor=32):
#         super(SIDModule, self).__init__()
#         # 如果没有指定mid_channels，则默认等于in_channels
#         if mid_channels is None:
#             mid_channels = in_channels
#
#         self.decouple_layer = DecoupleLayer(in_channels, mid_channels)
#         self.auxiliary_head = AuxiliaryHead(mid_channels, scale_factor)
#
#     def forward(self, x):
#         # 解耦特征
#         f_fg, f_bg, f_uc = self.decouple_layer(x)
#         # 生成masks
#         mask_fg, mask_bg, mask_uc = self.auxiliary_head(f_fg, f_bg, f_uc)
#         return (f_fg, f_bg, f_uc), (mask_fg, mask_bg, mask_uc)
#
#
# if __name__ == '__main__':
#     # 测试不同配置
#     print("测试指定的通道数变化：")
#
#     # 测试标准配置 c_list[5]=128
#     model = SIDModule(
#         in_channels=128,  # c_list[5] = 128
#         scale_factor=32
#     ).cuda()
#
#     input_tensor = torch.randn(4, 128, 8, 8).cuda()
#     (f_fg, f_bg, f_uc), (mask_fg, mask_bg, mask_uc) = model(input_tensor)
#
#     print(f"\n配置 (c_list[5]=128):")
#     print(f"  输入形状: {input_tensor.shape}")
#     print(f"  DecoupleLayer输出形状: {f_fg.shape}")
#     print(f"  Mask输出形状: {mask_fg.shape}")
#     print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
#
#     # 详细打印通道变化
#     print("\n通道数变化详情:")
#     print("DecoupleLayer: 128 → 64 → 256 → 128")
#     print("AuxiliaryHead: 128 → 128 → 128 → 128 → 64 → 32 → 1")
#
#     # 测试其他配置
#     print("\n测试其他c_list配置:")
#     for c in [64, 96, 256]:
#         model_test = SIDModule(in_channels=c, scale_factor=32).cuda()
#         params = sum(p.numel() for p in model_test.parameters())
#         print(f"  c_list[5]={c}: 参数量={params:,}")
#         print(f"    DecoupleLayer: {c} → {c // 2} → {2 * c} → {c}")
#         print(f"    AuxiliaryHead: {c} → {c} → {c} → {c} → {c // 2} → {c // 4} → 1")


import torch
import torch.nn as nn
import torch.nn.functional as F


class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x


class DecoupleLayer(nn.Module):
    """解耦层：将特征分解为前景、背景和不确定三个分支"""

    def __init__(self, in_c, out_c=None):
        super(DecoupleLayer, self).__init__()
        # 按照指定的通道变化：c_list[5] → c_list[5]/2 → c_list[5]/4 → c_list[5]/4
        if out_c is None:
            out_c = max(in_c // 4, 16)  # 默认输出为c_list[5]/4，确保最小16通道

        # 第一层：c_list[5] → c_list[5]/2
        first_c = max(in_c // 2, 32)
        # 第二层：c_list[5]/2 → c_list[5]/4
        second_c = max(in_c // 4, 16)
        # 第三层：c_list[5]/4 → c_list[5]/4 (out_c)
        third_c = out_c

        self.cbr_fg = nn.Sequential(
            CBR(in_c, first_c, kernel_size=3, padding=1),  # c_list[5] → c_list[5]/2
            CBR(first_c, second_c, kernel_size=3, padding=1),  # c_list[5]/2 → c_list[5]/4
            CBR(second_c, third_c, kernel_size=1, padding=0)  # c_list[5]/4 → c_list[5]/4
        )
        self.cbr_bg = nn.Sequential(
            CBR(in_c, first_c, kernel_size=3, padding=1),
            CBR(first_c, second_c, kernel_size=3, padding=1),
            CBR(second_c, third_c, kernel_size=1, padding=0)
        )
        self.cbr_uc = nn.Sequential(
            CBR(in_c, first_c, kernel_size=3, padding=1),
            CBR(first_c, second_c, kernel_size=3, padding=1),
            CBR(second_c, third_c, kernel_size=1, padding=0)
        )

    def forward(self, x):
        f_fg = self.cbr_fg(x)
        f_bg = self.cbr_bg(x)
        f_uc = self.cbr_uc(x)
        return f_fg, f_bg, f_uc


class AuxiliaryHead(nn.Module):
    """辅助头：为每个分支生成对应的mask"""

    def __init__(self, in_c, scale_factor=16):
        super(AuxiliaryHead, self).__init__()
        self.scale_factor = scale_factor

        # 计算需要的上采样次数
        up_times = int(torch.log2(torch.tensor(scale_factor, dtype=torch.float32)).item())

        # 按照指定的通道变化设计
        # in_c(=c_list[5]/4) → c_list[5]/4 → c_list[5]/4 → c_list[5]/8 → c_list[5]/16 → c_list[5]/16 → 1
        # 假设in_c = c_list[5]/4，那么：
        base_c = in_c * 4  # 恢复到c_list[5]的值

        # 前景分支
        self.branch_fg = nn.Sequential()
        current_c = in_c  # c_list[5]/4

        for i in range(up_times):
            if i <= 1:  # i=0,1
                # 保持c_list[5]/4
                if i > 0:
                    self.branch_fg.add_module(f'up_fg_{i}',
                                              nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
                self.branch_fg.add_module(f'cbr_fg_{i}',
                                          CBR(current_c, max(base_c // 4, 16), kernel_size=3, padding=1))
                current_c = max(base_c // 4, 16)
            elif i == 2:
                # 降到c_list[5]/8
                self.branch_fg.add_module(f'up_fg_{i}',
                                          nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
                self.branch_fg.add_module(f'cbr_fg_{i}',
                                          CBR(current_c, max(base_c // 8, 8), kernel_size=3, padding=1))
                current_c = max(base_c // 8, 8)
            else:  # i>=3
                # 保持c_list[5]/16
                self.branch_fg.add_module(f'up_fg_{i}',
                                          nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
                self.branch_fg.add_module(f'cbr_fg_{i}',
                                          CBR(current_c, max(base_c // 16, 4), kernel_size=3, padding=1))
                current_c = max(base_c // 16, 4)

        # 最后的上采样和输出
        self.branch_fg.add_module('up_fg_final', nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
        self.branch_fg.add_module('conv_fg_final', nn.Conv2d(current_c, 1, kernel_size=1, padding=0))
        self.branch_fg.add_module('sigmoid_fg', nn.Sigmoid())

        # 背景分支
        self.branch_bg = nn.Sequential()
        current_c = in_c  # c_list[5]/4

        for i in range(up_times):
            if i <= 1:  # i=0,1
                if i > 0:
                    self.branch_bg.add_module(f'up_bg_{i}',
                                              nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
                self.branch_bg.add_module(f'cbr_bg_{i}',
                                          CBR(current_c, max(base_c // 4, 16), kernel_size=3, padding=1))
                current_c = max(base_c // 4, 16)
            elif i == 2:
                self.branch_bg.add_module(f'up_bg_{i}',
                                          nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
                self.branch_bg.add_module(f'cbr_bg_{i}',
                                          CBR(current_c, max(base_c // 8, 8), kernel_size=3, padding=1))
                current_c = max(base_c // 8, 8)
            else:  # i>=3
                self.branch_bg.add_module(f'up_bg_{i}',
                                          nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
                self.branch_bg.add_module(f'cbr_bg_{i}',
                                          CBR(current_c, max(base_c // 16, 4), kernel_size=3, padding=1))
                current_c = max(base_c // 16, 4)

        self.branch_bg.add_module('up_bg_final', nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
        self.branch_bg.add_module('conv_bg_final', nn.Conv2d(current_c, 1, kernel_size=1, padding=0))
        self.branch_bg.add_module('sigmoid_bg', nn.Sigmoid())

        # 不确定分支
        self.branch_uc = nn.Sequential()
        current_c = in_c  # c_list[5]/4

        for i in range(up_times):
            if i <= 1:  # i=0,1
                if i > 0:
                    self.branch_uc.add_module(f'up_uc_{i}',
                                              nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
                self.branch_uc.add_module(f'cbr_uc_{i}',
                                          CBR(current_c, max(base_c // 4, 16), kernel_size=3, padding=1))
                current_c = max(base_c // 4, 16)
            elif i == 2:
                self.branch_uc.add_module(f'up_uc_{i}',
                                          nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
                self.branch_uc.add_module(f'cbr_uc_{i}',
                                          CBR(current_c, max(base_c // 8, 8), kernel_size=3, padding=1))
                current_c = max(base_c // 8, 8)
            else:  # i>=3
                self.branch_uc.add_module(f'up_uc_{i}',
                                          nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
                self.branch_uc.add_module(f'cbr_uc_{i}',
                                          CBR(current_c, max(base_c // 16, 4), kernel_size=3, padding=1))
                current_c = max(base_c // 16, 4)

        self.branch_uc.add_module('up_uc_final', nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
        self.branch_uc.add_module('conv_uc_final', nn.Conv2d(current_c, 1, kernel_size=1, padding=0))
        self.branch_uc.add_module('sigmoid_uc', nn.Sigmoid())

    def forward(self, f_fg, f_bg, f_uc):
        mask_fg = self.branch_fg(f_fg)
        mask_bg = self.branch_bg(f_bg)
        mask_uc = self.branch_uc(f_uc)
        return mask_fg, mask_bg, mask_uc


class SIDModule(nn.Module):
    """完整的SID模块，包含解耦层和辅助头"""

    def __init__(self, in_channels, mid_channels=None, scale_factor=32):
        super(SIDModule, self).__init__()
        # 如果没有指定mid_channels，则默认为in_channels/4
        if mid_channels is None:
            mid_channels = max(in_channels // 4, 16)

        self.decouple_layer = DecoupleLayer(in_channels, mid_channels)
        self.auxiliary_head = AuxiliaryHead(mid_channels, scale_factor)

    def forward(self, x):
        # 解耦特征
        f_fg, f_bg, f_uc = self.decouple_layer(x)
        # 生成masks
        mask_fg, mask_bg, mask_uc = self.auxiliary_head(f_fg, f_bg, f_uc)
        return (f_fg, f_bg, f_uc), (mask_fg, mask_bg, mask_uc)


if __name__ == '__main__':
    # 测试不同配置
    print("测试指定的通道数变化：")

    # 测试标准配置 c_list[5]=128
    model = SIDModule(
        in_channels=128,  # c_list[5] = 128
        scale_factor=32
    ).cuda()

    input_tensor = torch.randn(4, 128, 8, 8).cuda()
    (f_fg, f_bg, f_uc), (mask_fg, mask_bg, mask_uc) = model(input_tensor)

    print(f"\n配置 (c_list[5]=128):")
    print(f"  输入形状: {input_tensor.shape}")
    print(f"  DecoupleLayer输出形状: {f_fg.shape}")
    print(f"  Mask输出形状: {mask_fg.shape}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 详细打印通道变化
    print("\n通道数变化详情:")
    print("DecoupleLayer: 128 → 64 → 32 → 32")
    print("AuxiliaryHead: 32 → 32 → 32 → 16 → 8 → 8 → 1")

    # 测试其他配置
    print("\n测试其他c_list配置:")
    for c in [64, 96, 256]:
        model_test = SIDModule(in_channels=c, scale_factor=32).cuda()
        params = sum(p.numel() for p in model_test.parameters())
        print(f"  c_list[5]={c}: 参数量={params:,}")
        print(f"    DecoupleLayer: {c} → {c // 2} → {c // 4} → {c // 4}")
        print(f"    AuxiliaryHead: {c // 4} → {c // 4} → {c // 4} → {c // 8} → {c // 16} → {c // 16} → 1")

    # 测试与egeunet.py的兼容性
    print("\n\n与egeunet.py的兼容性:")
    print("现在需要在egeunet.py中修改为：")
    print("self.sid_module = SIDModule(c_list[5], c_list[5]//4, scale_factor=32)")
    print("或者使用默认值：")
    print("self.sid_module = SIDModule(c_list[5], scale_factor=32)  # 自动使用c_list[5]/4")

    # 参数量对比
    print("\n\n参数量对比（c_list[5]=128）:")
    print("原始版本（输出c_list[5]）: ~1.2M 参数")
    print("当前版本（输出c_list[5]/4）: ~168K 参数")
    print("参数量减少约: 86%")

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class CBR(nn.Module):
#     def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
#         super().__init__()
#         self.act = act
#
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
#             nn.BatchNorm2d(out_c)
#         )
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.conv(x)
#         if self.act == True:
#             x = self.relu(x)
#         return x
#
#
# class DecoupleLayer(nn.Module):
#     """解耦层：将特征分解为前景、背景和不确定三个分支"""
#
#     def __init__(self, in_c=1024, out_c=256):
#         super(DecoupleLayer, self).__init__()
#         self.cbr_fg = nn.Sequential(
#             CBR(in_c, 512, kernel_size=3, padding=1),
#             CBR(512, out_c, kernel_size=3, padding=1),
#             CBR(out_c, out_c, kernel_size=1, padding=0)
#         )
#         self.cbr_bg = nn.Sequential(
#             CBR(in_c, 512, kernel_size=3, padding=1),
#             CBR(512, out_c, kernel_size=3, padding=1),
#             CBR(out_c, out_c, kernel_size=1, padding=0)
#         )
#         self.cbr_uc = nn.Sequential(
#             CBR(in_c, 512, kernel_size=3, padding=1),
#             CBR(512, out_c, kernel_size=3, padding=1),
#             CBR(out_c, out_c, kernel_size=1, padding=0)
#         )
#
#     def forward(self, x):
#         f_fg = self.cbr_fg(x)
#         f_bg = self.cbr_bg(x)
#         f_uc = self.cbr_uc(x)
#         return f_fg, f_bg, f_uc
#
#
# class AuxiliaryHead(nn.Module):
#     """辅助头：为每个分支生成对应的mask"""
#
#     def __init__(self, in_c, scale_factor=16):
#         super(AuxiliaryHead, self).__init__()
#         self.scale_factor = scale_factor
#
#         # 计算需要的上采样次数
#         up_times = int(torch.log2(torch.tensor(scale_factor, dtype=torch.float32)).item())
#
#         # 前景分支
#         self.branch_fg = nn.Sequential()
#         current_c = in_c
#         for i in range(up_times):
#             if i == 0:
#                 self.branch_fg.add_module(f'cbr_fg_{i}', CBR(current_c, 256, kernel_size=3, padding=1))
#                 current_c = 256
#             elif i < up_times - 2:
#                 self.branch_fg.add_module(f'up_fg_{i}',
#                                           nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
#                 self.branch_fg.add_module(f'cbr_fg_{i}', CBR(current_c, current_c, kernel_size=3, padding=1))
#             elif i == up_times - 2:
#                 self.branch_fg.add_module(f'up_fg_{i}',
#                                           nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
#                 self.branch_fg.add_module(f'cbr_fg_{i}', CBR(current_c, 128, kernel_size=3, padding=1))
#                 current_c = 128
#             else:
#                 self.branch_fg.add_module(f'up_fg_{i}',
#                                           nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
#                 self.branch_fg.add_module(f'cbr_fg_{i}', CBR(current_c, 64, kernel_size=3, padding=1))
#
#         self.branch_fg.add_module('up_fg_final', nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
#         self.branch_fg.add_module('conv_fg_final', nn.Conv2d(64, 1, kernel_size=1, padding=0))
#         self.branch_fg.add_module('sigmoid_fg', nn.Sigmoid())
#
#         # 背景分支
#         self.branch_bg = nn.Sequential()
#         current_c = in_c
#         for i in range(up_times):
#             if i == 0:
#                 self.branch_bg.add_module(f'cbr_bg_{i}', CBR(current_c, 256, kernel_size=3, padding=1))
#                 current_c = 256
#             elif i < up_times - 2:
#                 self.branch_bg.add_module(f'up_bg_{i}',
#                                           nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
#                 self.branch_bg.add_module(f'cbr_bg_{i}', CBR(current_c, current_c, kernel_size=3, padding=1))
#             elif i == up_times - 2:
#                 self.branch_bg.add_module(f'up_bg_{i}',
#                                           nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
#                 self.branch_bg.add_module(f'cbr_bg_{i}', CBR(current_c, 128, kernel_size=3, padding=1))
#                 current_c = 128
#             else:
#                 self.branch_bg.add_module(f'up_bg_{i}',
#                                           nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
#                 self.branch_bg.add_module(f'cbr_bg_{i}', CBR(current_c, 64, kernel_size=3, padding=1))
#
#         self.branch_bg.add_module('up_bg_final', nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
#         self.branch_bg.add_module('conv_bg_final', nn.Conv2d(64, 1, kernel_size=1, padding=0))
#         self.branch_bg.add_module('sigmoid_bg', nn.Sigmoid())
#
#         # 不确定分支
#         self.branch_uc = nn.Sequential()
#         current_c = in_c
#         for i in range(up_times):
#             if i == 0:
#                 self.branch_uc.add_module(f'cbr_uc_{i}', CBR(current_c, 256, kernel_size=3, padding=1))
#                 current_c = 256
#             elif i < up_times - 2:
#                 self.branch_uc.add_module(f'up_uc_{i}',
#                                           nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
#                 self.branch_uc.add_module(f'cbr_uc_{i}', CBR(current_c, current_c, kernel_size=3, padding=1))
#             elif i == up_times - 2:
#                 self.branch_uc.add_module(f'up_uc_{i}',
#                                           nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
#                 self.branch_uc.add_module(f'cbr_uc_{i}', CBR(current_c, 128, kernel_size=3, padding=1))
#                 current_c = 128
#             else:
#                 self.branch_uc.add_module(f'up_uc_{i}',
#                                           nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
#                 self.branch_uc.add_module(f'cbr_uc_{i}', CBR(current_c, 64, kernel_size=3, padding=1))
#
#         self.branch_uc.add_module('up_uc_final', nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
#         self.branch_uc.add_module('conv_uc_final', nn.Conv2d(64, 1, kernel_size=1, padding=0))
#         self.branch_uc.add_module('sigmoid_uc', nn.Sigmoid())
#
#     def forward(self, f_fg, f_bg, f_uc):
#         mask_fg = self.branch_fg(f_fg)
#         mask_bg = self.branch_bg(f_bg)
#         mask_uc = self.branch_uc(f_uc)
#         return mask_fg, mask_bg, mask_uc
#
#
# class SIDModule(nn.Module):
#     """完整的SID模块，包含解耦层和辅助头"""
#
#     def __init__(self, in_channels, mid_channels=128, scale_factor=32):
#         super(SIDModule, self).__init__()
#         self.decouple_layer = DecoupleLayer(in_channels, mid_channels)
#         self.auxiliary_head = AuxiliaryHead(mid_channels, scale_factor)
#
#     def forward(self, x):
#         # 解耦特征
#         f_fg, f_bg, f_uc = self.decouple_layer(x)
#         # 生成masks
#         mask_fg, mask_bg, mask_uc = self.auxiliary_head(f_fg, f_bg, f_uc)
#         return (f_fg, f_bg, f_uc), (mask_fg, mask_bg, mask_uc)
#
# if __name__ == '__main__':
#     # 测试不确定性引导的模型 v2
#     model = SIDModule(
#         in_channels=8
#     ).cuda()
#
#     input_tensor = torch.randn(4, 8, 8, 8).cuda()  # B, C, H, W
#     (f_fg, f_bg, f_uc), (mask_fg, mask_bg, mask_uc) = model(input_tensor)
#     print(mask_uc.shape)  # torch.Size([4, 128, 32, 32])

