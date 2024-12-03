import torch
import torch.nn as nn
import torch.nn.functional as F


# SE Block (Squeeze-and-Excitation) 通道注意力模块
# 核心思想在于通过网络根据loss去学习特征权重，使得有效的feature map权重大，无效或效果小的feature map权重小的方式训练模型达到更好的结果。
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        # 定义第一个全连接层 压缩通道数
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        # 定义第二个全连接层 恢复通道数
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x):
        batch_size, channel, _, _ = x.size()
        # 通过全局平均池化、两个全连接层计算通道权重
        # 通过全局平均池化 将每个通道的空间区域压缩成标量
        se_weight = F.adaptive_avg_pool2d(x, 1).view(batch_size, channel)
        se_weight = F.relu(self.fc1(se_weight))
        se_weight = torch.sigmoid(self.fc2(se_weight)).view(batch_size, channel, 1, 1)
        #返回加权输入特征图
        return x * se_weight


class IdentityPadding(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(IdentityPadding, self).__init__()
        # 步幅为2，使用2x2的平均池化层来减小空间尺寸
        if stride == 2:
            self.pooling = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)
        else:
            # 步幅为1，不使用池化
            self.pooling = None
        # 输入和输出通道数之间的差异
        self.add_channels = out_channels - in_channels

    def forward(self, x):
        # 对输入进行填充，使得输入的通道数和输出的通道数一致
        out = F.pad(x, (0, 0, 0, 0, 0, self.add_channels))
        # 如果步幅为2，使用池化减小空间尺寸
        if self.pooling is not None:
            out = self.pooling(out)
        return out


# 改进后的残差块 包含 SE Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # 残差块的卷积操作
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # 定义通道注意力
        self.se_block = SEBlock(out_channels)
        # 对齐输入和输出特征图尺寸
        self.down_sample = IdentityPadding(in_channels, out_channels, stride)
        self.stride = stride

    def forward(self, x):
        shortcut = self.down_sample(x)
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        # 通道注意力
        out = self.se_block(out)
        # 残差连接
        out += shortcut
        return out


# PyramidNet 网络结构
class PyramidNet(nn.Module):
    def __init__(self, num_layers, alpha, block, num_classes=10):
        super(PyramidNet, self).__init__()
        # 初始化网络的相关参数
        self.in_channels = 16
        self.num_layers = num_layers
        self.addrate = alpha / (3 * self.num_layers * 1.0)
        # 定义卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        # 定义残差块层
        self.layer1 = self.get_layers(block, stride=1)
        self.layer2 = self.get_layers(block, stride=2)
        self.layer3 = self.get_layers(block, stride=2)
        # 计算网络的最终输出通道
        self.out_channels = int(round(self.out_channels))
        # 定义最后输出处理的操作
        self.bn_out = nn.BatchNorm2d(self.out_channels)
        self.relu_out = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        # 定义全连接层
        self.fc_out = nn.Linear(self.out_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用 Kaiming 正态初始化 来初始化卷积层的权重
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                #　批归一化层初始化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # 根据网络的深度，返回残差块
    def get_layers(self, block, stride):
        layers_list = []
        for _ in range(self.num_layers - 1):
            self.out_channels = self.in_channels + self.addrate
            layers_list.append(block(int(round(self.in_channels)),
                                     int(round(self.out_channels)),
                                     stride))
            self.in_channels = self.out_channels
            stride = 1

        return nn.Sequential(*layers_list)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn_out(x)
        x = self.relu_out(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_out(x)
        return x


# PyramidNet 初始化函数
def pyramidnet():
    block = ResidualBlock
    model = PyramidNet(num_layers=18, alpha=48, block=block)
    return model

