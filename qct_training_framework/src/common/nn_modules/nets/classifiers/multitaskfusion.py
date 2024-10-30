import torch
import torch.nn as nn
import torch.nn.functional as F

class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y

class SEResNeXtBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=32, reduction=16):
        super(SEResNeXtBottleneck, self).__init__()
        mid_channels = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(out_channels, reduction=reduction)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)

        out = self.se_module(out)
        out += identity
        out = self.relu(out)
        return out

class SENetEncoder(nn.Module):
    def __init__(self):
        super(SENetEncoder, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        )
        self.layer1 = self._make_layer(SEResNeXtBottleneck, 64, 256, blocks=3, stride=1, groups=32, reduction=16)
        self.layer2 = self._make_layer(SEResNeXtBottleneck, 256, 512, blocks=4, stride=2, groups=32, reduction=16)
        self.layer3 = self._make_layer(SEResNeXtBottleneck, 512, 1024, blocks=6, stride=2, groups=32, reduction=16)
        self.layer4 = self._make_layer(SEResNeXtBottleneck, 1024, 2048, blocks=3, stride=2, groups=32, reduction=16)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride, groups, reduction):
        layers = []
        layers.append(block(in_channels, out_channels, stride=stride, groups=groups, reduction=reduction))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, stride=1, groups=groups, reduction=reduction))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.encoder = SENetEncoder()

    def forward(self, x):
        return self.encoder(x)

class AugmentationWindowing(nn.Module):
    def __init__(self):
        super(AugmentationWindowing, self).__init__()
        self.conv2d = nn.Conv2d(1, 3, kernel_size=1, stride=1)
        self.activation = nn.Hardtanh(min_val=-1.0, max_val=1.0)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.activation(x)
        return x

class MultiFC(nn.Module):
    def __init__(self, in_features, out_features):
        super(MultiFC, self).__init__()
        self.fc_0 = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x):
        return self.fc_0(x)

class MultiTaskFusionNet(nn.Module):
    def __init__(self):
        super(MultiTaskFusionNet, self).__init__()
        self.backbone = Unet()
        self.win_opt = AugmentationWindowing()
        self.multi_fc = MultiFC(2048, 2)

    def forward(self, x):
        x = self.win_opt(x)
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.view(x.size(0), -1)
        x = self.multi_fc(x)
        return x
