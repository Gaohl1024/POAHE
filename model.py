import torch.nn as nn
import torch
from utils.model_util import feature_enhance, jigsaw_generator, feature_restrain


class FCA(nn.Module):
    def __init__(self, model, feature_size, num_ftrs, classes_num):
        super(FCA, self).__init__()
        self.backbone = model
        self.num_ftrs = num_ftrs
        self.im_sz = 448
        self.drop = 0.5

        # con1
        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs // 4, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True),
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(self.drop),
        )
        self.Linear1 = nn.Linear(feature_size, classes_num)

        # con2
        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs // 2, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True),
            nn.AdaptiveMaxPool2d(1)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(self.drop),
        )
        self.Linear2 = nn.Linear(feature_size, classes_num)

        # con3
        self.conv_block3 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True),
            nn.AdaptiveMaxPool2d(1)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(self.drop),
        )
        self.Linear3 = nn.Linear(feature_size, classes_num)

        # conC
        self.features_concat = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2 * 3),
            nn.Linear(self.num_ftrs // 2 * 3, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
        )
        self.classifier_concat = nn.Linear(feature_size, classes_num)

    def forward(self, x, hype_par, is_train=False, is_temp=False):
        y1, y2, y3, y4, f1, f2, f3, f4, y1e, y2e, y3e, y4e, f1e, f2e, f3e, f4e, f1a = (None,) * 17
        batch = x.shape[0]

        _, _, f1, f2, f3 = self.backbone(x)
        f1 = self.conv_block1(f1)
        f1a = f1.clone()
        f1 = nn.AdaptiveMaxPool2d(1)(f1).view(batch, -1)
        f2 = self.conv_block2(f2).view(batch, -1)
        f3 = self.conv_block3(f3).view(batch, -1)
        f4 = self.features_concat(torch.cat((f1, f2, f3), -1))
        f1r = f1.clone()

        # cls
        f1 = self.classifier1(f1)
        f2 = self.classifier2(f2)
        f3 = self.classifier3(f3)

        y1 = self.Linear1(f1)
        y2 = self.Linear2(f2)
        y3 = self.Linear3(f3)
        y4 = self.classifier_concat(f4)

        if is_train:
            _, p1 = torch.max(y1.data, 1)
            if is_temp:
                f1r = feature_restrain(f1r, p1)
                f1r = self.classifier1(f1r)
                f1e = feature_enhance(f1r, p1, self.Linear1.weight, hype_par)
            else:
                f1e = feature_enhance(f1, p1, self.Linear1.weight, hype_par)
            y1e = self.Linear1(f1e)
            _, p2 = torch.max(y2.data, 1)
            f2e = feature_enhance(f2, p2, self.Linear2.weight, hype_par)
            y2e = self.Linear2(f2e)
            _, p3 = torch.max(y3.data, 1)
            f3e = feature_enhance(f3, p3, self.Linear3.weight, hype_par)
            y3e = self.Linear3(f3e)

        _, p4 = torch.max(y4.data, 1)
        f4e = feature_enhance(f4, p4, self.classifier_concat.weight, hype_par)
        y4e = self.classifier_concat(f4e)

        return y1, y2, y3, y4, f1, f2, f3, f4, y1e, y2e, y3e, y4e, f1e, f2e, f3e, f4e, f1a


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
