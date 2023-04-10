import torch
from torch import nn
from torch import Tensor

from torchvision.transforms.functional import pil_to_tensor, resize
from PIL import Image


class FusedMBConv(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, expansion: int = 4):
        super(FusedMBConv, self).__init__()
        exp_features = in_features*expansion

        self.relu6 = nn.ReLU6(inplace=True)
        self.conv1 = nn.Conv2d(in_features, exp_features, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(exp_features)
        self.conv2 = nn.Conv2d(exp_features, out_features, kernel_size=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu6(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out
    

class FPN(nn.Module):
    def __init__(self, in_features, depth=5, bu_dim=16, td_dim=256):
        super(FPN, self).__init__()
        self.in_features = in_features
        self.depth = depth
        self.bu_dim = bu_dim
        self.td_dim = td_dim 


    def bottom_up(self, img: Tensor) -> list[Tensor]:
        outputs = []

        bu_in_features = self.in_features
        
        for _ in range(self.depth):
            bu_conv_module = nn.Sequential(
            FusedMBConv(bu_in_features, bu_in_features),
            nn.Conv2d(in_channels=bu_in_features, out_channels=self.bu_dim, kernel_size=(2, 2), stride=(2, 2), bias=False),
            nn.BatchNorm2d(self.bu_dim)
        )

            img = bu_conv_module(img)
            outputs.append(img)
            bu_in_features = self.bu_dim
            self.bu_dim *= 2
        
        return outputs[1:]
    

    def top_down(self, bu_outputs):
        bu_outputs = bu_outputs[::-1]
        td_inputs = []
        for bu_out in bu_outputs:
            m = nn.Conv2d(bu_out.shape[1], self.td_dim, kernel_size=(1, 1))(bu_out)
            td_inputs.append(m)
        
        td_outputs = [td_inputs[0]]
        for i in range(1, len(td_inputs)):
            m_prev = nn.Upsample(scale_factor=2, mode='nearest')(td_inputs[i-1])
            p = nn.Conv2d(self.td_dim, self.td_dim, kernel_size=(3, 3), padding=(1, 1), bias=False)(m_prev+td_inputs[i])
            td_outputs.append(p)

        return td_outputs


    def forward(self, x):
        bu_out = self.bottom_up(x)
        td_out = self.top_down(bu_out)

        return td_out


class RPN_with_FPN(nn.Module):
    def __init__(self, in_features, fpn_depth, fpn_bu_dim, fpn_td_dim, anchors_num=9):
        super(RPN_with_FPN, self).__init__()
        
        self.fpn = FPN(in_features, fpn_depth, fpn_bu_dim, fpn_td_dim)

        self.conv1 = nn.Conv2(fpn_td_dim, fpn_td_dim, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(fpn_td_dim)

        self.cls = nn.Conv2d(fpn_td_dim, anchors_num*2, kernel_size=(1, 1), bias=False)
        self.reg = nn.Conv2d(fpn_td_dim, anchors_num*4, kernel_size=(1, 1), bias=False)
        

    def forward(self, x):
        pass


class FasterRCNN(nn.Module):
    def __init__(self):
        super(FasterRCNN, self).__init__()