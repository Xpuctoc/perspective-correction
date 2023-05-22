import torch
import torch.nn as nn
import timm
from torchvision import models
from base import BaseModel
from tiny_vit import tiny_vit_21m_512


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 conv_kernel_size, conv_stride, conv_padding,
                 pool_kernel_size, pool_stride, pool_padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.pool(x)
        x = self.relu(x)
        return x


class HomographyRegressor(BaseModel):
    def __init__(self):
        super().__init__()
        # pretrained_model = models.densenet161(pretrained=True)
        # for param in pretrained_model.parameters():
        #     param.requires_grad = False
        #
        # self.pretrained_bridge = nn.Sequential(*list(pretrained_model.children())[:-2])
        #
        # self.conv1 = ConvBlock(3, 64, 3, 2, 1, 3, 1, 0)
        # self.conv2 = ConvBlock(64, 64, 3, 2, 1, 3, 1, 0)
        # self.conv3 = ConvBlock(64, 64, 3, 2, 1, 3, 1, 0)
        # self.conv4 = ConvBlock(64, 64, 3, 2, 1, 3, 1, 0)
        # self.conv5 = ConvBlock(64, 64, 3, 2, 1, 3, 1, 0)
        # self.conv6 = ConvBlock(64, 64, 3, 2, 1, 2, 1, 0)
        # self.conv7 = ConvBlock(64, 64, 3, 1, 1, 2, 1, 0)
        # self.conv8 = ConvBlock(64, 64, 3, 1, 1, 2, 1, 0)
        # self.conv9 = ConvBlock(64, 64, 3, 1, 1, 2, 1, 0)
        #
        # self.fc = nn.Sequential(
        #     nn.Linear(576, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 9)
        # )

        # ################################################################################
        # self.model = timm.create_model('tf_efficientnetv2_s_in21ft1k', pretrained=True)
        # # for param in self.model.parameters():
        # #     param.requires_grad = False
        # #
        # # for param in self.model.blocks[5].parameters():
        # #     param.requires_grad = True
        # #
        # # for param in self.model.conv_head.parameters():
        # #     param.requires_grad = True
        # #
        # # for param in self.model.bn2.parameters():
        # #     param.requires_grad = True
        # #
        # # for param in self.model.global_pool.parameters():
        # #     param.requires_grad = True
        #
        # self.model.classifier = nn.Sequential(
        #     nn.Linear(in_features=1280, out_features=512, bias=True),
        #     nn.SiLU(inplace=True),
        #     nn.Linear(in_features=512, out_features=9, bias=True)
        # )
        # ################################################################################

        ################################################################################
        # self.model = tiny_vit_21m_512(pretrained=True)
        # self.model.head = nn.Linear(in_features=576, out_features=9, bias=True)
        #
        # for param in self.model.parameters():
        #     param.requires_grad = False
        #
        # for param in self.model.layers[-1].parameters():
        #     param.requires_grad = True
        #
        # for param in self.model.layers[-2].parameters():
        #     param.requires_grad = True
        #
        # for param in self.model.norm_head.parameters():
        #     param.requires_grad = True
        #
        # for param in self.model.head.parameters():
        #     param.requires_grad = True
        ################################################################################

        # self.model = timm.create_model('convnextv2_nano.fcmae_ft_in22k_in1k_384', pretrained=True)
        # self.model.head.fc = nn.Linear(640, 9, bias=True)

        # self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        #
        # for param in self.model.parameters():
        #     param.requires_grad = False
        #
        # for i in [6, 7, 8, 9, 10, 11]:
        #     for param in self.model.blocks[i].parameters():
        #         param.requires_grad = True
        #
        # for param in self.model.norm.parameters():
        #     param.requires_grad = True
        #
        # self.model.head = nn.Sequential(
        #     nn.Linear(384, 512, bias=True),
        #     nn.GELU(approximate='none'),
        #     nn.Linear(512, 9, bias=True)
        # )

    def forward(self, x):
        # x = self.pretrained_bridge(x)
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        # x = self.conv6(x)
        # x = self.conv7(x)
        # x = self.conv8(x)
        # x = self.conv9(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        x = self.model(x)  # for effnetv2s, tinyViT, ...

        return x


class tinyViT(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = tiny_vit_21m_512(pretrained=True)  # requires timm==0.4.2
        self.model.head = nn.Linear(in_features=576, out_features=9, bias=True)

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layers[-1].parameters():
            param.requires_grad = True

        for param in self.model.layers[-2].parameters():
            param.requires_grad = True

        for param in self.model.norm_head.parameters():
            param.requires_grad = True

        for param in self.model.head.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x


class effnetv2s(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('tf_efficientnetv2_s_in21ft1k', pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=1280, out_features=512, bias=True),
            nn.SiLU(inplace=True),
            nn.Linear(in_features=512, out_features=9, bias=True)
        )

    def forward(self, x):
        x = self.model(x)
        return x


class dinov2(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        for param in self.model.parameters():
            param.requires_grad = False

        # for i in [6, 7, 8, 9, 10, 11]:
        #     for param in self.model.blocks[i].parameters():
        #         param.requires_grad = True

        for param in self.model.norm.parameters():
            param.requires_grad = True

        self.model.head = nn.Sequential(
            nn.Linear(384, 512, bias=True),
            nn.GELU(approximate='none'),
            nn.Linear(512, 512, bias=True),
            nn.GELU(approximate='none'),
            nn.Linear(512, 9, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x

