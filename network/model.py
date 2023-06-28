import torch
import torch.nn as nn
from torch.nn import functional as F
from AFusion import *

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, bath_normal=False):
        super(DoubleConv, self).__init__()
        channels = int(out_channels / 2)
        if in_channels > out_channels:
            channels = int(in_channels / 2)

        layers = [
            nn.Conv3d(in_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),

            nn.Conv3d(channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        ]
        if bath_normal:
            layers.insert(1, nn.BatchNorm3d(channels))
            layers.insert(len(layers) - 1, nn.BatchNorm3d(out_channels))

        # 构造序列器
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)

class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normal=False):
        super(DownSampling, self).__init__()
        self.maxpool_to_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, batch_normal)
        )

    def forward(self, x):
        return self.maxpool_to_conv(x)

class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normal=False, bilinear=True):
        super(UpSampling, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#        self.conv = DoubleConv(int(in_channels + in_channels / 2), out_channels, batch_normal)
        self.conv = DoubleConv(in_channels, out_channels, batch_normal)
    def forward(self, inputs1, inputs2):
        inputs1 = self.up(inputs1)
        outputs = torch.cat([inputs1, inputs2], dim=1)
        outputs = self.conv(outputs)
        return outputs

class LastConv(nn.Module):
    def __init__(self, in_channels, out_channels ):
        super(LastConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1 )

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels, num_classes=2, batch_normal=False, bilinear=True):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.batch_normal = batch_normal
        self.bilinear = bilinear

        self.inputs = DoubleConv(in_channels, 64, self.batch_normal)
        self.down_1 = DownSampling(64, 128, self.batch_normal)
        self.down_2 = DownSampling(128, 256, self.batch_normal)
        self.down_3 = DownSampling(256, 512, self.batch_normal)
        

        self.up_1 = UpSampling(512, 256, self.batch_normal, self.bilinear)
        self.up_2 = UpSampling(256, 128, self.batch_normal, self.bilinear)
        self.up_3 = UpSampling(128, 64, self.batch_normal, self.bilinear)
        self.outputs = LastConv(64, num_classes)

    def forward(self, x):
        x1 = self.inputs(x)       #64,64,64,64
        x2 = self.down_1(x1)      #128,64,64,64
        x3 = self.down_2(x2)      #256,16,16,16
        x4 = self.down_3(x3)      #512,8,8,8
        d_out = x4
        cls = x4
        x5 = self.up_1(x4, x3)    #256,16,16,16
        x6 = self.up_2(x5, x2)    #128,32,32,32
        x7 = self.up_3(x6, x1)    #64,64,64,64
        x = self.outputs(x7)      #4,64,64,64
        
        return x ,cls,d_out

class classifier(nn.Module):
    def __init__(self, in_channels, num_classes=3,batch_normal=True):
        super(classifier, self).__init__()
        self.in_channels = in_channels
        self.num_classes = 3
        self.batch_normal = batch_normal
        self.layer1 = nn.Sequential(
            DoubleConv(256, 512, self.batch_normal),
            DoubleConv(512, 256, self.batch_normal),
            DownSampling(256, 512, self.batch_normal),
        )
        self.layer2 = nn.Sequential(
            DownSampling(512, 1024, self.batch_normal),  # 1024,4,4,4
            nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
        )

        self.fc = nn.Linear(1024, num_classes, bias=True)
    def forward(self, x):
        cf_fp = self.layer1(x)
        cls = self.layer2(cf_fp)
        cls = self.fc(cls.squeeze())
        return cf_fp,cls

class prompt_UNet3D(nn.Module):
    def __init__(self, in_channels, num_classes=2, batch_normal=False, bilinear=True,prompt=True):
        super(prompt_UNet3D, self).__init__()
        self.in_channels = in_channels
        self.batch_normal = batch_normal
        self.bilinear = bilinear
        self.prompt = prompt
        self.inputs = DoubleConv(in_channels, 64, self.batch_normal)
        self.down_1 = DownSampling(64, 128, self.batch_normal)
        self.down_2 = DownSampling(128, 256, self.batch_normal)
        self.down_3 = DownSampling(256, 512, self.batch_normal)

        self.classifier = classifier(256,3)

        self.dane = AFusion(512)
        self.prompt_tf = nn.Sequential(
            nn.Conv3d(512, 512, 3, 1, 'same', bias=True),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, 3, 1, 'same', bias=True),
            nn.BatchNorm3d(512),
            nn.ReLU(),
        )

        self.up_1 = UpSampling(512, 256, self.batch_normal, self.bilinear)
        self.up_2 = UpSampling(256, 128, self.batch_normal, self.bilinear)
        self.up_3 = UpSampling(128, 64, self.batch_normal, self.bilinear)
        self.outputs = LastConv(64, num_classes)

    def forward(self, x):
        # down 部分
        x1 = self.inputs(x)  # 64,64,64,64
        x2 = self.down_1(x1)  # 128,64,64,64
        x3 = self.down_2(x2)  # 256,16,16,16
        cf_fp = x3
        x4 = self.down_3(x3)  # 512,8,8,8

        cf_fp,cls = self.classifier(cf_fp)     #512,8,8,8
        cf_fp1 = cf_fp
        cf_fp = self.prompt_tf(cf_fp)
        if self.prompt == True:
            d_out = self.dane(cf_fp,x4)           #true
        else:
            d_out = self.dane(x4,cf_fp)            #false
        # up部分
        x5 = self.up_1(d_out, x3)  # 256,16,16,16
        x6 = self.up_2(x5, x2)  # 128,32,32,32
        x7 = self.up_3(x6, x1)  # 64,64,64,64
        x = self.outputs(x7)  # 4,64,64,64

        return x, cls, d_out,cf_fp1,cf_fp,x4

class cls_UNet3D(nn.Module):
    def __init__(self, in_channels, num_classes=2, batch_normal=False, bilinear=True,prompt=True):
        super(cls_UNet3D, self).__init__()
        self.in_channels = in_channels
        self.batch_normal = batch_normal
        self.bilinear = bilinear
        self.prompt = prompt
        self.inputs = DoubleConv(in_channels, 64, self.batch_normal)
        self.down_1 = DownSampling(64, 128, self.batch_normal)
        self.down_2 = DownSampling(128, 256, self.batch_normal)
        self.down_3 = DownSampling(256, 512, self.batch_normal)

        self.classifier = classifier(256,3)

        self.dane = AFusion(512)
        self.prompt_tf = nn.Sequential(
            nn.Conv3d(512, 512, 3, 1, 'same', bias=True),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, 3, 1, 'same', bias=True),
            nn.BatchNorm3d(512),
            nn.ReLU(),
        )

        self.up_1 = UpSampling(512, 256, self.batch_normal, self.bilinear)
        self.up_2 = UpSampling(256, 128, self.batch_normal, self.bilinear)
        self.up_3 = UpSampling(128, 64, self.batch_normal, self.bilinear)
        self.outputs = LastConv(64, num_classes)

    def forward(self, x):
        # down 部分
        x1 = self.inputs(x)  # 64,64,64,64
        x2 = self.down_1(x1)  # 128,64,64,64
        x3 = self.down_2(x2)  # 256,16,16,16
        cf_fp = x3
        x4 = self.down_3(x3)  # 512,8,8,8

        cf_fp,cls = self.classifier(cf_fp)     #512,8,8,8

#        cf_fp = self.prompt_tf(cf_fp)
        if self.prompt == True:
            d_out = self.dane(cf_fp,x4)           #true
        else:
            d_out = self.dane(x4,cf_fp)            #false
        # up部分
        x5 = self.up_1(d_out, x3)  # 256,16,16,16
        x6 = self.up_2(x5, x2)  # 128,32,32,32
        x7 = self.up_3(x6, x1)  # 64,64,64,64
        x = self.outputs(x7)  # 4,64,64,64

        return x, cls, d_out

class prompt_UNet3D_add(nn.Module):
    def __init__(self, in_channels, num_classes=2, batch_normal=False, bilinear=True,prompt='add'):
        super(prompt_UNet3D_add, self).__init__()
        self.in_channels = in_channels
        self.batch_normal = batch_normal
        self.bilinear = bilinear
        self.prompt = prompt
        self.inputs = DoubleConv(in_channels, 64, self.batch_normal)
        self.down_1 = DownSampling(64, 128, self.batch_normal)
        self.down_2 = DownSampling(128, 256, self.batch_normal)
        self.down_3 = DownSampling(256, 512, self.batch_normal)

        self.classifier = classifier(256,3)
        self.prompt_tf = nn.Sequential(
            nn.Conv3d(512, 512, 3, 1, 'same', bias=True),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, 3, 1, 'same', bias=True),
            nn.BatchNorm3d(512),
            nn.ReLU(),
        )

        self.up_1 = UpSampling(512, 256, self.batch_normal, self.bilinear)
        self.up_2 = UpSampling(256, 128, self.batch_normal, self.bilinear)
        self.up_3 = UpSampling(128, 64, self.batch_normal, self.bilinear)
        self.outputs = LastConv(64, num_classes)

    def forward(self, x):
        # down 部分
        x1 = self.inputs(x)  # 64,64,64,64
        x2 = self.down_1(x1)  # 128,64,64,64
        x3 = self.down_2(x2)  # 256,16,16,16
        cf_fp = x3
        x4 = self.down_3(x3)  # 512,8,8,8

        cf_fp,cls = self.classifier(cf_fp)     #512,8,8,8

        cf_fp = self.prompt_tf(cf_fp)
        if self.prompt == 'add':
            d_out = cf_fp+x4         #add
        else:
            d_out = cf_fp*x4         #false
        # up部分
        x5 = self.up_1(d_out, x3)  # 256,16,16,16
        x6 = self.up_2(x5, x2)  # 128,32,32,32
        x7 = self.up_3(x6, x1)  # 64,64,64,64
        x = self.outputs(x7)  # 4,64,64,64

        return x, cls, d_out

class prompt_UNet3D_conv(nn.Module):
    def __init__(self, in_channels, num_classes=2, batch_normal=False, bilinear=True,prompt=True):
        super(prompt_UNet3D_conv, self).__init__()
        self.in_channels = in_channels
        self.batch_normal = batch_normal
        self.bilinear = bilinear
        self.prompt=prompt
        self.inputs = DoubleConv(in_channels, 64, self.batch_normal)
        self.down_1 = DownSampling(64, 128, self.batch_normal)
        self.down_2 = DownSampling(128, 256, self.batch_normal)
        self.down_3 = DownSampling(256, 512, self.batch_normal)

        self.classifier = classifier(256,3)
        self.dane = nn.Conv3d(1024,512,3,1,'same',bias=True)
        self.prompt_tf = nn.Sequential(
            nn.Conv3d(512, 512, 3, 1, 'same', bias=True),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, 3, 1, 'same', bias=True),
            nn.BatchNorm3d(512),
            nn.ReLU(),
        )

        self.up_1 = UpSampling(512, 256, self.batch_normal, self.bilinear)
        self.up_2 = UpSampling(256, 128, self.batch_normal, self.bilinear)
        self.up_3 = UpSampling(128, 64, self.batch_normal, self.bilinear)
        self.outputs = LastConv(64, num_classes)

    def forward(self, x):
        # down 部分
        x1 = self.inputs(x)  # 64,64,64,64
        x2 = self.down_1(x1)  # 128,64,64,64
        x3 = self.down_2(x2)  # 256,16,16,16
        cf_fp = x3
        x4 = self.down_3(x3)  # 512,8,8,8

        cf_fp,cls = self.classifier(cf_fp)     #512,8,8,8

        cf_fp = self.prompt_tf(cf_fp)
        d_out = self.dane(torch.concat([cf_fp,x4],dim=1))           #true
        # up部分
        x5 = self.up_1(d_out, x3)  # 256,16,16,16
        x6 = self.up_2(x5, x2)  # 128,32,32,32
        x7 = self.up_3(x6, x1)  # 64,64,64,64
        x = self.outputs(x7)  # 4,64,64,64

        return x, cls, d_out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            # nn.Conv2d(5, 5, 3, 2, 1)Out[24]: Conv2d(5, 5, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            block = [nn.Conv3d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout3d(0.25)]
            if bn:
                block.append(nn.BatchNorm3d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(512, 256, bn=False),
            *discriminator_block(256, 512),
            *discriminator_block(512, 256),
            *discriminator_block(256, 128),
        )

        ds_size = 2048 // 2 ** 4
        #        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.adv_layer = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity