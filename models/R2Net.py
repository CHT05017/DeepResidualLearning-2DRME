"""
R2Net: 2D Deep Residual Learning with Height Embedding for 3D Radio Map Estimation
Rao et al., 2026 IEEE TVT

Source Code Implementation:
https://github.com/lighttime2023/3DiRM3200
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def Conv2d_BN_ReLU_MaxPool(in_channels, out_channels, kernel, padding, pool):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.BatchNorm2d(out_channels, eps=1e-5, momentum=1 - 0.999),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(pool, stride=pool, padding=0, dilation=1, return_indices=False, ceil_mode=False) 
    )

def Conv2D_BN_ReLU(in_channels, out_channels, kernel, padding, pool):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.BatchNorm2d(out_channels, eps=1e-5, momentum=1 - 0.999),
        nn.ReLU(inplace=True)
    )

def Conv2d_BN(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.BatchNorm2d(out_channels, eps=1e-5, momentum=1 - 0.999)
    )

def Conv2DTranspose_ReLU(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=2, padding=padding),
        nn.ReLU(inplace=True)
    )

def DilatConv2D_BN_ReLU(in_channels, out_channels, kernel, padding, dilation):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding, dilation=dilation),
        nn.BatchNorm2d(out_channels, eps=1e-5, momentum=1 - 0.999),
        nn.ReLU(inplace=True) 
    )

class ChannelBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels,downsample):
        super().__init__()
        mid_ch = out_channels // 4
        self.reduce = Conv2d_BN_ReLU_MaxPool(in_channels, mid_ch, 1, 0, 1)
        self.conv3x3 = Conv2d_BN_ReLU_MaxPool(mid_ch, mid_ch, 3, 1, 1)
        self.increase = Conv2d_BN(mid_ch, out_channels, 1, 0)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, 1)
            if downsample
            else nn.Identity()
        )

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        h += self.shortcut(x)
        return F.relu(h)

class R2Net(nn.Module):
    def __init__(self, in_channel=2, out_channel=1):
        super().__init__()
        self.layer1 = Conv2D_BN_ReLU(in_channel, 6, 3, 1,1) 
        self.layer2 = Conv2d_BN_ReLU_MaxPool(6, 64, 5, 2,2)
        
        self.layer3 = Conv2d_BN_ReLU_MaxPool(64, 256, 5, 2,2)  
        self.layer4 = ChannelBottleneck(256, 256, downsample=True)
        self.layer40 = ChannelBottleneck(256, 256, downsample=False)
        self.layer400 = ChannelBottleneck(256, 256, downsample=False)
        
        self.layer5 = Conv2d_BN_ReLU_MaxPool(256, 512, 5, 2,2) 
        self.layer6 = ChannelBottleneck(512, 512, downsample=True)
        self.layer60 = ChannelBottleneck(512, 512, downsample=False)
        self.layer600 = ChannelBottleneck(512, 512, downsample=False)
        
        self.layer7 = Conv2d_BN_ReLU_MaxPool(512, 1024, 5, 2,2) 
        
        self.layer70 = ChannelBottleneck(1024, 1024, downsample=True)
        self.layer700 = ChannelBottleneck(1024, 1024, downsample=False)
        self.layer7000 = ChannelBottleneck(1024, 1024, downsample=False)

        
        self.layer71=ChannelBottleneck(1024, 1024, downsample=True)
        self.layer710=ChannelBottleneck(1024, 1024, downsample=False)
        self.layer7100=ChannelBottleneck(1024, 1024, downsample=False)

        
        self.layer72=Conv2d_BN_ReLU_MaxPool(1024, 256, 1, 0, 1)
        self.layer720=DilatConv2D_BN_ReLU(1024, 256, 3, padding=6, dilation=6)
        self.layer7200=DilatConv2D_BN_ReLU(1024, 256, 3, padding=12, dilation=12)
        self.layer72000=DilatConv2D_BN_ReLU(1024, 256, 3, padding=18, dilation=18)
        
        self.layer73=nn.AdaptiveAvgPool2d(1)
        self.layer730=Conv2d_BN_ReLU_MaxPool(1024, 256, 1, 0, 1)

        # NOTE
        # The 256 here comes from the result of the global avgpool (B, 256, 1, 1).
        # Of course, (B, 256, 1, 1) here was actually upsampled to (B, 256, 16, 16) by F.interpolate().
        # The 1024 here comes from 256 * 4, which means stacking four (B, 256, 16, 16) dilated convolution outputs.
        self.layer8 = Conv2D_BN_ReLU(1024+256, 1024, 1, 0, 1)
        
        self.conv_up3 = Conv2DTranspose_ReLU(1024 + 1024, 512, 4, 1) 
        self.conv_up4 = Conv2D_BN_ReLU(512 + 512, 512, 3, 1, 1) 
        self.conv_up5 = Conv2DTranspose_ReLU(512 + 512, 256, 6, 2) 
        self.conv_up6 = Conv2D_BN_ReLU(256 + 256, 256, 5, 2, 1) 
        self.conv_up7 = Conv2DTranspose_ReLU(256 + 256, 64, 6, 2)
        self.conv_up8 = Conv2DTranspose_ReLU(64 + 64, 64, 6, 2) 
        
        self.conv_up9 = Conv2D_BN_ReLU(64 + 6+in_channel, 32, 5, 2, 1)

        # BUG, ReLU() is \in [0, 1], training data is \in [-1, 1]
        # self.conv_up10 = Conv2D_BN_ReLU(32 + in_channel, out_channel, 5, 2, 1)
        self.conv_up10 = nn.Conv2d(
            in_channels=32 + in_channel,
            out_channels=out_channel,
            kernel_size=5,
            padding=2
        )

    def forward(self, input0):

        layer1 = self.layer1(input0) 
        layer2 = self.layer2(layer1) 
        layer3 = self.layer3(layer2) 
        layer4 = self.layer4(layer3) 
        layer40 = self.layer40(layer4) 
        layer400 = self.layer400(layer40)
        layer5 = self.layer5(layer400) 
        layer6 = self.layer6(layer5)
        layer60 = self.layer60(layer6)
        layer600 = self.layer600(layer60)
        layer7 = self.layer7(layer600) 
        layer70 = self.layer70(layer7)
        layer700 = self.layer700(layer70)
        layer7000 = self.layer7000(layer700)
        
        layer701 = self.layer70(layer7000)
        layer7010 = self.layer700(layer701)
        layer70100 = self.layer7000(layer7010)
        
        layer702 = self.layer70(layer70100)
        layer7020 = self.layer700(layer702)
        layer70200 = self.layer7000(layer7020)
        
        layer703 = self.layer70(layer70200)
        layer7030 = self.layer700(layer703)
        layer70300 = self.layer7000(layer7030)
        
        layer71 = self.layer71(layer70300)
        layer710 = self.layer710(layer71)
        layer7100 = self.layer7100(layer710)
        
        layer72 = self.layer72(layer7100)
        layer720 = self.layer720(layer7100)
        layer7200 = self.layer7200(layer7100)
        layer72000 = self.layer72000(layer7100)
        
        layer73 = self.layer73(layer7100)
        layer730 = self.layer730(layer73)
        layer7300 = F.interpolate(layer730, size=layer72.shape[-2:], mode="bilinear", align_corners=False)
        
        layer74=torch.cat([layer72,layer720,layer7200,layer72000,layer7300], dim=1)#185,16,16
        
        layer8 = self.layer8(layer74)
        
        layer2u = torch.cat([layer8, layer70300], dim=1) 
        
        layer30u = self.conv_up3(layer2u) 
        layer3u = torch.cat([layer30u, layer600], dim=1) 
        
        layer40u = self.conv_up4(layer3u)
        layer4u =  torch.cat([layer40u, layer5], dim=1)   
        
        layer50u = self.conv_up5(layer4u) 
        layer5u = torch.cat([layer50u, layer400], dim=1) 
        
        layer60u = self.conv_up6(layer5u)
        layer6u = torch.cat([layer60u, layer3], dim=1) 
        
        layer70u = self.conv_up7(layer6u) 
        layer7u = torch.cat([layer70u, layer2], dim=1)
        
        layer80u = self.conv_up8(layer7u) 
        layer81u = torch.cat([layer80u, layer1], dim=1) 
        layer8u = torch.cat([layer81u, input0], dim=1) 
        
        layer90u = self.conv_up9(layer8u) 
        layer9u = torch.cat([layer90u, input0], dim=1) 
        output = self.conv_up10(layer9u) 

        return output


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    model = R2Net(in_channel=6, out_channel=1)
    dummy = torch.randn(size=(4, 6, 128, 128))
    output = model(dummy)

    plt.figure()
    plt.imsave("./test.jpg", output[0, 0].detach().cpu().numpy(), cmap="viridis")
    plt.show()
    print(output.shape)
# python -m models.R2Net

