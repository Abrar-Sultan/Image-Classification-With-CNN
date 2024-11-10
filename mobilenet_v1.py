import torch
import torch.nn as nn
import torch.nn.functional as F

class MobileNet(nn.Module):
    def __init__(self, num_class = 10):
        super(MobileNet, self).__init__()

        # Block 1 (standard conv)
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.bn_1 = nn.BatchNorm2d(32)

        # Block 2 
        # Depthwise separable convolutions
        self.conv_dw_2_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32)
        self.bn_2_1 = nn.BatchNorm2d(32)

        # Pointwise convolution
        self.conv_pw_2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn_2_2 = nn.BatchNorm2d(64)

        # Block 3 
        # Depthwise separable convolutions
        self.conv_dw_3_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, groups=64)
        self.bn_3_1 = nn.BatchNorm2d(64)

        # Pointwise convolution
        self.conv_pw_3_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.bn_3_2 = nn.BatchNorm2d(128)

        # Block 4 
        # Depthwise separable convolutions
        self.conv_dw_4_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128)
        self.bn_4_1 = nn.BatchNorm2d(128)

        # Pointwise convolution
        self.conv_pw_4_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.bn_4_2 = nn.BatchNorm2d(128)

        # Block 5
        # Depthwise separable convolutions
        self.conv_dw_5_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, groups=128)
        self.bn_5_1 = nn.BatchNorm2d(128)

        # Pointwise convolution
        self.conv_pw_5_4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.bn_5_2 = nn.BatchNorm2d(256)

        # Block 6
        # Depthwise separable convolutions
        self.conv_dw_6_5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, groups=256)
        self.bn_6_1 = nn.BatchNorm2d(256)

        # Pointwise convolution
        self.conv_pw_6_5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.bn_6_2 = nn.BatchNorm2d(256)

        # Block 7
        # Depthwise separable convolutions
        self.conv_dw_7_6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, groups=256)
        self.bn_7_1 = nn.BatchNorm2d(256)

        # Pointwise convolution
        self.conv_pw_7_6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.bn_7_2 = nn.BatchNorm2d(512)

        # Block 8 -1
        # Depthwise separable convolutions
        self.conv_dw_8_7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, groups=512)
        self.bn_8_1 = nn.BatchNorm2d(512)

        # Pointwise convolution
        self.conv_pw_8_7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.bn_8_2 = nn.BatchNorm2d(512)

        # Block 9 -2
        # Depthwise separable convolutions
        self.conv_dw_9_8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, groups=512)
        self.bn_9_1 = nn.BatchNorm2d(512)

        # Pointwise convolution
        self.conv_pw_9_8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.bn_9_2 = nn.BatchNorm2d(512)

        # Block 10 -3
        # Depthwise separable convolutions
        self.conv_dw_10_9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, groups=512)
        self.bn_10_1 = nn.BatchNorm2d(512)

        # Pointwise convolution
        self.conv_pw_10_9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.bn_10_2 = nn.BatchNorm2d(512)

        # Block 11 -4
        # Depthwise separable convolutions
        self.conv_dw_11_10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, groups=512)
        self.bn_11_1 = nn.BatchNorm2d(512)

        # Pointwise convolution
        self.conv_pw_11_10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.bn_11_2 = nn.BatchNorm2d(512)

        # Block 12 -5
        # Depthwise separable convolutions
        self.conv_dw_12_11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, groups=512)
        self.bn_12_1 = nn.BatchNorm2d(512)

        # Pointwise convolution
        self.conv_pw_12_11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.bn_12_2 = nn.BatchNorm2d(512)

        # Block 13
        # Depthwise separable convolutions
        self.conv_dw_13_12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, groups=512)
        self.bn_13_1 = nn.BatchNorm2d(512)

        # Pointwise convolution
        self.conv_pw_13_12 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1, padding=0)
        self.bn_13_2 = nn.BatchNorm2d(1024)

        # Block 14
        # Depthwise separable convolutions
        self.conv_dw_14_13 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, groups=1024)
        self.bn_14_1 = nn.BatchNorm2d(1024)

        # Pointwise convolution
        self.conv_pw_14_13 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, padding=0)
        self.bn_14_2 = nn.BatchNorm2d(1024)

        # Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layer
        self.fc = nn.Linear(1024, num_class)

    def forward(self, x):
        # Block 1 (standard conv)
        x = F.relu(self.bn_1(self.conv_1(x)))

        # Block 2
        # depthwise conv
        x = F.relu(self.bn_2_1(self.conv_dw_2_1(x)))

        # pointwise conv
        x = F.relu(self.bn_2_2(self.conv_pw_2_1(x)))

        # Block 3
        # depthwise conv
        x = F.relu(self.bn_3_1(self.conv_dw_3_2(x)))
        
        # pointwise conv
        x = F.relu(self.bn_3_2(self.conv_pw_3_2(x)))

        # Block 4
        # depthwise conv
        x = F.relu(self.bn_4_1(self.conv_dw_4_3(x)))
        
        # pointwise conv
        x = F.relu(self.bn_4_2(self.conv_pw_4_3(x)))

        # Block 5
        # depthwise conv
        x = F.relu(self.bn_5_1(self.conv_dw_5_4(x)))
        
        # pointwise conv
        x = F.relu(self.bn_5_2(self.conv_pw_5_4(x)))

        # Block 6
        # depthwise conv
        x = F.relu(self.bn_6_1(self.conv_dw_6_5(x)))
    
        # pointwise conv
        x = F.relu(self.bn_6_2(self.conv_pw_6_5(x)))

        # Block 7
        # depthwise conv
        x = F.relu(self.bn_7_1(self.conv_dw_7_6(x)))
    
        # pointwise conv
        x = F.relu(self.bn_7_2(self.conv_pw_7_6(x)))

        # Block 8
        # depthwise conv
        x = F.relu(self.bn_8_1(self.conv_dw_8_7(x)))
    
        # pointwise conv
        x = F.relu(self.bn_8_2(self.conv_pw_8_7(x)))

        # Block 9
        # depthwise conv
        x = F.relu(self.bn_9_1(self.conv_dw_9_8(x)))
    
        # pointwise conv
        x = F.relu(self.bn_9_2(self.conv_pw_9_8(x)))

        # Block 10
        # depthwise conv
        x = F.relu(self.bn_10_1(self.conv_dw_10_9(x)))
    
        # pointwise conv
        x = F.relu(self.bn_10_2(self.conv_pw_10_9(x)))

        # Block 11
        # depthwise conv
        x = F.relu(self.bn_11_1(self.conv_dw_11_10(x)))
    
        # pointwise conv
        x = F.relu(self.bn_11_2(self.conv_pw_11_10(x)))

        # Block 12
        # depthwise conv
        x = F.relu(self.bn_12_1(self.conv_dw_12_11(x)))
    
        # pointwise conv
        x = F.relu(self.bn_12_2(self.conv_pw_12_11(x)))

        # Block 13
        # depthwise conv
        x = F.relu(self.bn_13_1(self.conv_dw_13_12(x)))
    
        # pointwise conv
        x = F.relu(self.bn_13_2(self.conv_pw_13_12(x)))

        # Block 14
        # depthwise conv
        x = F.relu(self.bn_14_1(self.conv_dw_14_13(x)))
    
        # pointwise conv
        x = F.relu(self.bn_14_2(self.conv_pw_14_13(x)))

        # Average Pooling
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x_o = self.fc(x)

        return x_o






