import torch
import torch.nn as nn
import torch.nn.functional as F


class Resnet18NoRes(nn.Module):
    def __init__(self, num_classes=10):
        super(Resnet18NoRes, self).__init__()

        # Block 1
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3) # padding = (f-1)/2 => (7-1)/2 => 3
        self.bn_1 = nn.BatchNorm2d(64)
        self.pool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Block 2 [(3 x 3, 64), (3 x 3, 64)] x 2
        self.conv_2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn_2_1 = nn.BatchNorm2d(64)
        self.conv_2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn_2_2 = nn.BatchNorm2d(64)

        self.conv_2_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn_2_3 = nn.BatchNorm2d(64)
        self.conv_2_4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn_2_4 = nn.BatchNorm2d(64)

        # Block 3 [(3 x 3, 128), (3 x 3, 128)] x 2
        self.conv_3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn_3_1 = nn.BatchNorm2d(128)
        self.conv_3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn_3_2 = nn.BatchNorm2d(128)
        

        self.conv_3_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn_3_3 = nn.BatchNorm2d(128)
        self.conv_3_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn_3_4 = nn.BatchNorm2d(128)

        # Block 4 [(3 x 3, 256), (3 x 3, 256)] x 2
        self.conv_4_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.bn_4_1 = nn.BatchNorm2d(256)
        self.conv_4_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn_4_2 = nn.BatchNorm2d(256)
        

        self.conv_4_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn_4_3 = nn.BatchNorm2d(256)
        self.conv_4_4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn_4_4 = nn.BatchNorm2d(256)

        # Block 5 [(3 x 3, 512), (3 x 3, 512)] x 2
        self.conv_5_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.bn_5_1 = nn.BatchNorm2d(512)
        self.conv_5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn_5_2 = nn.BatchNorm2d(512)

        self.conv_5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn_5_3 = nn.BatchNorm2d(512)
        self.conv_5_4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn_5_4 = nn.BatchNorm2d(512)

        # Average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # Fully connected layer
        self.fc = nn.Linear(in_features=512, out_features=num_classes)
    
    def forward(self, x):
        # Block 1 
        x_1 = self.bn_1(self.conv_1(x))
        x_1 = F.relu(x_1)
        x_1 = self.pool_1(x_1)

        # Block 2
        x_2 = self.bn_2_1(self.conv_2_1(x_1))
        x_2 = F.relu(x_2)
        x_2 = self.bn_2_2(self.conv_2_2(x_2))
        x_2_i = F.relu(x_2)  

        x_2 = self.bn_2_3(self.conv_2_3(x_2_i))
        x_2 = F.relu(x_2)
        x_2 = self.bn_2_4(self.conv_2_4(x_2))
        x_2 = F.relu(x_2)   

        # Block 3
        x_3 = self.bn_3_1(self.conv_3_1(x_2))
        x_3 = F.relu(x_3)
        x_3 = self.bn_3_2(self.conv_3_2(x_3))
        x_3_i = F.relu(x_3)  

        x_3 = self.bn_3_3(self.conv_3_3(x_3_i))
        x_3 = F.relu(x_3)
        x_3 = self.bn_3_4(self.conv_3_4(x_3))
        x_3 = F.relu(x_3)   

        # Block 4
        x_4 = self.bn_4_1(self.conv_4_1(x_3))
        x_4 = F.relu(x_4)
        x_4 = self.bn_4_2(self.conv_4_2(x_4))
        x_4_i = F.relu(x_4)   

        x_4 = self.bn_4_3(self.conv_4_3(x_4_i))
        x_4 = F.relu(x_4)
        x_4 = self.bn_4_4(self.conv_4_4(x_4))
        x_4 = F.relu(x_4)     

        # Block 5
        x_5 = self.bn_5_1(self.conv_5_1(x_4))
        x_5 = F.relu(x_5)
        x_5 = self.bn_5_2(self.conv_5_2(x_5))
        x_5_i = F.relu(x_5)   

        x_5 = self.bn_5_3(self.conv_5_3(x_5_i))
        x_5 = F.relu(x_5)
        x_5 = self.bn_5_4(self.conv_5_4(x_5))
        x_5 = F.relu(x_5)    

        # Average Pool
        x_o = self.avg_pool(x_5)
        x_o = torch.flatten(x_o, 1)
        x_o = self.fc(x_o)
        return x_o



