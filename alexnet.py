import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_class=10):
        super(AlexNet, self).__init__()

        # Layer 1
        import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        
        # Layer 1
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.pool_1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Layer 2
        self.conv_2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.pool_2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Layer 3
        self.conv_3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
      
        # Layer 4
        self.conv_4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        
        # Layer 5
        self.conv_5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.pool_5 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Fully connected layers
        self.fc_1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc_2 = nn.Linear(4096, 4096)
        self.fc_3 = nn.Linear(4096, num_classes)
        
        # Dropout layers
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Layer 1
        x = F.relu(self.conv_1(x))
        x = self.pool_1(x)
        
        # Layer 2
        x = F.relu(self.conv_2(x))
        x = self.pool_2(x)

        # Layer 3
        x = F.relu(self.conv_3(x))

        # Layer 4
        x = F.relu(self.conv_4(x))

        # Layer 5
        x = F.relu(self.conv_5(x))
        x = self.pool_5(x)
        
        # Fully connected layers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc_1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc_2(x))
        x = self.dropout(x)
        
        x_o = self.fc_3(x)
        
        return x_o



