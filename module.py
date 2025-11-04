import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module): # nn.Module là base class mà tất cả các mô hình học sâu kế thừa nó.
    def __init__(self, num_classes):
        super(LeNet, self).__init__() # Gọi hàm khởi tạo của lớp cha nn.Module

        # Lớp 1. 5x5 Conv (6) Input(3,32,32) -> Output(6,28,28)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        # Lớp 2. 2x2 Max Pooling -> Output(6,14,14)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # Lớp 3. 5x5 Conv (16) -> Output(16,10,10)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        # Lớp 4. 2x2 Max Pooling -> Output(16,5,5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # Lớp 5. Fully Connected (120)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        # Lớp 6. Fully Connected (84)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        # Lớp 7. Output (10)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = torch.flatten(x,1) # flatten tất cả các chiều ngoại trừ batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
