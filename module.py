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

class InceptionBlock(nn.Module):
    """
    Class này định nghĩa "viên gạch" Inception theo sơ đồ.
    Nó nhận vào các con số cấu hình để có thể tái sử dụng.
    """
    def __init__(self, 
                 in_channels,    # Số kênh của dữ liệu đi vào khối này (ví dụ: 192)
                 out_1x1,        # Số kênh đầu ra của Nhánh 1 (nhánh 1x1 Conv)
                 bottleneck_3x3, # Số kênh "thắt cổ chai" (lớp 1x1) của Nhánh 2
                 out_3x3,        # Số kênh đầu ra (lớp 3x3) của Nhánh 2
                 bottleneck_5x5, # Số kênh "thắt cổ chai" (lớp 1x1) của Nhánh 3
                 out_5x5,        # Số kênh đầu ra (lớp 5x5) của Nhánh 3
                 pool_proj):     # Số kênh đầu ra (lớp 1x1) của Nhánh 4 (sau MaxPool)
        super(InceptionBlock, self).__init__()

    # --- Nhánh 1: 1x1 Conv ---
        self.branch1 = nn.Sequential( # Nó giống như một cái hộp ( container ) chứa các thành phần xử lý nhỏ bên trong
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # --- Nhánh 2: 1x1 Conv -> 3x3 Conv ---
        # (bottleneck_3x3 là lớp "thắt cổ chai" 1x1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_3x3, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_3x3, out_3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # --- Nhánh 3: 1x1 Conv -> 5x5 Conv ---
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_5x5, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_5x5, out_5x5, kernel_size=5, padding=2), # 5x5 pad 2
            nn.ReLU(inplace=True)
        )

        # --- Nhánh 4: 3x3 MaxPool -> 1x1 Conv ---
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Chạy dữ liệu qua 4 nhánh song song
        out_branch1 = self.branch1(x)
        out_branch2 = self.branch2(x)
        out_branch3 = self.branch3(x)
        out_branch4 = self.branch4(x)
        
        # Nối (Concatenate) kết quả của 4 nhánh lại theo chiều kênh (dim=1)
        # [B, C1, H, W] + [B, C2, H, W] -> [B, C1+C2, H, W]
        return torch.cat([out_branch1, out_branch2, out_branch3, out_branch4], 1)

class GoogLeNet(nn.Module):
    """
    Class này định nghĩa "bộ khung" GoogLeNet (Inception v1)
    theo đúng các tham số (số kênh) trong bài báo gốc.
    """
    def __init__(self, num_classes=21): # Nhận num_classes
        super(GoogLeNet, self).__init__()

        # --- 1. Lớp "Stem" (Thân) ---
        # Đây là các lớp đầu tiên
        # Input: (3, 224, 224) -> Output: (192, 28, 28)
        self.stem = nn.Sequential(
            # (3, 224, 224) -> (64, 112, 112)
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            # (64, 112, 112) -> (64, 56, 56)
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # LRN (Local Response Norm) - đã lỗi thời, bỏ qua
            # (64, 56, 56) -> (64, 56, 56)
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            # (64, 56, 56) -> (192, 56, 56)
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # (192, 56, 56) -> (192, 28, 28)
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # --- 2. Chuỗi Inception (Thân) ---
        # Các con số (64, 96, 128,...) được lấy từ bài báo gốc
        
        # Input (192, 28, 28) -> Output (256, 28, 28)
        self.inception_3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        # Input (256, 28, 28) -> Output (480, 28, 28)
        self.inception_3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        
        # Giảm kích thước (28, 28) -> (14, 14)
        self.maxpool_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Input (480, 14, 14) -> Output (512, 14, 14)
        self.inception_4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception_4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        # Input (528, 14, 14) -> Output (832, 14, 14)
        self.inception_4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        
        # Giảm kích thước (14, 14) -> (7, 7)
        self.maxpool_4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Input (832, 7, 7) -> Output (832, 7, 7)
        self.inception_5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        # Input (832, 7, 7) -> Output (1024, 7, 7)
        self.inception_5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        # --- 3. Lớp "Head" (Đầu) ---
        # Đây là các lớp cuối cùng để phân loại
        
        # (1024, 7, 7) -> (1024, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling
        
        self.dropout = nn.Dropout(0.4) # Dropout 40% để giảm overfitting vì nó biến đổi giá trị 0 ngẫu nhiên vào trong quá trình huấn luyện và tăng các giá trị còn lại lên.
        
        # (1024) -> (num_classes)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Chạy tuần tự qua các lớp
        
        # 1. Stem
        x = self.stem(x)
        
        # 2. Inception 3
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.maxpool_3(x)
        
        # 3. Inception 4
        x = self.inception_4a(x)
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        x = self.inception_4e(x)
        x = self.maxpool_4(x)
        
        # 4. Inception 5
        x = self.inception_5a(x)
        x = self.inception_5b(x)
        
        # 5. Head
        x = self.avgpool(x)     # (B, 1024, 1, 1)
        x = torch.flatten(x, 1) # (B, 1024)
        x = self.dropout(x)
        x = self.fc(x)          # (B, num_classes)
        
        return x



class BasicBlock(nn.Module):
    """
    Class này định nghĩa "viên gạch" ResNet cơ bản (cho ResNet-18/34).
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # --- 1. Nhánh chính (Main Path) ---
        # [3x3 Conv -> Batch norm -> ReLU -> 3x3 Conv -> Batch norm]
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # --- 2. Nhánh tắt (Shortcut Path) ---
        self.shortcut = nn.Sequential() # Mặc định là một "identity" (không làm gì)
        
        # Nếu kích thước thay đổi (stride != 1) hoặc số kênh thay đổi
        # nhánh "x" (shortcut) phải đi qua 1x1 Conv để điều chỉnh
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x) # Lưu lại 'x' (hoặc 'x' đã biến đổi)
        
        # Cho 'x' đi qua nhánh chính
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # --- Phép cộng (Phép thuật của ResNet) ---
        out += identity
        out = self.relu(out) # ReLU *sau* khi cộng
        
        return out

class ResNet(nn.Module):
    """
    Class này định nghĩa "bộ khung" ResNet.
    """
    def __init__(self, block, num_blocks, num_classes=21):
        super(ResNet, self).__init__()
        self.in_channels = 64 # Theo dõi số kênh đầu vào cho mỗi tầng
        
        # --- 1. Lớp "Stem" ---
        # Input: (3, 224, 224) -> (64, 56, 56)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # --- 2. Các tầng ResNet (Lắp ráp "gạch") ---
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # --- 3. Lớp "Head" ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling
        self.fc = nn.Linear(512, num_classes) # ResNet-18/34 có 512 kênh ở cuối

    # Hàm trợ giúp để tạo một tầng (stack) các block
    def _make_layer(self, block, out_channels, num_blocks, stride):
        # Block đầu tiên của tầng có thể thay đổi kích thước (stride=2)
        # Các block còn lại stride=1
        strides = [stride] + [1] * (num_blocks - 1)
        
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels # Cập nhật in_channels cho block tiếp theo
            
        return nn.Sequential(*layers) # Gói các block lại thành một "hộp"

    def forward(self, x):
        # 1. Stem
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        # 2. Các tầng ResNet
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # 3. Head
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out

# --- Hàm tiện ích để gọi ResNet-18 ---
def ResNet18(num_classes=21):
    """
    Hàm này xây dựng ResNet-18 chuẩn với:
    - Block: BasicBlock
    - Cấu hình: [2, 2, 2, 2] (tổng cộng 8 block)
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)