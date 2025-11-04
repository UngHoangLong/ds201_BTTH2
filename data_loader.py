import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Kích thước LeNet-5 mong đợi
LENET_IMG_SIZE = 32

def get_vinfood_dataloaders_for_LeNet_5(batch_size=64, train_path=None, test_path=None):
    # định nghĩa các phép biến đổi ( transformer )
    data_transformer = transforms.Compose([ # 
        # Thay đổi kích thước ảnh về 32x32
        transforms.Resize((LENET_IMG_SIZE, LENET_IMG_SIZE)),
        # Chuyển ảnh sang Tensor [C,H,W] và scale giá trị pixel về [0.0,1.0]
        transforms.ToTensor(),
        # Vì dữ liệu là hình ảnh tự nhiên nên kế thứ mean và std của ImageNet để chuẩn hoá
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406], 
            std = [0.229, 0.224, 0.225]
        )
    ])

    try:
        # Tải dữ liệu train
        train_dataset = datasets.ImageFolder(
            root=train_path,
            transform=data_transformer
        )
        # Tải dữ liệu test
        test_dataset = datasets.ImageFolder(
            root=test_path,
            transform=data_transformer
        )
    except Exception as e:
        print("Error loading datasets: ", e)
        return None, None, -1

    # Lấy số lượng lớp 
    num_classes = len(train_dataset.classes)
    # Tạo Dataloeader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    print(f"Đã tải data thành công. Tổng cộng {num_classes} lớp.")

    return train_loader, test_loader, num_classes

