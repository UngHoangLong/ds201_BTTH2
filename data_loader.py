import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import os

# Kích thước LeNet-5 mong đợi
LENET_IMG_SIZE = 32
random_state = 42
def get_vinfood_dataloaders_for_LeNet_5(batch_size=64, train_path=None, test_path=None, val_split=0.2):
    # định nghĩa các phép biến đổi ( transformer )
    data_transformer = transforms.Compose([ # 
        # Thêm dòng này để xử lý ảnh PNG/Palette
        transforms.Convert("RGB"),
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

        full_train_dataset = datasets.ImageFolder(
            root=train_path,
            transform=data_transformer
        )
        
        test_dataset = datasets.ImageFolder(
            root=test_path,
            transform=data_transformer
        )
    except Exception as e:
        print("Error loading datasets: ", e)
        return None, None, -1

    # 4. Lấy danh sách nhãn (targets) từ dataset
    # .targets là một danh sách [0, 0, 1, 1, 2, ..., 20]
    all_labels = full_train_dataset.targets
    
    # 5. Lấy danh sách chỉ số (indices)
    all_indices = list(range(len(all_labels)))

    # 6. Dùng train_test_split để chia các *chỉ số*
    # Nó sẽ chia all_indices thành 2 nhóm, dựa trên all_labels
    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=val_split,       # Tỷ lệ 20% cho validation
        stratify=all_labels,     # Đây là mấu chốt: chia đều các nhãn
        random_state=random_state  # Giúp kết quả chia luôn giống nhau mỗi lần chạy
    )

    # 7. Tạo các tập con (Subset) từ các chỉ số đã chia
    # Subset sẽ bọc dataset gốc và chỉ lấy các index được chỉ định
    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(full_train_dataset, val_indices)

    # Lấy số lượng lớp 
    num_classes = len(full_train_dataset.classes)
    # Tạo Dataloader
    train_loader = DataLoader(
        dataset=train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        persistent_workers=True
    )

    val_loader = DataLoader(
        dataset=val_subset,
        batch_size=batch_size,
        shuffle=False, # Không cần shuffle tập valid/test
        num_workers=2,
        persistent_workers=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        persistent_workers=True
    )

    print(f"Đã tải data thành công. Tổng cộng {num_classes} lớp.")

    return train_loader, val_loader, test_loader, num_classes

