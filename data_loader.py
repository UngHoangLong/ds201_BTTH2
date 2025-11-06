import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import os
from transformers import AutoImageProcessor

# Kích thước LeNet-5 mong đợi
LENET_IMG_SIZE = 32
# Kích thước GoogleNet mong đợi
GOOGLENET_IMG_SIZE = 224
random_state = 42
def get_vinfood_dataloaders(batch_size=64, train_path=None, test_path=None, val_split=0.2, image_size=32):
    # định nghĩa các phép biến đổi ( transformer )
    data_transformer = transforms.Compose([ # 
        # Thêm dòng này để xử lý ảnh PNG/Palette
        transforms.Lambda(lambda img: img.convert("RGB")),
        # Thay đổi kích thước ảnh về 32x32
        transforms.Resize((image_size, image_size)),
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


def get_vinfood_dataloaders_for_HuggingFace(model_name="microsoft/resnet-50", batch_size=32, train_path=None, test_path=None, val_split=0.2, random_state=42):
    """
    Hàm này tạo DataLoaders cho VinaFood21
    SỬ DỤNG BỘ TIỀN XỬ LÝ (PROCESSOR) CỤ THỂ TỪ HUGGINGFACE.
    """
    
    # 1. Tải "bộ xử lý" (processor) của model
    # Processor này chứa mean, std, và kích thước resize (ví dụ: 224x224)
    # mà ResNet-50 đã được huấn luyện.
    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
    except Exception as e:
        print(f"Lỗi khi tải processor '{model_name}'. Bạn chắc chắn đã cài 'transformers'?")
        print(e)
        return None, None, None, -1

    # 2. Định nghĩa phép biến đổi
    if hasattr(processor, 'crop_size') and processor.crop_size is not None:
        # Dùng cho ResNet: (224, 224)
        target_size = (processor.crop_size['height'], processor.crop_size['width'])
    elif hasattr(processor, 'size') and processor.size is not None:
        if 'shortest_edge' in processor.size:
            # Dùng cho ConvNext: 224 (int)
            target_size = processor.size['shortest_edge']
        else:
            # Dùng cho ViT/BeIT: (224, 224)
            target_size = (processor.size['height'], processor.size['width'])
    else:
        # Mặc định an toàn
        target_size = (224, 224)

    # 3. Định nghĩa phép biến đổi
    data_transformer = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize(target_size), # <--- Dùng 'target_size' đa năng
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])
    
    # (Phần tải dataset, chia Stratified, và Dataloader giữ nguyên y hệt)
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
        print(f"Lỗi khi tải data: {e}")
        return None, None, None, -1

    all_labels = full_train_dataset.targets
    all_indices = list(range(len(all_labels)))
    train_indices, val_indices = train_test_split(
        all_indices, test_size=val_split, stratify=all_labels, random_state=random_state
    )
    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(full_train_dataset, val_indices)
    num_classes = len(full_train_dataset.classes)
    
    # Lấy 2 dictionary quan trọng
    label2id = full_train_dataset.class_to_idx
    id2label = {idx: label for label, idx in label2id.items()}

    train_loader = DataLoader(
        dataset=train_subset, batch_size=batch_size, shuffle=True,
        num_workers=2, persistent_workers=True
    )
    val_loader = DataLoader(
        dataset=val_subset, batch_size=batch_size, shuffle=False,
        num_workers=2, persistent_workers=True
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, persistent_workers=True
    )

    print(f"Đã tải data (chuẩn HF ResNet-50) thành công. Tổng cộng {num_classes} lớp.")
    print(f"Train: {len(train_subset)} ảnh | Validation: {len(val_subset)} ảnh (đã chia stratified).")
    
    # Trả về thêm 2 dicts (label2id, id2label)
    return train_loader, val_loader, test_loader, num_classes, label2id, id2label