import os
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import ViTImageProcessor
from torchvision import transforms


# CLASS NAMES


CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
    "Apple___healthy", "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot",
    "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

PLANTDOC_CLASS_NAMES = [
    "Apple Scab Leaf",                       # 0
    "Apple leaf",                            # 1
    "Apple rust leaf",                       # 2
    "Bell_pepper leaf spot",                 # 3
    "Bell_pepper leaf",                      # 4
    "Blueberry leaf",                        # 5
    "Cherry leaf",                           # 6
    "Corn Gray leaf spot",                   # 7
    "Corn leaf blight",                      # 8
    "Corn rust leaf",                        # 9
    "Peach leaf",                            # 10
    "Potato leaf early blight",              # 11
    "Potato leaf late blight",               # 12
    "Potato leaf",                           # 13
    "Raspberry leaf",                        # 14
    "Soyabean leaf",                         # 15
    "Soybean leaf",                          # 16
    "Squash Powdery mildew leaf",            # 17
    "Strawberry leaf",                       # 18
    "Tomato Early blight leaf",              # 19
    "Tomato Septoria leaf spot",             # 20
    "Tomato leaf bacterial spot",            # 21
    "Tomato leaf late blight",               # 22
    "Tomato leaf mosaic virus",              # 23
    "Tomato leaf yellow virus",              # 24
    "Tomato leaf",                           # 25
    "Tomato mold leaf",                      # 26
    "Tomato two spotted spider mites leaf",  # 27
    "grape leaf black rot",                  # 28
    "grape leaf",                            # 29
]


# PLANTDOC → PLANTVILLAGE CLASS MAPPING


PLANTDOC_TO_PLANTVILLAGE = {
    # Apple
    "Apple Scab Leaf":                       "Apple___Apple_scab",
    "Apple leaf":                            "Apple___healthy",
    "Apple rust leaf":                       "Apple___Cedar_apple_rust",

    # Bell pepper — có dấu _ trong tên PlantDoc
    "Bell_pepper leaf spot":                 "Pepper,_bell___Bacterial_spot",
    "Bell_pepper leaf":                      "Pepper,_bell___healthy",

    # Blueberry
    "Blueberry leaf":                        "Blueberry___healthy",

    # Cherry
    "Cherry leaf":                           "Cherry_(including_sour)___healthy",

    # Corn
    "Corn Gray leaf spot":                   "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight":                      "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf":                        "Corn_(maize)___Common_rust_",

    # Peach
    "Peach leaf":                            "Peach___healthy",

    # Potato
    "Potato leaf early blight":              "Potato___Early_blight",
    "Potato leaf late blight":               "Potato___Late_blight",
    "Potato leaf":                           "Potato___healthy",

    # Raspberry
    "Raspberry leaf":                        "Raspberry___healthy",

    # Soybean — PlantDoc có 2 tên (1 bị typo)
    "Soybean leaf":                          "Soybean___healthy",
    "Soyabean leaf":                         "Soybean___healthy",   # typo trong PlantDoc

    # Squash
    "Squash Powdery mildew leaf":            "Squash___Powdery_mildew",

    # Strawberry
    "Strawberry leaf":                       "Strawberry___healthy",

    # Tomato
    "Tomato Early blight leaf":              "Tomato___Early_blight",
    "Tomato Septoria leaf spot":             "Tomato___Septoria_leaf_spot",
    "Tomato leaf bacterial spot":            "Tomato___Bacterial_spot",
    "Tomato leaf late blight":               "Tomato___Late_blight",
    "Tomato leaf mosaic virus":              "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus":              "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato leaf":                           "Tomato___healthy",
    "Tomato mold leaf":                      "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf":  "Tomato___Spider_mites Two-spotted_spider_mite",

    # Grape — viết thường trong PlantDoc
    "grape leaf black rot":                  "Grape___Black_rot",
    "grape leaf":                            "Grape___healthy",
}


# ĐỌC DỮ LIỆU


def read_split_file(split_file, dataroot):
    """Đọc PlantVillage từ split .txt file."""
    samples = []
    with open(split_file, "r") as f:
        for line in f:
            rel_path = line.strip()
            if not rel_path:
                continue
            parts = rel_path.split("/")
            if len(parts) < 4:
                continue
            class_name = parts[2]
            if class_name not in CLASS_TO_IDX:
                continue
            samples.append((
                os.path.join(dataroot, rel_path),
                CLASS_TO_IDX[class_name],
                "plantvillage",                   # ← tag domain
            ))
    return samples



def read_plantdoc(plantdoc_root, target_split="train"):
    """
    Đọc PlantDoc YOLO format theo từng tập (train/valid/test)
    và in ra danh sách các ảnh bị thiếu nhãn.
    """
    samples = []
    skipped = 0
    
    # Tạo một list để lưu tên các file không có nhãn
    missing_labels_list = [] 

    img_dir   = os.path.join(plantdoc_root, target_split, "images")
    label_dir = os.path.join(plantdoc_root, target_split, "labels")

    if not os.path.exists(img_dir):
        return samples

    for fname in os.listdir(img_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path   = os.path.join(img_dir, fname)

        # Tìm file label tương ứng
        label_name = os.path.splitext(fname)[0] + ".txt"
        label_path = os.path.join(label_dir, label_name)

        # Trường hợp 1: File .txt không tồn tại
        if not os.path.exists(label_path):
            missing_labels_list.append(fname)
            continue

        with open(label_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        # Trường hợp 2: File .txt tồn tại nhưng trống rỗng (không có dòng nào)
        if not lines:
            missing_labels_list.append(fname)
            continue

        class_ids = [int(l.split()[0]) for l in lines]
        class_id  = max(set(class_ids), key=class_ids.count)

        if class_id >= len(PLANTDOC_CLASS_NAMES):
            skipped += 1
            continue

        pd_class_name = PLANTDOC_CLASS_NAMES[class_id]
        pv_class_name = PLANTDOC_TO_PLANTVILLAGE.get(pd_class_name)

        if pv_class_name is None or pv_class_name not in CLASS_TO_IDX:
            skipped += 1
            continue

        samples.append((
            img_path,
            CLASS_TO_IDX[pv_class_name],
            "plantdoc",
        ))

    
    print(f"PlantDoc ({target_split.upper()}): {len(samples):,} ảnh | no_label={len(missing_labels_list)} | skipped={skipped}")
            
    return samples

# AUGMENTATION
#

# PlantVillage — ảnh studio, augmentation nhẹ
pv_augmentation = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
])

# PlantDoc — ảnh thực tế, augmentation mạnh hơn để bù thiếu data
pd_augmentation = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
])

# Val/Test — không augmentation
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
])

# DATASET


processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")


class LeafDiseaseDataset(Dataset):
    def __init__(self, samples, is_train=False):
        """
        samples: list of (image_path, label, domain)
        domain : "plantvillage" | "plantdoc"
        """
        self.samples  = samples
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, domain = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        # Augmentation theo domain
        if self.is_train:
            if domain == "plantdoc":
                image = pd_augmentation(image)
            else:
                image = pv_augmentation(image)
        else:
            image = val_transform(image)

        # ViTImageProcessor: normalize + convert to tensor
        image = processor(image, return_tensors="pt")["pixel_values"].squeeze(0)

        return image, label



# DATALOADER


def get_dataloader(dataroot, split_file, data_type="segmented",
                   plantdoc_root=None,
                   batch_size=32, val_ratio=0.2, num_workers=4):
    """
    plantdoc_root : đường dẫn PlantDoc dataset (None = chỉ dùng PlantVillage)
    """

    # ── 1. Đọc PlantVillage ──
    train_txt = os.path.join(split_file, f"{data_type}_train.txt")
    test_txt  = os.path.join(split_file, f"{data_type}_test.txt")

    train_val_samples = read_split_file(train_txt, dataroot)
    test_samples      = read_split_file(test_txt,  dataroot)

    # ── 2. Chia train / val cho PlantVillage TRƯỚC ──
    labels_pv = [s[1] for s in train_val_samples]
    train_samples, val_samples = train_test_split(
        train_val_samples,
        test_size=val_ratio,
        stratify=labels_pv,
        random_state=42,
    )

    # ── 3. Gộp PlantDoc vào ĐÚNG TẬP (Chống rò rỉ dữ liệu) ──
    if plantdoc_root:
        pd_train = read_plantdoc(plantdoc_root, target_split="train")
        pd_val   = read_plantdoc(plantdoc_root, target_split="valid")
        pd_test  = read_plantdoc(plantdoc_root, target_split="test")

        train_samples = train_samples + pd_train
        val_samples   = val_samples + pd_val
        test_samples  = test_samples + pd_test

    # ── In thống kê ──
    n_pv_train = sum(1 for s in train_samples if s[2] == "plantvillage")
    n_pd_train = sum(1 for s in train_samples if s[2] == "plantdoc")
    print(f"Train     : {len(train_samples):,} (PV={n_pv_train:,} | PD={n_pd_train:,})")
    
    n_pv_val = sum(1 for s in val_samples if s[2] == "plantvillage")
    n_pd_val = sum(1 for s in val_samples if s[2] == "plantdoc")
    print(f"Val       : {len(val_samples):,} (PV={n_pv_val:,} | PD={n_pd_val:,})")
    
    n_pv_test = sum(1 for s in test_samples if s[2] == "plantvillage")
    n_pd_test = sum(1 for s in test_samples if s[2] == "plantdoc")
    print(f"Test      : {len(test_samples):,} (PV={n_pv_test:,} | PD={n_pd_test:,})")

    # ── 4. Dataset ──
    train_dataset = LeafDiseaseDataset(train_samples, is_train=True)
    val_dataset   = LeafDiseaseDataset(val_samples,   is_train=False)
    test_dataset  = LeafDiseaseDataset(test_samples,  is_train=False)

    train_labels = torch.tensor([s[1] for s in train_samples])
    class_counts = torch.bincount(train_labels, minlength=len(CLASS_NAMES)).float()
    
    class_weights = 1.0 / (class_counts + 1e-6)
    
    sample_weights = []
    for s in train_samples:
        label = s[1]
        domain = s[2]
        
        weight = class_weights[label].item()
        
        # Nếu là ảnh PlantDoc, tăng ưu tiên bốc thăm lên 10 lần
        if domain == "plantdoc":
            weight = weight * 10.0  
            
        sample_weights.append(weight)

    sample_weights = torch.tensor(sample_weights, dtype=torch.float)
    
    if plantdoc_root and n_pd_train > 0:
        print("Sampler   : Class-balanced & Domain Prioritize (PlantDoc x10)")
    else:
        print("Sampler   : Class-balanced")

    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(train_samples),
        replacement=True,
    )

    # ── 6. DataLoader ──
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=sampler, num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False,  num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False,  num_workers=num_workers,
                              pin_memory=True)

    return train_loader, val_loader, test_loader