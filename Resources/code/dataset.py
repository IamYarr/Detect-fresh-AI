import cv2
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


BASE_DIR = Path(__file__).resolve().parent.parent.parent
IMG_DIR = BASE_DIR / "Resources" / "img"

CLASSES = ["FRESH", "MINOR_DEFECT", "SPOILED"]

train_transform = A.Compose([
    A.Rotate(limit=30, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.Blur(blur_limit=3, p=0.3),
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

class FreshnessDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for label_idx, class_name in enumerate(CLASSES):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
                
            for img_path in class_dir.iterdir():
                if img_path.is_file():
                    self.image_paths.append(img_path)
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = str(self.image_paths[idx])
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, torch.tensor(label, dtype=torch.long)

def get_dataloaders(batch_size=16):
    train_dir = IMG_DIR / 'train'
    val_dir = IMG_DIR / 'val'
    
    train_dataset = FreshnessDataset(train_dir, transform=train_transform)
    val_dataset = FreshnessDataset(val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, train_dataset, val_dataset
