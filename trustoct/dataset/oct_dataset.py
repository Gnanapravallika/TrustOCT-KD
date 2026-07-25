import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class CLAHEPreprocessing:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to OCT images.
    Enhances micro-structural contrast of retinal layers (RPE, ILM, fluid pockets).
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def __call__(self, img_pil):
        img_np = np.array(img_pil)
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l_clahe = self.clahe.apply(l)
            lab_clahe = cv2.merge((l_clahe, a, b))
            img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        else:
            img_clahe = self.clahe.apply(img_np)
            img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(img_clahe)


class OCTDataset(Dataset):
    """
    PyTorch Dataset for Kermany Retinal OCT images (CNV, DME, DRUSEN, NORMAL).
    """
    def __init__(self, root_dir, split='train', classes=['CNV', 'DME', 'DRUSEN', 'NORMAL'], 
                 use_clahe=True, transform=None, max_samples_per_class=None):
        self.root_dir = os.path.abspath(root_dir)
        self.split = split
        self.classes = classes
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.transform = transform
        self.use_clahe = use_clahe
        self.clahe = CLAHEPreprocessing() if use_clahe else None
        
        self.image_paths = []
        self.labels = []
        
        split_dir = os.path.join(self.root_dir, split)
        if not os.path.exists(split_dir):
            split_dir = self.root_dir
            
        for cls_name in self.classes:
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.exists(cls_dir):
                continue
            
            valid_exts = ('.jpeg', '.jpg', '.png', '.tiff', '.bmp')
            filenames = [f for f in os.listdir(cls_dir) if f.lower().endswith(valid_exts)]
            
            if max_samples_per_class is not None:
                filenames = filenames[:max_samples_per_class]
                
            for fname in filenames:
                self.image_paths.append(os.path.join(cls_dir, fname))
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

        if self.use_clahe and self.clahe is not None:
            image = self.clahe(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_transforms(image_size=(224, 224), is_train=True):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if is_train:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


def print_class_distributions(data_dir, classes=['CNV', 'DME', 'DRUSEN', 'NORMAL']):
    print("\n--- Split Class Distributions ---")
    print(f"{'Class':<12} | {'Train':<8} | {'Val':<8} | {'Test':<8}")
    print("-" * 45)
    
    counts = {cls: {'train': 0, 'val': 0, 'test': 0} for cls in classes}
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if os.path.exists(split_dir):
            for cls in classes:
                cls_dir = os.path.join(split_dir, cls)
                if os.path.exists(cls_dir):
                    counts[cls][split] = len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
                    
    for cls in classes:
        print(f"{cls:<12} | {counts[cls]['train']:<8} | {counts[cls]['val']:<8} | {counts[cls]['test']:<8}")
    print("-" * 45 + "\n")


def get_dataloaders(data_dir, batch_size=32, num_workers=2, image_size=(224, 224), 
                    use_clahe=True, max_samples_per_class=None, val_ratio=0.10, seed=42):
    """
    Creates train, val, and test DataLoaders for OCT classification.
    
    Note on Kermany dataset:
      The raw Kermany 2018 folder has only 32 images in `val/` (8 per class),
      which causes artificially 100% validation accuracy.
      To fix this, we split 10% of `train/` (~8,348 images) into a realistic
      Validation set if `val/` has fewer than 500 images.
    """
    classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    
    train_transform = get_transforms(image_size, is_train=True)
    val_transform = get_transforms(image_size, is_train=False)
    
    full_train_dataset = OCTDataset(data_dir, split='train', classes=classes, use_clahe=use_clahe,
                                    transform=train_transform, max_samples_per_class=max_samples_per_class)
    
    test_dataset = OCTDataset(data_dir, split='test', classes=classes, use_clahe=use_clahe,
                             transform=val_transform, max_samples_per_class=max_samples_per_class)
    
    raw_val_dir = os.path.join(data_dir, 'val')
    raw_val_count = 0
    if os.path.exists(raw_val_dir):
        for cls in classes:
            cpath = os.path.join(raw_val_dir, cls)
            if os.path.exists(cpath):
                raw_val_count += len([f for f in os.listdir(cpath) if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
                
    # If official val directory has < 500 images (like Kermany's 32 images), do a proper 10% split of train
    if raw_val_count < 500:
        val_size = int(len(full_train_dataset) * val_ratio)
        train_size = len(full_train_dataset) - val_size
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=generator)
        # Apply validation transform to val split
        val_dataset.dataset.transform = val_transform
    else:
        train_dataset = full_train_dataset
        val_dataset = OCTDataset(data_dir, split='val', classes=classes, use_clahe=use_clahe,
                                transform=val_transform, max_samples_per_class=max_samples_per_class)

    print(f"\n📊 DataLoader Setup Summary:")
    print(f"   Train samples: {len(train_dataset):,}")
    print(f"   Val samples:   {len(val_dataset):,}")
    print(f"   Test samples:  {len(test_dataset):,}\n")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader
