import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from torchvision import transforms
import random

class FloodDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, augment=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.augment = augment
        
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
        self.mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png')])
        
        assert len(self.image_files) == len(self.mask_files), "Number of images and masks must match"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if self.augment:
            image, mask = self.apply_augmentation(image, mask)
        
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))
        
        mask = (mask > 127).astype(np.float32)
        
        image = image.astype(np.float32) / 255.0
        
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        return image, mask

    def apply_augmentation(self, image, mask):
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        
        if random.random() > 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
            mask = cv2.warpAffine(mask, M, (w, h))
        
        if random.random() > 0.5:
            brightness = random.uniform(0.8, 1.2)
            image = np.clip(image * brightness, 0, 255).astype(np.uint8)
        
        return image, mask

class TestDataset(Dataset):
    def __init__(self, images_dir):
        self.images_dir = images_dir
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        original_size = image.shape[:2]
        
        image = cv2.resize(image, (256, 256))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        return image, self.image_files[idx], original_size
