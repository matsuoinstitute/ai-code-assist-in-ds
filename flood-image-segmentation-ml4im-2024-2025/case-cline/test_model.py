#!/usr/bin/env python3
"""
Quick test script to verify the flood segmentation model works
"""

import torch
import torch.nn as nn
from flood_segmentation import UNet, FloodDataset, load_data, dice_coefficient
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

def test_model():
    print("Testing flood segmentation model...")

    # Test device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Test data loading
    print("\n1. Testing data loading...")
    image_paths, mask_paths = load_data()
    print(f"Found {len(image_paths)} training samples")

    if len(image_paths) == 0:
        print("ERROR: No training data found!")
        return False

    # Test model creation
    print("\n2. Testing model creation...")
    model = UNet(in_channels=3, out_channels=1)
    model = model.to(device)
    print(f"Model created successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    print("\n3. Testing forward pass...")
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

    # Test data loading with transforms
    print("\n4. Testing data transforms...")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Load one sample
    image = Image.open(image_paths[0]).convert('RGB')
    mask = Image.open(mask_paths[0]).convert('L')

    print(f"Original image size: {image.size}")
    print(f"Original mask size: {mask.size}")

    # Apply transforms
    image_tensor = transform(image).unsqueeze(0).to(device)
    mask_tensor = mask_transform(mask).unsqueeze(0).to(device)
    mask_tensor = (mask_tensor > 0.5).float()

    print(f"Transformed image shape: {image_tensor.shape}")
    print(f"Transformed mask shape: {mask_tensor.shape}")
    print(f"Mask unique values: {torch.unique(mask_tensor)}")

    # Test model prediction
    print("\n5. Testing model prediction...")
    with torch.no_grad():
        prediction = model(image_tensor)

    print(f"Prediction shape: {prediction.shape}")
    print(f"Prediction range: [{prediction.min():.4f}, {prediction.max():.4f}]")

    # Test DICE calculation
    print("\n6. Testing DICE calculation...")
    dice_score = dice_coefficient(prediction, mask_tensor)
    print(f"DICE score (random model): {dice_score:.4f}")

    # Test dataset class
    print("\n7. Testing dataset class...")
    dataset = FloodDataset([image_paths[0]], [mask_paths[0]], transform, mask_transform)
    sample_image, sample_mask = dataset[0]
    print(f"Dataset sample image shape: {sample_image.shape}")
    print(f"Dataset sample mask shape: {sample_mask.shape}")

    print("\n✅ All tests passed! Model is ready for training.")
    return True

if __name__ == "__main__":
    success = test_model()
    if success:
        print("\n🚀 You can now run 'python train.py' to start training!")
    else:
        print("\n❌ Please fix the issues before training.")
