import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import UNet
from dataset import FloodDataset
from evaluate import dice_coefficient
import numpy as np

print("=== Quick Test of Flood Segmentation Model ===")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

print("\n1. Testing model instantiation...")
model = UNet(n_channels=3, n_classes=1).to(device)
print(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")

print("\n2. Testing dataset loading...")
dataset = FloodDataset('Training/Images', 'Training/Masks', augment=False)
print(f"✓ Dataset loaded with {len(dataset)} samples")

print("\n3. Testing data loading...")
dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
images, masks = next(iter(dataloader))
print(f"✓ Batch loaded - Images: {images.shape}, Masks: {masks.shape}")

print("\n4. Testing model forward pass...")
model.eval()
with torch.no_grad():
    images = images.to(device)
    masks = masks.to(device)
    outputs = model(images)
    print(f"✓ Forward pass successful - Output: {outputs.shape}")

print("\n5. Testing DICE score calculation...")
dice_score = dice_coefficient(outputs, masks)
print(f"✓ DICE score calculated: {dice_score:.4f}")

print("\n6. Testing loss calculation...")
criterion = nn.BCEWithLogitsLoss()
loss = criterion(outputs, masks)
print(f"✓ Loss calculated: {loss.item():.4f}")

print("\n7. Testing prediction conversion...")
predictions = torch.sigmoid(outputs)
binary_preds = (predictions > 0.5).float()
print(f"✓ Binary predictions shape: {binary_preds.shape}")
print(f"✓ Prediction range: [{predictions.min().item():.3f}, {predictions.max().item():.3f}]")

print("\n=== All Tests Passed! ===")
print("The flood segmentation model implementation is working correctly.")
print("Ready for training and inference on the full dataset.")
