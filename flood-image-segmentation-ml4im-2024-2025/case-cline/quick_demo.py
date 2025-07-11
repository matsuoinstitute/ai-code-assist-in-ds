#!/usr/bin/env python3
"""
Quick demo script to test the flood segmentation model with a few epochs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from flood_segmentation import (
    UNet, FloodDataset, load_data, dice_coefficient,
    CombinedLoss
)

def quick_demo():
    print("Quick Demo: Flood Segmentation Model")
    print("=" * 50)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    image_paths, mask_paths = load_data()
    print(f"Found {len(image_paths)} training samples")

    if len(image_paths) == 0:
        print("ERROR: No training data found!")
        return

    # Use a small subset for quick demo
    subset_size = min(50, len(image_paths))
    image_paths = image_paths[:subset_size]
    mask_paths = mask_paths[:subset_size]

    print(f"Using {subset_size} samples for quick demo")

    # Split data
    train_images, val_images, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=0.3, random_state=42
    )

    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Smaller size for quick demo
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Create datasets
    train_dataset = FloodDataset(train_images, train_masks, train_transform, mask_transform)
    val_dataset = FloodDataset(val_images, val_masks, train_transform, mask_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Initialize model (smaller for quick demo)
    model = UNet(in_channels=3, out_channels=1, features=[32, 64, 128, 256]).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loss and optimizer
    criterion = CombinedLoss(alpha=0.5)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Quick training loop (just 3 epochs)
    num_epochs = 3
    print(f"\nStarting quick training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_dice = 0

        for images, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dice += dice_coefficient(outputs, masks).item()

        # Validation phase
        model.eval()
        val_loss = 0
        val_dice = 0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item()
                val_dice += dice_coefficient(outputs, masks).item()

        # Calculate averages
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train DICE: {train_dice:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val DICE: {val_dice:.4f}')
        print('-' * 40)

    # Save demo model
    torch.save(model.state_dict(), 'demo_flood_model.pth')
    print("Demo model saved as 'demo_flood_model.pth'")

    print("\n✅ Quick demo completed successfully!")
    print("The model architecture and training pipeline work correctly.")
    print("You can now run 'python train.py' for full training.")

if __name__ == "__main__":
    quick_demo()
