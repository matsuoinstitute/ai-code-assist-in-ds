import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class FloodDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, mask_transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        if self.transform:
            image = self.transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        # Convert mask to binary (0 or 1)
        mask = (mask > 0.5).float()

        return image, mask

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return torch.sigmoid(self.final_conv(x))

def dice_coefficient(pred, target, smooth=1e-6):
    """Calculate DICE coefficient"""
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return dice

def dice_loss(pred, target, smooth=1e-6):
    """DICE loss function"""
    return 1 - dice_coefficient(pred, target, smooth)

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss_val = dice_loss(pred, target)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss_val

def load_data():
    """Load training data paths"""
    train_image_dir = 'Training/Images'
    train_mask_dir = 'Training/Masks'

    image_paths = []
    mask_paths = []

    for filename in os.listdir(train_image_dir):
        if filename.endswith('.png'):
            image_path = os.path.join(train_image_dir, filename)
            mask_path = os.path.join(train_mask_dir, filename)

            if os.path.exists(mask_path):
                image_paths.append(image_path)
                mask_paths.append(mask_path)

    return image_paths, mask_paths

def train_model():
    # Load data
    image_paths, mask_paths = load_data()
    print(f'Total training samples: {len(image_paths)}')

    # Split data
    train_images, val_images, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Create datasets
    train_dataset = FloodDataset(train_images, train_masks, train_transform, mask_transform)
    val_dataset = FloodDataset(val_images, val_masks, train_transform, mask_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

    # Initialize model
    model = UNet(in_channels=3, out_channels=1).to(device)

    # Loss and optimizer
    criterion = CombinedLoss(alpha=0.5)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # Training loop
    num_epochs = 50
    best_val_dice = 0
    train_losses = []
    val_losses = []
    val_dice_scores = []

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

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dice_scores.append(val_dice)

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train DICE: {train_dice:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val DICE: {val_dice:.4f}')
        print('-' * 50)

        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), 'best_flood_model.pth')
            print(f'New best model saved with DICE: {best_val_dice:.4f}')

        scheduler.step(val_loss)

    # Plot training history
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(val_dice_scores, label='Val DICE Score')
    plt.title('Validation DICE Score')
    plt.xlabel('Epoch')
    plt.ylabel('DICE Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

    return model

def predict_test_images():
    """Generate predictions for test images"""
    # Load best model
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load('best_flood_model.pth', map_location=device))
    model.eval()

    # Test transform
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create predictions directory
    os.makedirs('predictions', exist_ok=True)

    test_dir = 'Testing/Images'
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.png')])

    with torch.no_grad():
        for filename in tqdm(test_files, desc='Generating predictions'):
            # Load and preprocess image
            image_path = os.path.join(test_dir, filename)
            image = Image.open(image_path).convert('RGB')
            original_size = image.size

            # Transform for model
            image_tensor = test_transform(image).unsqueeze(0).to(device)

            # Predict
            prediction = model(image_tensor)
            prediction = prediction.squeeze().cpu().numpy()

            # Resize back to original size
            prediction_pil = Image.fromarray((prediction * 255).astype(np.uint8))
            prediction_pil = prediction_pil.resize(original_size, Image.NEAREST)

            # Save prediction
            prediction_path = os.path.join('predictions', filename)
            prediction_pil.save(prediction_path)

    print(f'Predictions saved to predictions/ directory')

def visualize_predictions(num_samples=5):
    """Visualize some predictions"""
    # Load model
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load('best_flood_model.pth', map_location=device))
    model.eval()

    # Load some training data for visualization
    image_paths, mask_paths = load_data()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    plt.figure(figsize=(15, num_samples * 3))

    with torch.no_grad():
        for i in range(min(num_samples, len(image_paths))):
            # Load image and mask
            image = Image.open(image_paths[i]).convert('RGB')
            mask = Image.open(mask_paths[i]).convert('L')

            # Transform
            image_tensor = transform(image).unsqueeze(0).to(device)
            mask_tensor = mask_transform(mask)

            # Predict
            prediction = model(image_tensor).squeeze().cpu()

            # Denormalize image for visualization
            image_vis = image_tensor.squeeze().cpu()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_vis = image_vis * std + mean
            image_vis = torch.clamp(image_vis, 0, 1)

            # Plot
            plt.subplot(num_samples, 4, i*4 + 1)
            plt.imshow(image_vis.permute(1, 2, 0))
            plt.title('Original Image')
            plt.axis('off')

            plt.subplot(num_samples, 4, i*4 + 2)
            plt.imshow(mask_tensor.squeeze(), cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')

            plt.subplot(num_samples, 4, i*4 + 3)
            plt.imshow(prediction, cmap='gray')
            plt.title('Prediction')
            plt.axis('off')

            plt.subplot(num_samples, 4, i*4 + 4)
            plt.imshow((prediction > 0.5).float(), cmap='gray')
            plt.title('Binary Prediction')
            plt.axis('off')

    plt.tight_layout()
    plt.savefig('prediction_samples.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Starting flood segmentation training...")

    # Train the model
    model = train_model()

    # Visualize some predictions
    print("Visualizing predictions...")
    visualize_predictions()

    # Generate test predictions
    print("Generating test predictions...")
    predict_test_images()

    print("Training and prediction complete!")
