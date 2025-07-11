import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class FloodDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, mask_transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.mask_transform = mask_transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
            
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])(mask)
        
        mask = (mask > 0.5).float()
        
        return image, mask

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
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
                x = transforms.functional.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return torch.sigmoid(self.final_conv(x))

def dice_coefficient(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice

def dice_loss(pred, target, smooth=1e-6):
    return 1 - dice_coefficient(pred, target, smooth)

def combined_loss(pred, target, alpha=0.5):
    bce = nn.BCELoss()(pred, target)
    dice = dice_loss(pred, target)
    return alpha * bce + (1 - alpha) * dice

def get_data_paths():
    train_images_dir = "/content/drive/Shareddrives/rd_dsagent/dataset/kaggle_image_task/Training/Images"
    train_masks_dir = "/content/drive/Shareddrives/rd_dsagent/dataset/kaggle_image_task/Training/Masks"
    
    image_paths = sorted(glob.glob(os.path.join(train_images_dir, "*.png")))
    mask_paths = []
    
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        mask_name = img_name.replace("train_aug_", "train_mask_aug_")
        mask_path = os.path.join(train_masks_dir, mask_name)
        if os.path.exists(mask_path):
            mask_paths.append(mask_path)
        else:
            print(f"Warning: Mask not found for {img_name}")
    
    return image_paths[:len(mask_paths)], mask_paths

def train_model():
    print("Loading data paths...")
    image_paths, mask_paths = get_data_paths()
    print(f"Found {len(image_paths)} training images and {len(mask_paths)} masks")
    
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    train_dataset = FloodDataset(train_img_paths, train_mask_paths, transform, mask_transform)
    val_dataset = FloodDataset(val_img_paths, val_mask_paths, transform, mask_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    model = UNet(in_channels=3, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    num_epochs = 20
    best_dice = 0.0
    
    train_losses = []
    val_losses = []
    val_dices = []
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=False)
        for images, masks in train_bar:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = combined_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', leave=False)
            for images, masks in val_bar:
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images)
                loss = combined_loss(outputs, masks)
                dice = dice_coefficient(outputs, masks)
                
                val_loss += loss.item()
                val_dice += dice.item()
                val_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'Dice': f'{dice.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_dices.append(avg_val_dice)
        
        scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Val Dice: {avg_val_dice:.4f}')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(model.state_dict(), '/content/best_flood_model.pth')
            print(f'  New best model saved! Dice: {best_dice:.4f}')
        
        print('-' * 60)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(val_dices, label='Val Dice')
    plt.title('Validation Dice Score')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    sample_img, sample_mask = val_dataset[0]
    model.eval()
    with torch.no_grad():
        pred = model(sample_img.unsqueeze(0).to(device))
        pred = pred.squeeze().cpu().numpy()
    
    plt.imshow(pred, cmap='gray')
    plt.title('Sample Prediction')
    
    plt.tight_layout()
    plt.savefig('/content/training_results.png')
    plt.show()
    
    print(f'Training completed! Best Dice Score: {best_dice:.4f}')
    return model

def predict_test_images():
    print("Loading best model for inference...")
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load('/content/best_flood_model.pth'))
    model.eval()
    
    test_images_dir = "/content/drive/Shareddrives/rd_dsagent/dataset/kaggle_image_task/Testing/Images"
    test_image_paths = sorted(glob.glob(os.path.join(test_images_dir, "*.png")))
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    os.makedirs('/content/predictions', exist_ok=True)
    
    print(f"Processing {len(test_image_paths)} test images...")
    
    with torch.no_grad():
        for img_path in tqdm(test_image_paths, desc="Generating predictions"):
            image = Image.open(img_path).convert('RGB')
            original_size = image.size
            
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            prediction = model(image_tensor)
            prediction = prediction.squeeze().cpu().numpy()
            
            prediction_resized = cv2.resize(prediction, original_size)
            prediction_binary = (prediction_resized > 0.5).astype(np.uint8) * 255
            
            filename = os.path.basename(img_path)
            pred_filename = filename.replace('test_', 'pred_')
            
            cv2.imwrite(os.path.join('/content/predictions', pred_filename), prediction_binary)
    
    print("Predictions saved to /content/predictions/")
    
    sample_idx = 0
    if test_image_paths:
        sample_img_path = test_image_paths[sample_idx]
        sample_pred_path = os.path.join('/content/predictions', 
                                       os.path.basename(sample_img_path).replace('test_', 'pred_'))
        
        if os.path.exists(sample_pred_path):
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            original_img = Image.open(sample_img_path)
            pred_img = Image.open(sample_pred_path)
            
            axes[0].imshow(original_img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            axes[1].imshow(pred_img, cmap='gray')
            axes[1].set_title('Predicted Mask')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig('/content/sample_prediction.png')
            plt.show()

if __name__ == "__main__":
    print("Starting flood segmentation training...")
    model = train_model()
    print("\nStarting test predictions...")
    predict_test_images()
    print("Complete!")