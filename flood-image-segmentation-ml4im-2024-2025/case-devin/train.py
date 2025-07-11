import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import UNet
from dataset import FloodDataset
from evaluate import evaluate_model, dice_coefficient

def train_model(model, train_loader, val_loader, device, num_epochs=50, learning_rate=1e-4):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    best_dice = 0
    train_losses = []
    val_dices = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        val_dice = evaluate_model(model, val_loader, device)
        val_dices.append(val_dice)
        
        scheduler.step(val_dice)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_loss:.4f}')
        print(f'  Val DICE: {val_dice:.4f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
            }, 'best_model.pth')
            print(f'  New best model saved! DICE: {best_dice:.4f}')
        
        print('-' * 50)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_dices)
    plt.title('Validation DICE Score')
    plt.xlabel('Epoch')
    plt.ylabel('DICE')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()
    
    return best_dice

def main(num_epochs=50, learning_rate=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    dataset = FloodDataset(
        images_dir='Training/Images',
        masks_dir='Training/Masks',
        augment=True
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    
    model = UNet(n_channels=3, n_classes=1).to(device)
    
    best_dice = train_model(model, train_loader, val_loader, device, num_epochs=num_epochs, learning_rate=learning_rate)
    
    print(f'Training completed! Best validation DICE: {best_dice:.4f}')

if __name__ == '__main__':
    main()
