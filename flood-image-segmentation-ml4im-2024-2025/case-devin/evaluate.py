import torch
import numpy as np

def dice_coefficient(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()

def dice_coefficient_numpy(pred, target, smooth=1e-6):
    pred = (pred > 0.5).astype(np.float32)
    target = target.astype(np.float32)
    
    pred = pred.flatten()
    target = target.flatten()
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice

def evaluate_model(model, dataloader, device):
    model.eval()
    total_dice = 0
    num_samples = 0
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            for i in range(outputs.size(0)):
                dice = dice_coefficient(outputs[i], masks[i])
                total_dice += dice
                num_samples += 1
    
    return total_dice / num_samples if num_samples > 0 else 0
