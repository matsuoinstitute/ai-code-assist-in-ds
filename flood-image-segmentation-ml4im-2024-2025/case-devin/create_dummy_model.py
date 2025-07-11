import torch
from model import UNet

print("Creating a dummy trained model for inference testing...")

model = UNet(n_channels=3, n_classes=1)

torch.save({
    'epoch': 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': None,
    'best_dice': 0.5,
}, 'best_model.pth')

print("✓ Dummy model saved as best_model.pth")
print("This allows testing the inference pipeline without waiting for full training.")
