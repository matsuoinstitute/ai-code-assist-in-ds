import torch
from model import UNet
from dataset import FloodDataset
print('Testing model instantiation...')
model = UNet(n_channels=3, n_classes=1)
print(f'Model created successfully with {sum(p.numel() for p in model.parameters())} parameters')

print('Testing dataset loading...')
dataset = FloodDataset('Training/Images', 'Training/Masks', augment=False)
print(f'Dataset loaded with {len(dataset)} samples')

image, mask = dataset[0]
print(f'Sample image shape: {image.shape}')
print(f'Sample mask shape: {mask.shape}')
print('All components working correctly!')
