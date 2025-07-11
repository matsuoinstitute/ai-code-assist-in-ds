import torch
from torch.utils.data import DataLoader
from dataset import TestDataset

print("=== Debugging TestDataset Structure ===")

test_dataset = TestDataset('Testing/Images')
print(f"Dataset length: {len(test_dataset)}")

print("\n1. Testing single item access:")
image, filename, original_size = test_dataset[0]
print(f"Image shape: {image.shape}")
print(f"Filename: {filename}")
print(f"Original size: {original_size}")
print(f"Original size type: {type(original_size)}")

print("\n2. Testing DataLoader batch:")
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
batch_data = next(iter(test_loader))

print(f"Batch data length: {len(batch_data)}")
print(f"Images shape: {batch_data[0].shape}")
print(f"Filenames: {batch_data[1]}")
print(f"Original sizes: {batch_data[2]}")
print(f"Original sizes type: {type(batch_data[2])}")
print(f"Original sizes shape: {batch_data[2].shape if hasattr(batch_data[2], 'shape') else 'No shape'}")

if hasattr(batch_data[2], 'shape'):
    print(f"First original size: {batch_data[2][0]}")
    print(f"First original size type: {type(batch_data[2][0])}")
