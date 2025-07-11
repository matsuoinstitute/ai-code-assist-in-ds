import cv2
import numpy as np
import os

img_path = 'Training/Images/train_aug_0_1456.png'
mask_path = 'Training/Masks/train_mask_aug_0_1456.png'

img = cv2.imread(img_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

print(f'Training image shape: {img.shape}')
print(f'Training mask shape: {mask.shape}')
print(f'Mask unique values: {np.unique(mask)}')

test_img_path = 'Testing/Images/test_001.png'
test_img = cv2.imread(test_img_path)
print(f'Test image shape: {test_img.shape}')

train_images = len(os.listdir('Training/Images'))
train_masks = len(os.listdir('Training/Masks'))
test_images = len(os.listdir('Testing/Images'))

print(f'Training images: {train_images}')
print(f'Training masks: {train_masks}')
print(f'Test images: {test_images}')
