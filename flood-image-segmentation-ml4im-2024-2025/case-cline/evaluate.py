#!/usr/bin/env python3
"""
Evaluation script to calculate DICE scores for flood segmentation predictions
"""

import os
import numpy as np
from PIL import Image
import torch
from flood_segmentation import dice_coefficient

def calculate_dice_score(pred_path, target_path, threshold=0.5):
    """Calculate DICE score between prediction and target mask"""
    # Load images
    pred = Image.open(pred_path).convert('L')
    target = Image.open(target_path).convert('L')

    # Convert to numpy arrays
    pred_array = np.array(pred) / 255.0  # Normalize to [0, 1]
    target_array = np.array(target) / 255.0  # Normalize to [0, 1]

    # Apply threshold to prediction
    pred_binary = (pred_array > threshold).astype(np.float32)
    target_binary = (target_array > threshold).astype(np.float32)

    # Convert to tensors
    pred_tensor = torch.from_numpy(pred_binary)
    target_tensor = torch.from_numpy(target_binary)

    # Calculate DICE
    dice = dice_coefficient(pred_tensor, target_tensor)

    return dice.item()

def evaluate_predictions(pred_dir='predictions', target_dir='Training/Masks'):
    """Evaluate all predictions against ground truth masks"""

    if not os.path.exists(pred_dir):
        print(f"Prediction directory '{pred_dir}' not found!")
        print("Please run training first to generate predictions.")
        return

    if not os.path.exists(target_dir):
        print(f"Target directory '{target_dir}' not found!")
        return

    # Get all prediction files
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.png')]

    if len(pred_files) == 0:
        print("No prediction files found!")
        return

    dice_scores = []
    valid_predictions = 0

    print(f"Evaluating {len(pred_files)} predictions...")
    print("-" * 50)

    for pred_file in sorted(pred_files):
        pred_path = os.path.join(pred_dir, pred_file)
        target_path = os.path.join(target_dir, pred_file)

        if os.path.exists(target_path):
            try:
                dice = calculate_dice_score(pred_path, target_path)
                dice_scores.append(dice)
                valid_predictions += 1
                print(f"{pred_file}: DICE = {dice:.4f}")
            except Exception as e:
                print(f"Error processing {pred_file}: {e}")
        else:
            print(f"Target mask not found for {pred_file}")

    if len(dice_scores) > 0:
        mean_dice = np.mean(dice_scores)
        std_dice = np.std(dice_scores)
        min_dice = np.min(dice_scores)
        max_dice = np.max(dice_scores)

        print("-" * 50)
        print(f"Evaluation Results:")
        print(f"Valid predictions: {valid_predictions}")
        print(f"Mean DICE Score: {mean_dice:.4f} ± {std_dice:.4f}")
        print(f"Min DICE Score: {min_dice:.4f}")
        print(f"Max DICE Score: {max_dice:.4f}")
        print("-" * 50)

        # Save results
        with open('evaluation_results.txt', 'w') as f:
            f.write("Flood Segmentation Evaluation Results\n")
            f.write("=" * 40 + "\n")
            f.write(f"Valid predictions: {valid_predictions}\n")
            f.write(f"Mean DICE Score: {mean_dice:.4f} ± {std_dice:.4f}\n")
            f.write(f"Min DICE Score: {min_dice:.4f}\n")
            f.write(f"Max DICE Score: {max_dice:.4f}\n")
            f.write("\nIndividual Scores:\n")
            f.write("-" * 20 + "\n")

            for i, (pred_file, dice) in enumerate(zip(sorted(pred_files), dice_scores)):
                if i < len(dice_scores):
                    f.write(f"{pred_file}: {dice:.4f}\n")

        print("Results saved to 'evaluation_results.txt'")
    else:
        print("No valid predictions found for evaluation!")

def evaluate_validation_set():
    """Evaluate model on validation set using saved model"""
    from flood_segmentation import UNet, load_data, FloodDataset
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    from sklearn.model_selection import train_test_split

    # Check if model exists
    if not os.path.exists('best_flood_model.pth'):
        print("Trained model not found! Please run training first.")
        return

    print("Evaluating model on validation set...")

    # Load data
    image_paths, mask_paths = load_data()
    _, val_images, _, val_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Dataset and loader
    val_dataset = FloodDataset(val_images, val_masks, transform, mask_transform)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load('best_flood_model.pth', map_location=device))
    model.eval()

    # Evaluate
    dice_scores = []

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            # Calculate DICE for each sample in batch
            for i in range(outputs.size(0)):
                dice = dice_coefficient(outputs[i], masks[i])
                dice_scores.append(dice.item())

    mean_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)

    print(f"Validation Set Results:")
    print(f"Samples: {len(dice_scores)}")
    print(f"Mean DICE Score: {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"Min DICE Score: {np.min(dice_scores):.4f}")
    print(f"Max DICE Score: {np.max(dice_scores):.4f}")

if __name__ == "__main__":
    print("Flood Segmentation Evaluation")
    print("=" * 40)

    # Evaluate validation set if model exists
    evaluate_validation_set()

    print("\n" + "=" * 40)

    # Evaluate predictions if they exist
    evaluate_predictions()
