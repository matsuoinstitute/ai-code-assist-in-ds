#!/usr/bin/env python3
"""
Simple training script for flood segmentation model
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flood_segmentation import train_model, predict_test_images, visualize_predictions

def main():
    print("Starting flood segmentation training...")
    print("=" * 60)

    try:
        # Train the model
        print("Training model...")
        model = train_model()

        # Visualize some predictions
        print("\nVisualizing predictions...")
        visualize_predictions()

        # Generate test predictions
        print("\nGenerating test predictions...")
        predict_test_images()

        print("\n" + "=" * 60)
        print("Training and prediction complete!")
        print("Files generated:")
        print("- best_flood_model.pth (trained model)")
        print("- training_history.png (training plots)")
        print("- prediction_samples.png (sample predictions)")
        print("- predictions/ (test predictions)")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
