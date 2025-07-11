import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='Flood Image Segmentation')
    parser.add_argument('--mode', choices=['train', 'inference', 'both'], default='both',
                        help='Mode to run: train, inference, or both')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    
    args = parser.parse_args()
    
    if args.mode in ['train', 'both']:
        print("Starting training...")
        from train import main as train_main
        train_main(args.epochs, args.lr)
    
    if args.mode in ['inference', 'both']:
        if not os.path.exists('best_model.pth'):
            print("Error: No trained model found. Please run training first.")
            sys.exit(1)
        
        print("Starting inference...")
        from inference import main as inference_main
        inference_main()

if __name__ == '__main__':
    main()
