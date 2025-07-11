import torch
import cv2
import numpy as np
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model import UNet
from dataset import TestDataset

def load_model(model_path, device):
    model = UNet(n_channels=3, n_classes=1).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def predict_test_images(model, test_loader, device, output_dir='predictions'):
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            images = batch_data[0].to(device)
            filenames = batch_data[1]
            original_sizes = batch_data[2]
            
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            
            batch_size = outputs.size(0)
            for i in range(batch_size):
                pred = outputs[i].cpu().numpy().squeeze()
                filename = filenames[i]
                
                if i < len(original_sizes):
                    if isinstance(original_sizes[i], torch.Tensor):
                        size_tensor = original_sizes[i]
                        if size_tensor.numel() >= 2:
                            original_h, original_w = size_tensor[0].item(), size_tensor[1].item()
                        else:
                            original_h, original_w = 512, 512
                    else:
                        original_h, original_w = original_sizes[i]
                else:
                    original_h, original_w = 512, 512
                
                pred_resized = cv2.resize(pred, (original_w, original_h))
                
                pred_binary = (pred_resized > 0.5).astype(np.uint8) * 255
                
                output_path = os.path.join(output_dir, f'pred_{filename}')
                cv2.imwrite(output_path, pred_binary)
                
                predictions.append({
                    'filename': filename,
                    'prediction': pred_binary,
                    'prediction_prob': pred_resized
                })
    
    return predictions

def visualize_predictions(test_images_dir, predictions, output_dir='visualizations', num_samples=10):
    os.makedirs(output_dir, exist_ok=True)
    
    for i, pred_info in enumerate(predictions[:num_samples]):
        filename = pred_info['filename']
        prediction = pred_info['prediction']
        
        original_image = cv2.imread(os.path.join(test_images_dir, filename))
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(original_image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(prediction, cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(original_image)
        plt.imshow(prediction, alpha=0.5, cmap='Blues')
        plt.title('Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'visualization_{filename}'))
        plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    if not os.path.exists('best_model.pth'):
        print("Error: No trained model found. Please run train.py first.")
        return
    
    model = load_model('best_model.pth', device)
    print("Model loaded successfully!")
    
    test_dataset = TestDataset('Testing/Images')
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    print(f'Processing {len(test_dataset)} test images...')
    
    predictions = predict_test_images(model, test_loader, device)
    
    print(f'Predictions saved to predictions/ directory')
    
    visualize_predictions('Testing/Images', predictions)
    print(f'Visualizations saved to visualizations/ directory')
    
    print("Inference completed successfully!")

if __name__ == '__main__':
    main()
