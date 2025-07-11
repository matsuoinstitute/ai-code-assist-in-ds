# Flood Image Segmentation

This project implements an automatic flood image segmentation model using U-Net architecture to identify flooded areas in RGB aerial photographs.

## Dataset

- **Training**: 3,112 RGB aerial images with corresponding binary masks
- **Testing**: 289 test images for inference
- **Evaluation**: DICE score between predicted and target segmentations

## Model Architecture

The model uses U-Net, a convolutional neural network designed for semantic segmentation:
- Encoder-decoder structure with skip connections
- Input: 256x256 RGB images
- Output: Binary segmentation masks (flooded vs non-flooded areas)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training and Inference (Complete Pipeline)
```bash
python main.py --mode both --epochs 50 --lr 1e-4
```

### Training Only
```bash
python main.py --mode train --epochs 50
```

### Inference Only
```bash
python main.py --mode inference
```

### Individual Scripts
```bash
# Training
python train.py

# Inference
python inference.py

# Data inspection
python check_data.py
```

## Output

- **Model**: `best_model.pth` - Best performing model checkpoint
- **Predictions**: `predictions/` - Binary masks for test images
- **Visualizations**: `visualizations/` - Side-by-side comparisons
- **Training curves**: `training_curves.png` - Loss and DICE score plots

## Model Performance

The model is evaluated using DICE coefficient:
```
DICE = 2 * |predicted ∩ target| / (|predicted| + |target|)
```

## File Structure

- `model.py` - U-Net architecture implementation
- `dataset.py` - Data loading and augmentation
- `train.py` - Training pipeline
- `inference.py` - Prediction and visualization
- `evaluate.py` - DICE score calculation
- `main.py` - Main execution script
