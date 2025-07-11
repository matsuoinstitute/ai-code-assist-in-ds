# Flood Image Segmentation

This project implements an automatic flood segmentation model for aerial RGB images using deep learning. The model is designed to identify flooded areas in aerial photographs and is evaluated using the DICE score between predicted and target segmentations.

## Dataset Structure

- **Training Data**: RGB aerial images with corresponding 0/1 binary masks indicating flooded regions
  - `Training/Images/`: Training images
  - `Training/Masks/`: Corresponding binary masks
- **Test Data**: RGB aerial images without masks
  - `Testing/Images/`: Test images for prediction

## Model Architecture

The implementation uses a U-Net architecture, which is well-suited for image segmentation tasks:

- **Encoder**: Downsampling path with convolutional blocks and max pooling
- **Decoder**: Upsampling path with transposed convolutions and skip connections
- **Loss Function**: Combined Binary Cross-Entropy and DICE loss
- **Optimizer**: Adam optimizer with learning rate scheduling

## Key Features

- **Data Augmentation**: Built-in data augmentation through transforms
- **DICE Score Evaluation**: Primary metric for model performance
- **Visualization**: Training progress plots and prediction samples
- **Flexible Architecture**: Configurable U-Net with different feature sizes

## Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train.py
```

This will:
- Load and split the training data (80% train, 20% validation)
- Train the U-Net model for 50 epochs
- Save the best model based on validation DICE score
- Generate training history plots
- Create sample prediction visualizations
- Generate predictions for all test images

### 3. Output Files

After training, the following files will be generated:

- `best_flood_model.pth`: Trained model weights
- `training_history.png`: Training and validation loss/DICE score plots
- `prediction_samples.png`: Sample predictions on training data
- `predictions/`: Directory containing predictions for all test images

## Model Details

### Architecture
- **Input**: 256x256 RGB images
- **Output**: 256x256 binary masks (probability maps)
- **Features**: [64, 128, 256, 512] channels in encoder layers
- **Activation**: Sigmoid output for probability prediction

### Training Configuration
- **Batch Size**: 8
- **Learning Rate**: 1e-4
- **Epochs**: 50
- **Loss**: Combined BCE + DICE loss (α=0.5)
- **Scheduler**: ReduceLROnPlateau with patience=5

### Data Preprocessing
- **Resize**: All images resized to 256x256
- **Normalization**: ImageNet normalization for RGB images
- **Binary Masks**: Thresholded at 0.5 for binary classification

## Evaluation

The model is evaluated using the DICE coefficient:

```
DICE = (2 * |X ∩ Y|) / (|X| + |Y|)
```

Where X is the predicted mask and Y is the ground truth mask.

## File Structure

```
.
├── flood_segmentation.py    # Main model implementation
├── train.py                 # Training script
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── Training/
│   ├── Images/             # Training images
│   └── Masks/              # Training masks
├── Testing/
│   └── Images/             # Test images
└── predictions/            # Generated predictions (after training)
```

## Technical Implementation

### U-Net Components
- **DoubleConv**: Two consecutive convolution blocks with BatchNorm and ReLU
- **Encoder**: Downsampling with max pooling
- **Decoder**: Upsampling with transposed convolutions
- **Skip Connections**: Feature concatenation between encoder and decoder

### Loss Function
- **Binary Cross-Entropy**: Pixel-wise classification loss
- **DICE Loss**: Overlap-based segmentation loss
- **Combined Loss**: Weighted combination for balanced training

### Data Loading
- **Custom Dataset**: PyTorch Dataset class for image-mask pairs
- **Transforms**: Separate transforms for images and masks
- **Data Splitting**: Train/validation split with stratification

## Performance Optimization

- **GPU Support**: Automatic CUDA detection and usage
- **Batch Processing**: Efficient batch-wise training and inference
- **Memory Management**: Proper tensor device management
- **Progress Tracking**: TQDM progress bars for training monitoring

## Customization

The model can be customized by modifying parameters in `flood_segmentation.py`:

- **Image Size**: Change resize dimensions in transforms
- **Batch Size**: Adjust based on available GPU memory
- **Learning Rate**: Modify optimizer learning rate
- **Architecture**: Change U-Net feature dimensions
- **Loss Weights**: Adjust α parameter in CombinedLoss

## Requirements

- Python 3.7+
- PyTorch 1.9+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM for training
