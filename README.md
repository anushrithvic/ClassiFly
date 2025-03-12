# ClassiFly

## Overview
ClassiFly is a deep learning-based image classification system utilizing EfficientNetB0 for accurate and efficient categorization of images. It supports data augmentation, transfer learning, and fine-tuning for optimal performance.

## Features
- Supports multiple classification algorithms with EfficientNetB0
- Image preprocessing and real-time data augmentation
- Model training with early stopping and learning rate reduction
- Fine-tuning support for improved accuracy 
- Generates predictions with class label mapping
- Saves best model for reuse and future inference


## Usage
1. Launch the script to train the model:
   ```bash
   python script.py
   ```
2. Enable fine-tuning if needed:
   ```bash
   python script.py --fine_tune
   ```
3. View predictions after training:
   ```bash
   cat submission.csv
   ```

## Technologies Used
- Python
- TensorFlow & Keras
- EfficientNetB0 (Transfer Learning)
- Scikit-learn
- Pandas & NumPy (for data processing)
- Matplotlib & Seaborn (for data visualization)

