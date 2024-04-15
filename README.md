### Ad-Creatives-Detection
**Image Classification with DeiT**

This project implements image classification using the Data-efficient image Transformer (DeiT) model. It includes data preprocessing, k-fold cross-validation, early stopping, and model saving functionalities.

### Features:
- **Data Preprocessing:** Resize, random crop, horizontal flip, and normalization.
- **Model Training:** K-fold cross-validation with early stopping.
- **Model Architecture:** Utilizes the DeiT model with configurable parameters.
- **Model Evaluation:** Option to use a separate test set and save the final trained model.
- **Tensorboard Logging:** Logs training progress and metrics.
- **Configuration:** Configurable via a JSON file for easy experimentation.
- **Documentation:** Includes detailed documentation for classes and functions.

### Usage:
1. Update the JSON configuration file as needed.
2. Run the main Python script to start model training and evaluation.

### Requirements:
- PyTorch
- torchvision
- timm
- scikit-learn
- tensorboard

### File Structure:
- `main.py`: Main script for model evaluation.
-  `model_train.py`: Main script for training.
- `config.json`: Configuration file for specifying parameters.
- `README.md`: Overview of the project and instructions.

### License:
This project is licensed under the MIT License.
