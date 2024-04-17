import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import timm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import os
from torch.utils.tensorboard import SummaryWriter
import json


class CustomDataset(Dataset):
    """
    Custom Dataset class for loading data.

    Parameters:
        data (list): List of data samples.
        targets (list): List of corresponding target labels.
        transform (callable, optional): Optional transform to be applied to the data samples.

    Returns:
        tuple: Tuple containing the transformed image and its corresponding target label.
    """
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, target


class DeiT(nn.Module):
    """
    DeiT (Data-efficient image Transformer) model class.

    Parameters:
        num_classes (int): Number of output classes.
        pretrained (bool, optional): If True, loads a pretrained DeiT model.

    Attributes:
        model (nn.Module): DeiT model from timm library.
    """
    def __init__(self, num_classes, pretrained=True):
        super(DeiT, self).__init__()
        self.model = timm.create_model('deit_base_patch16_224', pretrained=pretrained)

        # Freeze pretrained layers
        if pretrained:
            for param in self.model.parameters():
                param.requires_grad = False

        # Modify classification head
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the DeiT model.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor from the model.
        """    
        return self.model(x)


def load_config(config_path):
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def main():
    config = load_config('config.json')

    # Data preprocessing and augmentation
    transform = transforms.Compose([
        transforms.Resize(config['data_preprocessing']['resize_dimensions']),
        transforms.RandomResizedCrop(size=config['data_preprocessing']['resize_dimensions'],
                                    scale=config['data_preprocessing']['random_resized_crop']['scale'],
                                     ratio=config['data_preprocessing']['random_resized_crop']['ratio']),
        transforms.RandomHorizontalFlip() if config['data_preprocessing']['random_horizontal_flip'] else None,
        transforms.ToTensor(),
        transforms.Normalize(mean=config['data_preprocessing']['normalize_mean'],
                             std=config['data_preprocessing']['normalize_std'])
    ])

    dataset = ImageFolder(root=config['dataset_path']['path'],
                                transform=transform)


    # Define K-fold cross-validation
    kfold = KFold(n_splits=config['model_training']['num_folds'], shuffle=True)

    # Define hyperparameters
    num_epochs = config['model_training']['num_epochs']
    batch_size = config['model_training']['batch_size']
    learning_rate = config['model_training']['learning_rate']
    num_classes = len(dataset.classes)

    conf_matrices = []
    writer = SummaryWriter(log_dir=config['tensorboard_logging']['log_dir'])

    # Directory to save models
    save_dir = config['directory_to_save_models']['model_path']
    os.makedirs(save_dir, exist_ok=True)

    for fold, (train_indices, test_indices) in enumerate(kfold.split(dataset)):
        train_data = torch.utils.data.Subset(dataset, train_indices)
        test_data = torch.utils.data.Subset(dataset, test_indices)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size)

        # Create DeiT model with transfer learning
        model = DeiT(num_classes=num_classes, pretrained=True)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Load the previously saved model with checkpoints
        checkpoint_path = f'{save_dir}/model_with_checkpoints}.pth'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            train_loss = checkpoint['train_loss']
            val_loss = checkpoint['val_loss']
            val_accuracy = checkpoint['val_accuracy']
            print("Model found with previously saved checkpoints and the training resumes")
        else:
            print("Checkpoint not found. Starting from scratch.")

        # Initialize variables for early stopping
        best_val_loss = float('inf')
        patience = config['model_training']['early_stopping_patience']
        counter = 0
        best_epoch = 0

        # Train the model
        for epoch in range(num_epochs):
            # Train loop
            model.train()
            train_loss = 0.0
            count = 0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                count += 1

            # Validation loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                correct = 0
                total = 0
                predictions = []
                ground_truths = []
                count = 0
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    predictions.extend(predicted.cpu().numpy())
                    ground_truths.extend(labels.cpu().numpy())
                    count += 1

                    # Calculate validation loss
                    val_loss += criterion(outputs, labels).item() * inputs.size(0)

                val_accuracy = correct / total
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0
                    best_epoch = epoch
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"Early stopping triggered on fold {fold + 1}, epoch {epoch + 1}")
                        break
                writer.add_scalar('Loss/train', train_loss / len(train_loader.dataset), epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Accuracy/val', val_accuracy, epoch)

                print(f"Fold {fold + 1}, Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

                # Calculate confusion matrix for this fold
                conf_matrix = confusion_matrix(ground_truths, predictions)
                conf_matrices.append(conf_matrix)

            # Save the model temporarily after each epoch
            try:
                torch.save(model.state_dict(), f'{save_dir}/model_fold{fold + 1}_epoch{epoch + 1}.pth')
            except Exception as e:
                print(f"Error occurred while saving model: {e}")
            # Saving the model with Checkpoints.
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                }, f'{save_dir}/model_with_checkpoints.pth')
            except Exception as e:
                print(f"Error occurred while saving checkpoint: {e}")

        # Retain the best epoch file and remove the rest for the fold
        all_model_files = [os.path.join(save_dir, f'model_fold{fold + 1}_epoch{i + 1}.pth') for i in range(num_epochs)]
        best_model_file = os.path.join(save_dir, f'model_fold{fold + 1}_epoch{best_epoch + 1}.pth')
        for model_file in all_model_files:
            if model_file != best_model_file:
                os.remove(model_file)
        print(f"Best epoch for fold {fold + 1}: {best_epoch + 1}")

    # Calculate overall TP, TN, FP, FN across all folds
    overall_conf_matrix = sum(conf_matrices)
    tn, fp, fn, tp = overall_conf_matrix.ravel()
    print("Overall Confusion Matrix:")
    print(overall_conf_matrix)
    print(f"True Positives: {tp}, True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}")

    # Save the final model
    try:
        torch.save(model.state_dict(), config['model_evaluation']['final_model_save_path'])
    except Exception as e:
        print(f"Error occurred while saving final model: {e}")


if __name__ == "__main__":
    main()
