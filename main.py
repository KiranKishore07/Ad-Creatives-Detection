import torch
import torchvision.transforms as transforms
from PIL import Image
import timm
import torch.nn as nn

# Define the class labels
class_labels = ['ad_creatives', 'non_ad_creatives']

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform = transforms.Compose([
    transforms.Resize(config['data_preprocessing']['resize_dimensions']),
    transforms.ToTensor(),
    transforms.Normalize(mean=config['data_preprocessing']['normalize_mean'],
                         std=config['data_preprocessing']['normalize_std'])
])


class DeiT(nn.Module):
    """
    Custom DeiT (Data-efficient image Transformer) model class.
    """
    def __init__(self, num_classes, pretrained=True):
        """
        Initialize the DeiT model with the specified number of classes and pretrained option.

        Args:
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to load pretrained weights (default: True).
        """
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

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)

# Load the pretrained model
model = DeiT(num_classes=len(class_labels), pretrained=False)
best_model = config['best_model_path']['path']
model.load_state_dict(torch.load(best_model, map_location=torch.device('cpu')))
model.eval()

# Load and preprocess the image
image = Image.open(config['model_predictions']['image_path'])
image_tensor = transform(image).unsqueeze(0)

# Perform inference
with torch.no_grad():
    output = model(image_tensor)

# Get the predicted class label and confidence scores
probabilities = torch.softmax(output, dim=1)[0]
predicted_prob, predicted_idx = torch.max(probabilities, 0)
predicted_label = class_labels[predicted_idx.item()]

print('Predicted label:', predicted_label)
print('Confidence score:', predicted_prob.item())
