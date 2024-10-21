import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

# Load the model
model = torchvision.models.resnet18(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Preprocess the image
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the image
image = Image.open("image.jpg")
image = transform(image)

# Add an extra dimension to the image
image = image.unsqueeze(0)

# Run the image through the model
output = model(image)

# Get the predicted class probabilities
probabilities = torch.softmax(output, dim=1)

# Get the predicted class
predicted_class = torch.argmax(probabilities, dim=1)
print(f"Predicted class: {predicted_class.item()}")
