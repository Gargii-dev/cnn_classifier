import torch
from torchvision import transforms
from PIL import Image
from model import FruitCNN

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = FruitCNN(num_classes=6)
model.load_state_dict(torch.load("fruit_model.pth", map_location=device))
model.to(device)
model.eval()

# Get class names automatically
from torchvision import datasets
dataset = datasets.ImageFolder("dataset")
class_names = dataset.classes

# Transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load test image
image_path = "test.jpg"   # change if needed
img = Image.open(image_path).convert("RGB")
img = transform(img).unsqueeze(0).to(device)

# Predict
import torch.nn.functional as F

with torch.no_grad():
    output = model(img)
    probs = F.softmax(output, dim=1)
    confidence, predicted = torch.max(probs, 1)

print("Prediction:", class_names[predicted.item()])
print("Confidence:", confidence.item())