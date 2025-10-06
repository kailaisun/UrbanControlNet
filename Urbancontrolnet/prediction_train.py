import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from PIL import Image
import os
import json
import pandas as pd
import re



os.environ["CUDA_VISIBLE_DEVICES"] = "2"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the dataset class
class ImageRegressionDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = []
        self.transform = transform
        # self.max_values = [135.9, 5379.6, 38447.0]
        self.max_values = [1550.61, 1000, 95.1,37.83]
        with open(data_dir, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)



    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        image = Image.open(target_filename).convert("RGB")

        labels = re.findall(r'\b\d+\.\d+|\b\d+\b', prompt)

      
        labels = [float(num) / self.max_values[i] for i, num in enumerate(labels)]


        labels = torch.tensor(labels, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image,labels


# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def calculate_min_max_per_position(data_dir):
    dataset = ImageRegressionDataset(data_dir)
    all_numbers = []

    for idx in range(len(dataset)):
        numbers = dataset[idx]
        all_numbers.append(numbers)

 
    all_numbers_transposed = list(zip(*all_numbers))

    
    min_max_per_position = [(min(position), max(position)) for position in all_numbers_transposed]

    return min_max_per_position


# min_max_values = calculate_min_max_per_position(data_dir)
# for i, (min_val, max_val) in enumerate(min_max_values):
#     print(f"Position {i+1}: Min Value = {min_val}, Max Value = {max_val}")

# Dataset and DataLoader
data_dir = './urban_data/tencities/train.json' # CSV with image names and regression labels
dataset = ImageRegressionDataset(data_dir, transform=transform)

# Split dataset into training and validation sets
val_split = int(0.3 * len(dataset))
train_split = len(dataset) - val_split
train_dataset, val_dataset = random_split(dataset, [train_split, val_split])

train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# Load pre-trained ResNet50 model
model = models.resnet50(pretrained=True)

# Modify the fully connected layer for regression
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 4),
    nn.Sigmoid()  # Add Sigmoid activation function
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss/len(train_dataloader):.4f}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss/len(val_dataloader):.4f}")

# Save the trained model
torch.save(model.state_dict(), "resnet50_regression_four_metric.pth")

print("Training complete!")
