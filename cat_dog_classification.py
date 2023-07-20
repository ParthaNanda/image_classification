# Databricks notebook source
#!pip install torch torchvision

# COMMAND ----------

# %sh
# unzip /cat_dog/test_set.zip -d /dbfs/FileStore/cat_dog/
# unzip /cat_dog/training_set.zip -d /dbfs/FileStore/cat_dog/


# COMMAND ----------

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Define the CNN model
class CatsDogsCNN(nn.Module):
    def __init__(self):
        super(CatsDogsCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 32 * 32, 512)  # Updated size after pooling
        self.fc2 = nn.Linear(512, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)  # Flatten the feature map
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Data transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load the dataset
train_data = datasets.ImageFolder('./cat_dog/training_set/', transform=transform)
test_data = datasets.ImageFolder('./cat_dog/test_set/', transform=transform)



# COMMAND ----------

# Create data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Initialize the model
model = CatsDogsCNN()

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

save_path = '/models/pytorch/'
os.makedirs(save_path, exist_ok=True)  # Create the path if it doesn't exist
model_filename = 'model.pth'
torch.save(model.state_dict(), os.path.join(save_path, model_filename))

# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# COMMAND ----------