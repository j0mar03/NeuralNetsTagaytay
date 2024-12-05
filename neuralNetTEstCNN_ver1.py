from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
import os

# Data transformations
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomCrop((140, 140)),
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset path
train_path = r'D:\neuralNets\Day 2\Dataset\PetImages'

# Load dataset
full_dataset = datasets.ImageFolder(train_path, transform=transform)

# Verify dataset and filter problematic files
def validate_dataset(dataset):
    valid_indices = []
    for idx, (img, label) in enumerate(dataset):
        try:
            img = dataset[idx][0]  # Attempt to load the image
            valid_indices.append(idx)
        except Exception as e:
            print(f"Skipping index {idx} due to error: {e}")
    return torch.utils.data.Subset(dataset, valid_indices)

full_dataset = validate_dataset(full_dataset)

# Split dataset into training and validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Use ResNet18 with new weights parameter
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 1),
    nn.Sigmoid()
)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=3):
    device = 'cpu'
    model = model.to(device)
    train_loss, val_loss, train_acc, val_acc = [], [], [], []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        correct, total, running_loss = 0, 0, 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss.append(running_loss / len(train_loader))
        train_acc.append(correct / total * 100)

        # Validation phase
        model.eval()
        val_correct, val_total, val_running_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels.float())
                val_running_loss += loss.item()
                predicted = (outputs.squeeze() > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss.append(val_running_loss / len(val_loader))
        val_acc.append(val_correct / val_total * 100)
        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss[-1]:.4f}, Train Accuracy: {train_acc[-1]:.2f}%")
        print(f"  Val Loss: {val_loss[-1]:.4f}, Val Accuracy: {val_acc[-1]:.2f}%")

    return train_loss, val_loss, train_acc, val_acc

# Train the model
train_loss, val_loss, train_acc, val_acc = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10)

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_acc, label='Train Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.show()

image_path = r'D:\neuralNets\Day 2\Dataset\test'

def predict_image(model, image_path):
    # Load the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Transform and add batch dimension

    # Get the model's prediction
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        output = model(image)
        prediction = 'cat' if output.item() < 0.5 else 'dog'  # Threshold at 0.5

    return prediction


# Iterate over all images in the folder
def predict_folder(model, folder_path):
    predictions = {}
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Ensure the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            prediction = predict_image(model, file_path)
            predictions[filename] = prediction

            # Optionally display the image and prediction
            plt.imshow(Image.open(file_path))
            plt.axis('off')
            plt.title(f"Prediction: {prediction}")
            plt.show()

    return predictions


# Load your trained model (assuming it is already trained)
# model = SimpleCNN()  # Uncomment if you're using your SimpleCNN
# model.load_state_dict(torch.load('path_to_your_model.pth'))  # Load your saved model

# Predict all images in the folder
predictions = predict_folder(model, image_path)

# Print the predictions
for image_name, prediction in predictions.items():
    print(f"{image_name}: {prediction}")

# Example usage: Change the path to the image you want to test

#predict_image(image_path)