from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os


# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 18 * 18, 256)  # Assuming image size 150x150
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 18 * 18)  # Flatten the output from convolutional layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


# Image transformations for data augmentation and normalization
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

# Define the path to your dataset
train_path = r'D:\neuralNets\Repository\ECDS-NeuralNets\Day 3\Datasets\Eggs Classification'

# Load the full dataset using ImageFolder
full_dataset = datasets.ImageFolder(train_path, transform=transform)

# Explicitly set class labels for egg damage classification
full_dataset.classes = ['Damaged', 'Not Damaged']

# Split into training (80%) and validation (20%)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create DataLoader objects to load the data in batches
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize the model, loss function, and optimizer
model = SimpleCNN()
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    train_loss = []
    train_accuracy = []

    # Ensure the model is on the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        for images, labels in train_loader:
            # Move images and labels to the device
            images = images.to(device)
            labels = labels.float().to(device)

            # Reshape labels to match model output
            labels = labels.view(-1, 1)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()

            # Calculate accuracy
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # Compute epoch statistics
        epoch_loss = running_loss / len(train_loader)
        accuracy = correct / total * 100
        train_loss.append(epoch_loss)
        train_accuracy.append(accuracy)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return train_loss, train_accuracy


def validate_model(model, val_loader, criterion):
    # Ensure the model is on the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            # Move images and labels to the device
            images = images.to(device)
            labels = labels.float().to(device)

            # Reshape labels to match model output
            labels = labels.view(-1, 1)

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Compute accuracy
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / len(val_loader)
    accuracy = correct / total * 100
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")
    return val_loss, accuracy


# Train and validate the model
num_epochs = 3
train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer, num_epochs)
val_loss, val_accuracy = validate_model(model, val_loader, criterion)

# Plotting the training loss and accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_loss, label='Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracy, label='Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.tight_layout()
plt.show()


def predict_image(model, image_path):
    # Ensure the model is on the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Open and transform the image
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)

    # Move image to the device
    image_tensor = image_tensor.to(device)

    # Set model to evaluation mode
    model.eval()

    # Predict without calculating gradients
    with torch.no_grad():
        output = model(image_tensor)
        confidence = output.item()
        prediction = 'Damaged' if confidence < 0.5 else 'Not Damaged'

    return prediction, confidence


def predict_folder(model, folder_path):
    predictions = {}
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            prediction, confidence = predict_image(model, file_path)
            predictions[filename] = {
                'class': prediction,
                'confidence': confidence
            }

            # Visualization
            plt.figure(figsize=(8, 6))
            plt.imshow(Image.open(file_path))
            plt.axis('off')
            plt.title(f"Prediction: {prediction}\nConfidence: {confidence:.2f}")
            plt.show()

    return predictions


# Load a folder of test images and predict
test_path = r'D:\neuralNets\Repository\ECDS-NeuralNets\Day 3\Datasets\Test'
predictions = predict_folder(model, test_path)

# Print detailed predictions
for image_name, result in predictions.items():
    print(f"{image_name}: Class: {result['class']}, Confidence: {result['confidence']:.4f}")

# Optional: Save the trained model
torch.save(model.state_dict(), 'egg_damage_classifier.pth')