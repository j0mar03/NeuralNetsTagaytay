from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torchvision.models import resnet18, ResNet18_Weights
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
train_path = r'D:\neuralNets\Day 2\Dataset\PetImages'  # Use raw string to avoid escape characters

# Load the full dataset using ImageFolder
full_dataset = datasets.ImageFolder(train_path, transform=transform)

# Split into training (80%) and validation (20%)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create DataLoader objects to load the data in batches
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize the model, loss function, and optimizer
#model = resnet18(pretrained=True)#SimpleCNN()
model = SimpleCNN()
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
# Function to train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=3):
    train_loss = []
    train_accuracy = []

    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            predicted = (outputs.squeeze() > 0.5).float()  # Apply threshold of 0.5
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        accuracy = correct / total * 100
        train_loss.append(epoch_loss)
        train_accuracy.append(accuracy)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return train_loss, train_accuracy

# Function to validate the model
def validate_model(model, val_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())
            running_loss += loss.item()

            predicted = (outputs.squeeze() > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / len(val_loader)
    accuracy = correct / total * 100
    print(f"validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Train and validate the model
num_epochs = 15
train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer, num_epochs)
validate_model(model, val_loader, criterion)

# Plotting the training loss and accuracy
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_loss)
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_accuracy)
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.show()


# Function to predict and display an image
'''
def predict_image(image_path):
    # Load the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Transform and add batch dimension

    # Get the model's prediction
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        output = model(image)
        prediction = 'cat' if output.item() < 0.5 else 'dog'  # Thresholding at 0.5

    # Display the image and print the prediction
    plt.imshow(Image.open(image_path))
    plt.axis('off')  # Turn off axes for better visualization
    plt.show()
    print(f"This is a {prediction}.")
    '''
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