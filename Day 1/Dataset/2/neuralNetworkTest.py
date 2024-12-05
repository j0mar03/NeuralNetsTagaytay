# import kagglehub
# import os
# import shutil

# # Define the path where you want to download the dataset
#file_path = r"D:\OneDrive - Polytechnic University of the Philippines\2024_Feb\Designation\ITech Designee\2024 Events\12-02_06_Neural_Networks\ECDS-NeuralNets\Day 1\Dataset\2\Iris.csv"
desired_path = "D:/OneDrive - Polytechnic University of the Philippines/2024_Feb/Designation/ITech Designee/2024 Events/12-02_06_Neural_Networks/ECDS-NeuralNets/Day 1/Dataset"
# # Download laval version
# default_path = kagglehub.dataset_download("uciml/iris")

# # Move the downloaded files to your desired directory
# shutil.move(default_path, desired_path)

# print("Path to dataset files:", desired_path)
import pandas as pd
iris_data = pd.read_csv(f'{desired_path}/2/Iris.csv')
iris_data
iris_data['Species'].unique()
iris_data.drop('Id', axis=1, inplace=True)
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Define the Simple Neural Network
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        # Input layer (4 features) -> Hidden layer (36 neurons)
        self.fc1 = nn.Linear(input_size, 36)
        # Hidden layer (36 neurons) -> Hidden layer (24 neurons)
        self.fc2 = nn.Linear(36, 24)
        # Hidden layer (24 neurons) -> Output layer (3 classes)
        self.fc3 = nn.Linear(24, 3)
        # ReLU activation function
        self.relu = nn.ReLU()
        # Softmax activation for multi-class classification
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Pass input through the network
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return self.softmax(x)
# Extract features and labels
from sklearn.preprocessing import StandardScaler, LabelEncoder

X = iris_data.drop('Species', axis=1)  # Features (4 columns)
y = iris_data['Species']  # Original string labels

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use LabelEncoder to convert string labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Convert string labels into numeric labels

# Split the dataset into training and valing sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# # Check the shapes of the splits to confirm they match
# print("X_train shape:", X_train.shape)  # Should be (120, 4)
# print("y_train shape:", y_train.shape)  # Should be (120,)
# print("X_val shape:", X_val.shape)    # Should be (30, 4)
# print("y_val shape:", y_val.shape)    # Should be (30,)

# Convert to PyTorch tensors
X_train_tensor = torch.Tensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_val_tensor = torch.Tensor(X_val)
y_val_tensor = torch.LongTensor(y_val)

# # Verify tensor shapes
# print("X_train_tensor shape:", X_train_tensor.shape)  # Should be (120, 4)
# print("y_train_tensor shape:", y_train_tensor.shape)  # Should be (120,)
# print("X_val_tensor shape:", X_val_tensor.shape)    # Should be (30, 4)
# print("y_val_tensor shape:", y_val_tensor.shape)    # Should be (30,)
import matplotlib.pyplot as plt

# Initialize model, loss, and optimizer
input_size = X.shape[1]
model = SimpleNN(input_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Lists to store loss and accuracy
losses = []
accuracies = []

# Training loop
for epoch in range(100):  # Number of epochs
    model.train()  # Set model to training mode

    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    losses.append(loss.item())  # Store the loss

    # Backward pass
    optimizer.zero_grad()  # Zero out gradients
    loss.backward()        # Compute gradients
    optimizer.step()       # Update weights using SGD

    # Calculate accuracy on training data
    with torch.no_grad():
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(y_train_tensor.numpy(), predicted.numpy())
        accuracies.append(accuracy)  # Store accuracy

    # Print loss and accuracy every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}, Accuracy: {accuracy * 100:.2f}%')

# Check if the lists are populated correctly
print(f"Total Losses: {len(losses)}")
print(f"Total Accuracies: {len(accuracies)}")

# Plot Loss vs. Epochs (Learning Curve)
plt.figure(figsize=(10, 5))
plt.plot(range(1, 101), losses, label='Training Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curve (Loss vs. Epoch)')
plt.legend()
plt.grid()
plt.show()

# Plot Accuracy vs. Epochs
plt.figure(figsize=(10, 5))
plt.plot(range(1, 101), accuracies, label='Training Accuracy', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy vs. Epoch')
plt.legend()
plt.grid()
plt.show()
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
# Evaluate the model on the val set
with torch.no_grad():
    model.eval()  # Set the model to evaluation mode
    y_pred_tensor = model(X_val_tensor)
    _, y_pred = torch.max(y_pred_tensor, 1)  # Get the predicted classes

# Convert predictions and true labels to numpy arrays for scikit-learn functions
y_pred = y_pred.numpy()
y_val = y_val_tensor.numpy()

# Confusion Matrix
cm = confusion_matrix(y_val, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification Report
report = classification_report(y_val, y_pred)
print("\nClassification Report:")
print(report)

# Plot the Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()