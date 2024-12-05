import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


# Load the dataset
desired_path = "D:/OneDrive - Polytechnic University of the Philippines/2024_Feb/Designation/ITech Designee/2024 Events/12-02_06_Neural_Networks/ECDS-NeuralNets/Day 1/Practice_Dataset"
data = pd.read_csv(f'{desired_path}/predictive_maintenance.csv')

# Preprocess the dataset
data.drop('UDI', axis=1, inplace=True)  # Drop the 'UDI' column
X = data.drop('Failure Type', axis=1)  # Features
y = data['Failure Type']  # Target variable

# Check for non-numeric columns and encode them
X = pd.get_dummies(X)  # Convert categorical to numeric if needed

# Handle missing values if any
X = X.fillna(0)  # Replace NaNs with 0

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.Tensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_val_tensor = torch.Tensor(X_val)
y_val_tensor = torch.LongTensor(y_val)

# Define the Neural Network
class PredictiveMaintenanceNN(nn.Module):
    def __init__(self, input_size):
        super(PredictiveMaintenanceNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 36)
        self.fc2 = nn.Linear(36, 24)
        self.fc3 = nn.Linear(24, len(label_encoder.classes_))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x  # Remove softmax here, as CrossEntropyLoss applies it internally
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_encoded),
    y=y_encoded
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

# Use the weights in CrossEntropyLoss
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
# Initialize the model, loss function, and optimizer
input_size = X.shape[1]
model = PredictiveMaintenanceNN(input_size)
#criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training the model
losses = []
accuracies = []
for epoch in range(100):
    model.train()

    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    losses.append(loss.item())

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Calculate accuracy
    with torch.no_grad():
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(y_train_tensor.numpy(), predicted.numpy())
        accuracies.append(accuracy)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}, Accuracy: {accuracy * 100:.2f}%')

# Plot Loss vs Epoch
plt.figure(figsize=(10, 5))
plt.plot(range(1, 101), losses, label='Training Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curve (Loss vs Epoch)')
plt.legend()
plt.grid()
plt.show()

# Plot Accuracy vs Epoch
plt.figure(figsize=(10, 5))
plt.plot(range(1, 101), accuracies, label='Training Accuracy', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy vs Epoch')
plt.legend()
plt.grid()
plt.show()

# Evaluate the model
with torch.no_grad():
    model.eval()
    y_pred_tensor = model(X_val_tensor)
    _, y_pred = torch.max(y_pred_tensor, 1)

# Convert predictions and true labels to numpy arrays
y_pred = y_pred.numpy()
y_val = y_val_tensor.numpy()

# Confusion Matrix
cm = confusion_matrix(y_val, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification Report
report = classification_report(y_val, y_pred, target_names=label_encoder.classes_)
print("\nClassification Report:")
print(report)

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
