import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from feedforward import ClusteringModel

# Load the dataset
data = pd.read_csv("res/processed_data.csv")

# Split data into features and labels
X = data.drop('LABEL', axis=1).values
y = data['LABEL'].values

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # Using long for class indices
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize the model, loss function, and optimizer
input_dim = X_train.shape[1]  # Number of features
output_dim = len(label_encoder.classes_)  # Number of classes

model = ClusteringModel(input_dim, output_dim)
criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)



# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    total_loss = 0
    
    for inputs, targets in train_loader:
        optimizer.zero_grad()  # Clear gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, targets)  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        
        total_loss += loss.item()
    
    average_loss = total_loss / len(train_loader)
    
    # Print training progress
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}")



def evaluate_accuracy(model, X_test_tensor, y_test_tensor):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(X_test_tensor)  # Get model outputs
        _, predicted = torch.max(outputs, dim=1)  # Get predicted classes
        accuracy = (predicted == y_test_tensor).float().mean()  # Calculate accuracy
    return accuracy.item()

# Calculate accuracy on the test set
accuracy = evaluate_accuracy(model, X_test_tensor, y_test_tensor)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

