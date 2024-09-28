import sqlite3
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import Tuple, List, Optional
from sklearn.preprocessing import LabelEncoder
from feedforward import FeedForwardNet
from states import Actions, FocalLoss
from db import Database

db = Database('records.db')
data = db.fetch_all()
db.close_conn()

# Step 3: Convert data to NumPy arrays
data = np.array(data)

# Extract inputs and labels
inputs = data[:, 1:-1]
labels = data[:, -1]

# Step 4: Convert to PyTorch tensors
inputs = torch.from_numpy(inputs.astype(np.float32))
labels = torch.from_numpy(labels.astype(np.long))

action = Actions(inputs, labels)

# Step 6: Split dataset
train_size = int(0.9 * len(action.dataset))
test_size = len(action.dataset) - train_size
train_dataset, test_dataset = random_split(action.dataset, [train_size, test_size])


# Step 7: Create DataLoaders
batch_size = 512

train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Assuming 'labels' is your labels tensor
num_classes = len(torch.unique(labels))

# Create an instance of the model
model = FeedForwardNet(input_size=3, hidden_sizes=[64, 128], num_classes=num_classes)


# Assume 'labels_encoded' is your numpy array of labels
classes, class_counts = np.unique(labels, return_counts=True)

#criterion = nn.CrossEntropyLoss(weight=class_weights)
class_counts = torch.tensor(class_counts, dtype=torch.float)

# Calculate class weights inversely proportional to class counts
class_weights = 1.0 / class_counts

# Normalize the weights (optional)
class_weights = class_weights / class_weights.sum()
criterion = nn.CrossEntropyLoss(weight=class_weights)
#criterion = FocalLoss(alpha=class_weights, gamma=2, reduction='mean')

optimizer = optim.SGD(model.parameters(), lr=0.001)

# Define the evaluation function

# Train the model
num_epochs: int = 400  # Adjust as needed
Actions.train_model(model, train_loader, criterion, optimizer, num_epochs)

# Evaluate the model
# Get class names from the label encoder
class_names = ["0","1","2","3"]
accuracy, cm = Actions.evaluate_model(model, test_loader, class_names=class_names)

# Save the model
#torch.save(model.state_dict(), 'feedfwdnet.pth')

# Load the model
#model = SimpleFeedforwardNet(input_size=inputs.shape[1], hidden_sizes=[15, 20], num_classes=num_classes)
#model.load_state_dict(torch.load('model.pth'))
#model.eval()