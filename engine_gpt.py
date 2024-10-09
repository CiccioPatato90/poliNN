import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from db import Database
from states import Actions
import pandas as pd
from feedforward import *
from sklearn.preprocessing import StandardScaler
from evaluate import *

# Step 1: Fetch data from SQLite and convert to NumPy arrays
db = Database('res/records.db')
data = db.fetch_all()  # Fetch all records from the database
db.close_conn()

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(data, columns=['a01', 'a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'a08', 'a09',
                                 'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19', 
                                 'a20', 'a21', 'a22', 'a23', 'a24', 'label'])

# Convert all columns to float
reduced_df = df.apply(pd.to_numeric, errors='coerce')
# Display the first few rows of the DataFrame to confirm the conversion
#print(df.head())

# Run the visualizations
#reduced_df = apply_pca(df, variance_threshold=0.95)
print(reduced_df.head())

# Step 1: Convert the reduced dataset to a NumPy array
reduced_features = reduced_df.drop(columns='label').values
labels = reduced_df['label'].values

# Step 2: Convert to PyTorch tensors
inputs_tensor = torch.from_numpy(reduced_features.astype(np.float32))
labels_tensor = torch.from_numpy(labels.astype(np.long))

# Step 3: Create a DataLoader for training
batch_size = 128
dataset = TensorDataset(inputs_tensor, labels_tensor)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define your FeedForward Neural Network using PyTorch
class FeedForwardNet(nn.Module):
    def __init__(self, input_size=17, hidden_sizes=[32, 32], num_classes=4):
        super(FeedForwardNet, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_sizes[0])
        self.act = nn.Sigmoid()
        self.hidden2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.output = nn.Linear(hidden_sizes[1], num_classes)  # Logits as outputs

    def forward(self, x):
        x = self.act(self.hidden1(x))
        x = self.act(self.hidden2(x))
        x = self.output(x)  # Logits output
        return x

# Step 4: Initialize the model, loss function, and optimizer
model = FeedForwardNet(input_size=24, hidden_sizes=[64, 64], num_classes=len(np.unique(labels)))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Step 5: Training loop
num_epochs = 70

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs_batch, labels_batch in train_loader:
        outputs = model(inputs_batch)
        loss = criterion(outputs, labels_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Step 6: Evaluate the model using a confusion matrix
def plot_confusion_matrix(model, inputs_tensor, labels_tensor):
    model.eval()
    with torch.no_grad():
        outputs = model(inputs_tensor)
        _, predictions = torch.max(outputs, 1)
        cm = confusion_matrix(labels_tensor.numpy(), predictions.numpy())
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.show()

plot_confusion_matrix(model, inputs_tensor, labels_tensor)