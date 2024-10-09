import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from db import Database
from states import Actions
import pandas as pd
from feedforward import *
from sklearn.preprocessing import StandardScaler
from evaluate import *

# Step 1: Fetch data from SQLite and convert to NumPy arrays
db = Database('res/records.db')
data = db.fetch_all()
num_records = len(data)
db.close_conn()

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(data, columns=['a01', 'a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'a08', 'a09',
                                 'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19', 
                                 'a20', 'a21', 'a22', 'a23', 'a24', 'label'])

# Convert all columns to float
df = df.apply(pd.to_numeric, errors='coerce')
# Display the first few rows of the DataFrame to confirm the conversion
#print(df.head())

# Run the visualizations
#reduced_df = apply_pca(df, variance_threshold=0.95)
#print(reduced_df.head())

#plot_feature_distributions(df)
#plot_correlation_heatmap(df)
#plot_correlation_heatmap(reduced_df)
#plot_class_distribution(df)
#plot_boxplots(df)

df_labels = df['label']
df_features = df.drop(columns=['label'])

# Step 2: Convert data to NumPy array
#data_pr = np.array(data)

#print(data_pr)

# Extract inputs (first 24 columns) and labels (last column)
#inputs_db = data_pr[:,1:24]  # First 24 columns as features
#labels = data_pr[:, -1]    # Last column as labels

#scaler = StandardScaler()
#inputs = scaler.fit_transform(inputs_db)

#print(inputs_db[1])
#print(labels[1])

# Step 4: Convert to PyTorch tensors
#inputs = torch.from_numpy(inputs_db.astype(np.float32))
#labels = torch.from_numpy(labels.astype(np.long))
tensor_features = torch.tensor(df_features.values, dtype=torch.float32)
tensor_labels = torch.tensor(df_labels.values, dtype=torch.long)

print(type(tensor_features))
print(type(tensor_labels))


action = Actions(tensor_features, tensor_labels, num_records, batch_size=128)

# Get cpu, gpu or mps device for training.


# Assuming 'labels' is your labels tensor

# Create an instance of the model
model = FeedForwardNet(input_size=action.get_input_size(), num_classes=action.get_num_unique_labels())

#class_weights = Actions.compute_class_weights(labels=labels)

criterion = nn.CrossEntropyLoss()  # binary cross entropy
#criterion = FocalLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Define the evaluation function

# Train the model
Actions.train_model(model, action.train_loader, criterion, optimizer)

# Evaluate the model
# Get class names from the label 
accuracy, cm = Actions.evaluate_model(model, action.test_loader, class_names=action.get_unique_labels)

# Save the model
#torch.save(model.state_dict(), 'feedfwdnet.pth')

# Load the model
#model = SimpleFeedforwardNet(input_size=inputs.shape[1], hidden_sizes=[15, 20], num_classes=num_classes)
#model.load_state_dict(torch.load('model.pth'))
#model.eval())