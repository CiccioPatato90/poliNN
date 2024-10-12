import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch import tensor, cuda, manual_seed, zeros, nn, optim
from db import Database
from states import Actions
import pandas as pd
from feedforward import *
from evaluate import *
import random

from sklearn.preprocessing import MinMaxScaler

# nndebugger functions
#from bin.nndebugger import constants, loss, dl_debug

random.seed(17)
manual_seed(17)
np.random.seed(17)

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
#print(df.head())

# Run the visualizations
#reduced_df = apply_pca(df, variance_threshold=0.95)
#print(reduced_df.head())

df_labels = df['label']
df_features = df.drop(columns=['label'])

#scaler = StandardScaler()
#scaled_features = scaler.fit_transform(df_features.values)

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df_features.values)
df_features_scaled = pd.DataFrame(scaled_features)


tensor_features = torch.tensor(df_features.values, dtype=torch.float32)
tensor_labels = torch.tensor(df_labels.values, dtype=torch.long)


action = Actions(tensor_features, tensor_labels, num_records, batch_size=128)

df = pd.DataFrame(action.features_tensor)
print(df.head())

device = torch.device(action.get_device())

model = FeedForwardNet(input_dim=action.get_input_size(), output_dim=action.get_num_unique_labels(), capacity=10).to(device=device)
#model = ClusteringModel(input_dim=action.get_input_size(), output_dim=action.get_num_unique_labels(), capacity=20).to(device=device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.025)

Actions.train_model(model, action.train_loader, criterion, optimizer, device=device)

cm = Actions.evaluate_model(model, action.test_loader, class_names=action.get_unique_labels(), device=device)

# Save the model
#torch.save(model.state_dict(), 'feedfwdnet.pth')

# Load the model
#model = SimpleFeedforwardNet(input_size=inputs.shape[1], hidden_sizes=[15, 20], num_classes=num_classes)
#model.load_state_dict(torch.load('model.pth'))
#model.eval())