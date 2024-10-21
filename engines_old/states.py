import torch
from torch import nn, Tensor
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, balanced_accuracy_score, classification_report
from typing import Tuple, List, Optional
import torch.nn.functional as F
from torch.utils.data import TensorDataset,random_split

class Actions():
    def __init__(self, features_tensor: Tensor, labels_tensor: Tensor, num_records, batch_size):
        self.dataset = TensorDataset(features_tensor, labels_tensor)
        self.features_tensor = features_tensor
        self.labels_tensor = labels_tensor
        # Split dataset
        gen = torch.Generator().manual_seed(9)
        self.train_size = int(np.floor(0.7 * num_records))
        self.test_size = num_records - self.train_size
        train_dataset, test_dataset = random_split(self.dataset, [self.train_size, self.test_size], gen)
        
        # Create DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size)

    def compute_class_weights(labels):
        # Calculate the frequency of each class
        class_sample_count = torch.bincount(labels)
        # Invert the frequencies: less frequent classes get higher weight
        class_weights = 1.0 / class_sample_count.float()
        return class_weights
    
    def get_input_size(self):
        return self.features_tensor.size(dim=1)

    def get_num_unique_labels(self):
        return len(torch.unique(self.labels_tensor))
    
    def get_unique_labels(self):
        print(torch.unique(self.labels_tensor))
        return torch.unique(self.labels_tensor)
    
    def get_device(self):
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {device} device")
        return device

    def train_model(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device) -> None:
        model.train()
        n_epochs = 30
        for epoch in range(n_epochs):
            running_loss = 0.0
            for inputs_batch_gen, labels_batch_gen in train_loader:
                # Forward pass: compute the model prediction (no need for slicing)
                inputs_batch = inputs_batch_gen.to(device) 
                labels_batch = labels_batch_gen.to(device)
                

                #distances, _ = model(inputs_batch)
                #loss = criterion(distances, labels_batch)
                outputs = model(inputs_batch)
                # Compute the loss
                loss = criterion(outputs, labels_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            average_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {average_loss:.4f}")



    def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device, class_names: Optional[List[str]] = None) -> Tuple[float, np.ndarray]:
        model.eval()  # Disable dropout for evaluation

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs_batch, labels_batch in test_loader:
                inputs_batch, labels_batch = inputs_batch.to(device), labels_batch.to(device)  # Move data to device
                
                outputs = model(inputs_batch)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())  # Move predictions to CPU and convert to numpy
                all_labels.extend(labels_batch.cpu().numpy())  # Move labels to CPU and convert to numpy

            print(all_preds[:5])

    # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        print(cm)

        return cm
  
