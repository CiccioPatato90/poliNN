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

# Step 5: Create custom dataset
class EnergyConsumptionDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

class Actions():
    def __init__(self, features_tensor: Tensor, labels_tensor: Tensor, num_records, batch_size):
        self.dataset = TensorDataset(features_tensor, labels_tensor)
        self.features_tensor = features_tensor
        self.labels_tensor = labels_tensor
        # Split dataset
        gen = torch.Generator().manual_seed(9)
        train_size = int(0.7 * num_records)
        test_size = num_records - train_size
        train_dataset, test_dataset = random_split(self.dataset, [train_size, test_size], gen)
        
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

    def train_model(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer) -> None:
        model.train()
        n_epochs = 50
        for epoch in range(n_epochs):
            for inputs_batch, labels_batch in train_loader:
                # Forward pass: compute the model prediction (no need for slicing)
                optimizer.zero_grad()
                y_pred = model(inputs_batch)
                # Compute the loss
                loss = criterion(y_pred, labels_batch)
                loss.backward()
                optimizer.step()
            print(f'Finished epoch {epoch}, latest loss {loss}')



    def evaluate_model(model: nn.Module, test_loader: DataLoader, class_names: Optional[List[str]] = None) -> Tuple[float, np.ndarray]:
        model.eval()  # Disable dropout for evaluation

        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs_batch, labels_batch in test_loader:
                outputs = model(inputs_batch)
                
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())

            print(all_preds[:5])

            # Check the distribution of true labels and predicted labels
        unique_labels, counts_labels = np.unique(all_labels, return_counts=True)
        print(f"True labels distribution: {dict(zip(unique_labels, counts_labels))}")

        unique_preds, counts_preds = np.unique(all_preds, return_counts=True)
        print(f"Predicted labels distribution: {dict(zip(unique_preds, counts_preds))}")

        # Compute accuracy
        correct = np.sum(np.array(all_preds) == np.array(all_labels))
        total = len(all_labels)
        accuracy: float = 100 * correct / total
        print(f'Accuracy on test set: {accuracy:.2f}%')

        # Check the distribution of predictions
        unique_preds, counts_preds = np.unique(all_preds, return_counts=True)
        print("Predictions distribution:", dict(zip(unique_preds, counts_preds)))

        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        print('Confusion Matrix:')
        print(cm)

        # Plot confusion matrix
        if class_names is None:
            class_names = [str(i) for i in range(len(cm))]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.show()

        # Compute additional metrics: precision, recall, F1-score
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)

        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F1-Score: {f1:.2f}')
        print(f'Balanced Accuracy: {balanced_acc:.2f}')

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))

        return accuracy, cm    
