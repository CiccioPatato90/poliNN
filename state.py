import torch
from torch import nn
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import Tuple, List, Optional

class State():
    def __init__(self, inputs=0, labels=0):
        pass

    def train_model(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, num_epochs: int) -> None:
        for epoch in range(num_epochs):
            model.train()  # Set the model to training mode
            running_loss = 0.0
            for inputs_batch, labels_batch in train_loader:
                
                # Debug statements
                if inputs_batch.device.type != 'cpu' or labels_batch.device.type != 'cpu':
                    print(f"Warning: Inputs or labels are not on CPU. Inputs device: {inputs_batch.device}, Labels device: {labels_batch.device}")

                optimizer.zero_grad()
                outputs = model(inputs_batch)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs_batch.size(0)
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')


    def evaluate_model(model: nn.Module, test_loader: DataLoader, class_names: Optional[List[str]] = None) -> Tuple[float, np.ndarray]:
        model.eval()  # Set the model to evaluation mode
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs_batch, labels_batch in test_loader:
                outputs = model(inputs_batch)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())

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

        return accuracy, cm
    
    def process_data(data):
        """ takes data from db, returns pytorch tensors divided in data and labels"""
        data = np.array(data)

        # Extract inputs and labels
        inputs = data[:, 1:-1]
        labels = data[:, -1]

        # Step 4: Convert to PyTorch tensors
        inputs = torch.from_numpy(inputs.astype(np.float32))
        labels = torch.from_numpy(labels.astype(np.long))
        return [inputs, labels]