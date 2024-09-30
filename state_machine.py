import torch
from torch import nn
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader,random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import Tuple, List, Optional
from db import Database, EnergyConsumptionDataset
from state import State

class StateMachine():
    def __init__(self):
        self.db = Database('res/records.db')
        self.data = self.db.fetch_all()
        self.db.close_conn()
        self.states = State()
        self.batch_size = 128
        obj = State.process_data(data=self.data)
        self.dataset = EnergyConsumptionDataset(obj, self.batch_size)
        self.get_device()
        
    def get_device(self):
        # Get cpu, gpu or mps device for training.
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {self.device} device")
        
        





