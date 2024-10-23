import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from db import Database
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import evaluate as Evaluate
import random
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE