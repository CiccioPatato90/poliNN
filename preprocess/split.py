import numpy as np
from db import Database
import pickle
from constants import *
import random
from typing import List, Dict
import utils as ut

class ValueInfo:
    def __init__(self):
        self.total: int = 0
        self.values: np.ndarray = np.array([])  # NumPy array for faster processing
        self.cluster_one_hot: np.ndarray = np.array([])

class ValuesDict:
    def __init__(self):
        #self.total: int = 0
        self.values_dict: Dict[int, ValueInfo] = {
            0: ValueInfo(),
            1: ValueInfo(),
            2: ValueInfo(),
            3: ValueInfo(),
        }

class Data:
    def __init__(self):
        self.total_count: int = 0
        self.total_values: ValuesDict = ValuesDict()
        self.train_values: ValuesDict = ValuesDict()
        self.test_values: ValuesDict = ValuesDict()

    def init_pickle(self, labels):
        """ 
        Call this function once to load `total_values` into their clusters
        and cache the object for future runs.
        """
        db = Database('res/records.db')
        for label in labels:
            cluster_data = db.fetch_cluster(label)

            # utils function to_one_hot
            encoded_label = ut.to_one_hot(label, len(labels))

            self.total_count += len(cluster_data)
            self.total_values.values_dict[int(label)].total = len(cluster_data)
            self.total_values.values_dict[int(label)].cluster_one_hot = encoded_label
            self.total_values.values_dict[int(label)].values = np.array(cluster_data)

        # Cache the dataset using pickle
        with open(CACHE_DIR, 'wb') as f:
            pickle.dump(self, f)

    def split(self, proportion=0.5, verbose=False):
        """
        Split the dataset into training and test sets efficiently using NumPy arrays.
        """
        # Load the cached data
        with open(CACHE_DIR, 'rb') as f:
            data_instance: Data = pickle.load(f)

        print("Total count:", data_instance.total_count)

        for label, cluster in data_instance.total_values.values_dict.items():
            cluster_size = cluster.total
            train_size = int(cluster_size * proportion)
            test_size = cluster_size - train_size

            # Create a shuffled index for the dataset
            shuffled_idx = np.random.permutation(cluster_size)
            if verbose:
                print("shuffled: ", shuffled_idx, "for cluster: ", cluster.cluster_one_hot)

            # Split the shuffled indices into train and test sets
            train_indices = shuffled_idx[:train_size]
            test_indices = shuffled_idx[train_size:]

            # Assign to train and test sets
            self.train_values.values_dict[label].values = cluster.values[train_indices]
            self.train_values.values_dict[label].cluster_one_hot = cluster.cluster_one_hot
            self.test_values.values_dict[label].values = cluster.values[test_indices]
            self.test_values.values_dict[label].cluster_one_hot = cluster.cluster_one_hot
            
            if verbose:
                print(f"Cluster {label} - Train size: {len(train_indices)}, Test size: {len(test_indices)}")

    def extract_train_test(self):
        """
        Extract the train and test data as pairs of (ValueInfo.values, ValueInfo.cluster_one_hot).
        Each element in the resulting lists will be a tuple (features, label).
        """

        # Lists to store the train and test data
        train_data = []
        test_data = []

        # Extract the training data from train_values
        for label, value_info in self.train_values.values_dict.items():
            for value in value_info.values:
                train_data.append((value, value_info.cluster_one_hot))  # Append (features, label) pair

        # Extract the testing data from test_values
        for label, value_info in self.test_values.values_dict.items():
            for value in value_info.values:
                test_data.append((value, value_info.cluster_one_hot))  # Append (features, label) pair

        return train_data, test_data


    def print_results(self):
        """
        Print the current distribution of training and testing sets.
        """
        n_train = 0
        n_test = 0
        print("\n--- Training set ---")
        
        for label, cluster in self.train_values.values_dict.items():
            n_train = n_train + len(cluster.values)
            print(f"Cluster {label} - Samples: {cluster.values.shape[0]}")
            print(f"    Ex: {cluster.values[60]}, One-Hot: {cluster.cluster_one_hot}")
        print("N_train: ", n_train)

        print("\n--- Testing set ---")
        for label, cluster in self.test_values.values_dict.items():
            n_test = n_test + len(cluster.values)
            print(f"Cluster {label} - Samples: {cluster.values.shape[0]}")
            print(f"    Ex: {cluster.values[60]}, One-Hot: {cluster.cluster_one_hot}")
        print("N_test: ", n_test)
    
    def __to_string__ (self):
        with open(CACHE_DIR, 'rb') as f:
            data_instance: Data = pickle.load(f)
        for label, list_cluster in data_instance.total_values.values_dict.items():
            print("cluster: ", label, "\nfeature: ", list_cluster.values[0][60], "\none_hot:", list_cluster.cluster_one_hot, "\nsize:", list_cluster.total,"\n\n")
            print("Total count:", data_instance.total_count)





