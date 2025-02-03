import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Dict
import random

from db import Database
from constants import *
import utils as ut
from classes import ValuesDict, MetadataDict
import evaluate



class Data:
    def __init__(self):
        self.total_count: int = 0
        self.cluster_labels: np.ndarray = np.array([0,1,2,3])
        self.total_values: ValuesDict = ValuesDict()
        self.train_values: ValuesDict = ValuesDict()
        self.test_values: ValuesDict = ValuesDict()
        self.metadata: MetadataDict = MetadataDict()

    def pickle_dump(self, dir=CACHE_DIR):
        # Cache the dataset using pickle
        with open(dir, 'wb') as f:
            pickle.dump(self, f)

    def update_metadata(self, pds: pd.DataFrame, binary_label):
        self.metadata.values_dict[int(binary_label)].size = pds.shape[0]
        self.metadata.values_dict[int(binary_label)].mean = pds.mean()
        self.metadata.values_dict[int(binary_label)].median = pds.median()
        self.metadata.values_dict[int(binary_label)].std = pds.std()
        self.metadata.values_dict[int(binary_label)].var = pds.var()
        self.metadata.values_dict[int(binary_label)].min = pds.min()
        self.metadata.values_dict[int(binary_label)].max = pds.max()
        self.metadata.values_dict[int(binary_label)].range = (pds.max() - pds.min())
        self.metadata.values_dict[int(binary_label)].skewness = pds.skew()
        self.metadata.values_dict[int(binary_label)].sum_squared_errors = np.sum((pds - pds.mean())**2)

    def init_pickle(self, labels, db: Database):
        """ 
        Call this function once to load `total_values` into their clusters
        and cache the object for future runs.
        """
        
        #for each cluster [0,1,2,3]
        for label in labels:
            #retrieve data from db
            cluster_data = db.fetch_cluster(label)

            # utils function to_one_hot
            encoded_label = ut.to_one_hot(label, len(labels))

            #update total len
            self.total_count += len(cluster_data)
            # assign cluster len
            self.total_values.values_dict[int(label)].total = len(cluster_data)
            # assign one-hot encoding
            self.total_values.values_dict[int(label)].cluster_one_hot = encoded_label
            # assign binary class encoding
            self.total_values.values_dict[int(label)].cluster_binary = label
            # copy cluster records to cache
            self.total_values.values_dict[int(label)].values = np.array(cluster_data)

            # convert to dataframe for metadata calculations
            pds = pd.DataFrame(cluster_data)
            self.update_metadata(pds, label)
        # dump in cache
        self.pickle_dump()

    def info(self, clusters, filename, cache_dir = CACHE_DIR):
        with open(cache_dir, 'rb') as f:
            data_instance: Data = pickle.load(f)
            data_instance.metadata.print(clusters=clusters, file=filename)
            
    def analyze(self, clusters, filename, cache_dir = CACHE_DIR,clean=False, outliers_threshold = 5):
        with open(cache_dir, 'rb') as f:
            data_instance: Data = pickle.load(f)
            if clean:
                new_data_instance = Data()
                new_total_count = 0

            for cluster in clusters:
                new_cluster = data_instance.total_values.values_dict[int(cluster)].analyze(file=filename, metadata=data_instance.metadata.values_dict[int(cluster)], clean=clean, min_outlier_threshold=outliers_threshold)
                
                if clean:
                    #ricevo un nuovo cluster ogni volta, voglio aggiornare il modello dei dati
                    #copia i nuovi valori
                    new_data_instance.total_values.values_dict[int(cluster)].values = np.array(new_cluster)
                    #aggiorna la nuova len
                    new_data_instance.total_values.values_dict[int(cluster)].total = len(np.array(new_cluster))
                    new_total_count += len(np.array(new_cluster))
                    encoded_label = ut.to_one_hot(cluster, len(self.cluster_labels))
                    new_data_instance.total_values.values_dict[int(cluster)].cluster_one_hot = encoded_label
                    new_data_instance.update_metadata(pd.DataFrame(new_cluster), cluster)
                    new_data_instance.total_count+=len(np.array(new_cluster))
            if clean:
                new_data_instance.total_count = new_total_count
                new_data_instance.pickle_dump(NO_OUTLIERS_CACHE_DIR)



    def reconstruct_and_save_records(self, X_data, y_data, cache_dir, db: Database, data_instance,dataset_type="train"):
        # Concatenate records with labels
        records = np.empty((len(X_data), X_data.shape[1] + 1))
        for idx, (record, label_binary) in enumerate(zip(X_data, y_data)):
            records[idx] = np.concatenate((record, [int(label_binary)]))
        
        # Debug print the reconstructed data
        ut.print_debug(records, info=f"reconstructed_{dataset_type}", num_elements=3)
        
        # Save records to the appropriate database table
        if dataset_type == "train":
            db.save_train_records(records)
        elif dataset_type == "test":
            db.save_test_records(records)
        
        # Load the data instance and process clusters
        
            
        # Fetch clusters and update `data_instance` for each label
        for label in self.cluster_labels:
            if dataset_type == "train":
                cluster_data = db.fetch_train_cluster(label)
            elif dataset_type == "test":
                cluster_data = db.fetch_test_cluster(label)
            
            data_instance_values = data_instance.train_values if dataset_type == "train" else data_instance.test_values
            data_instance_values.values_dict[int(label)].cluster_one_hot = ut.to_one_hot(label, len(self.cluster_labels))
            data_instance_values.values_dict[int(label)].cluster_binary = label
            data_instance_values.values_dict[int(label)].total = len(cluster_data)
            data_instance_values.values_dict[int(label)].values = np.array(cluster_data)
            
            # Pickle dump the updated data instance
            
    
    def sklearn_split(self, db : Database, test_size=0.8, random=False, debug=True, cache_dir=NO_OUTLIERS_CACHE_DIR, is_less = False):
        if is_less:
            X, y = db.fetch_some_divide()
        else:
            X, y = db.fetch_all_divide()

        ut.print_debug(X, info="X", num_elements=3)
        ut.print_debug(y, info="y", num_elements=3)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y,random_state=42)    

        if False:
            num_elements=3
            #debug statement to check that the train_split_test worked correctly
            ut.print_debug(X_train, info="X_train", num_elements=num_elements)
            ut.print_debug(X_test, info="X_test", num_elements=num_elements)
            ut.print_debug(y_train, info="y_train", num_elements=num_elements)
            ut.print_debug(y_test, info="y_test", num_elements=num_elements)

            evaluate.bar_chart_binary(y, 'pre')
            evaluate.bar_chart_binary(y_train, 'train')
            evaluate.bar_chart_binary(y_test, 'test')

        if True:
            with open(cache_dir, 'rb') as f:
                data_instance: Data = pickle.load(f)
                db.clean_test()
                db.clean_train()
                self.reconstruct_and_save_records(X_train, y_train, cache_dir, db, dataset_type="train", data_instance=data_instance)
                self.reconstruct_and_save_records(X_test, y_test, cache_dir, db, dataset_type="test", data_instance=data_instance)
                #exit()
                data_instance.pickle_dump(cache_dir)
        

    def split(self, proportion=0.5,cache_dir = CACHE_DIR,verbose=False):
        """
        Split the dataset into training and test sets efficiently using NumPy arrays.
        """
        # Load the cached data
        with open(cache_dir, 'rb') as f:
            data_instance: Data = pickle.load(f)

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

    def extract_train_test(self, cache_dir):
        """
        Extract the train and test data as pairs of (ValueInfo.values, ValueInfo.cluster_one_hot).
        Each element in the resulting lists will be a tuple (features, label).
        """

        # Lists to store the train and test data
        train_data = []
        test_data = []
        with open(cache_dir, 'rb') as f:
            data_instance: Data = pickle.load(f)

            print(data_instance.test_values.values_dict[1].values)

            print(data_instance.train_values.values_dict[1].values)

            exit()

            # Extract the training data from train_values
            for label, value_info in data_instance.train_values.values_dict.items():
                for value in value_info.values:
                    train_data.append((value, value_info.cluster_one_hot))  # Append (features, label) pair

            # Extract the testing data from test_values
            for label, value_info in data_instance.test_values.values_dict.items():
                for value in value_info.values:
                    test_data.append((value, value_info.cluster_one_hot))  # Append (features, label) pair

            return train_data, test_data

    def extract_train_test_db(self, db: Database):
        X_test, y_test = db.fetch_all_test_divide()
        X_train, y_train = db.fetch_all_train_divide()
        print("Extraction completed.")
        return X_train, y_train, X_test, y_test

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
    
    def __to_string__ (self, cache_dir):
        with open(cache_dir, 'rb') as f:
            data_instance: Data = pickle.load(f)
        for label, list_cluster in data_instance.total_values.values_dict.items():
            print("cluster: ", label, "\nfeature: ", list_cluster.values[0][60], "\none_hot:", list_cluster.cluster_one_hot, "\nsize:", list_cluster.total,"\n\n")
            print("Total count:", data_instance.total_count)
        for label, list_cluster in data_instance.train_values.values_dict.items():
            print("cluster: ", label, "\nfeature: ", list_cluster.values[0][60], "\none_hot:", list_cluster.cluster_one_hot, "\nsize:", list_cluster.total,"\n\n")
            print("Total count:", data_instance.total_count)
        for label, list_cluster in data_instance.test_values.values_dict.items():
            print("cluster: ", label, "\nfeature: ", list_cluster.values[0][60], "\none_hot:", list_cluster.cluster_one_hot, "\nsize:", list_cluster.total,"\n\n")
            print("Total count:", data_instance.total_count)





