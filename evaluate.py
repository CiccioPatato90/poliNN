from imports import *
from constants import *
from collections import Counter
import pickle
from matplotlib import pyplot
import utils as ut

def cross_correlation(self, cluster_pairs):
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        with open(CACHE_DIR, 'rb') as f:
            data_instance = pickle.load(f)

        for cluster_a, cluster_b in cluster_pairs:
            # Retrieve the two clusters
            print("a", cluster_a)
            print("b", cluster_b)
            data_a = data_instance.total_values.values_dict[cluster_a].values
            data_b = data_instance.total_values.values_dict[cluster_b].values

            ut.print_debug(data_a, num_elements=5)
            ut.print_debug(data_b, num_elements=5)
            
            # Check if the clusters have different sizes and handle accordingly
            min_size = min(len(data_a), len(data_b))
            data_a = data_a[:min_size]
            data_b = data_b[:min_size]
            
            # Combine the two clusters into a DataFrame for correlation
            if isinstance(data_a, np.ndarray):
                data_a = pd.DataFrame(data_a)
            if isinstance(data_b, np.ndarray):
                data_b = pd.DataFrame(data_b)
            
            combined_data = pd.concat([data_a, data_b], axis=1, ignore_index=True)
            
            # Calculate the correlation matrix
            corr_matrix = combined_data.corr()
            
            # Plot the correlation matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title(f'Correlation between Cluster {cluster_a} and Cluster {cluster_b}')
            plt.show()

def bar_chart_binary(x: np.array, info: str):    
    map_count = {0: 0, 1: 0,2: 0 ,3: 0}
    # Count occurrences of each class
    i=0
    #count_class_occ
    for label in np.nditer(x):
        map_count[int(label)] += 1
    print(f"Class counts ({info}): {map_count}")
    # Plot the class distribution
    plt.bar(map_count.keys(), map_count.values())
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Dataset')
    plt.savefig(f"./res/{info}")
    plt.close()

@ut.time_it
def bar_chart_one_hot(x: np.array, info: str):    
    map_count = {0: 0, 1: 0,2: 0 ,3: 0}
    """
    for one_hot in np.nditer(x, flags=['external_loop'], order='F'):
        label = ut.to_binary(one_hot)
    """
    for one_hot in x:
        label = ut.to_binary(one_hot)
        map_count[int(label)] += 1
    print(f"Class counts ({info}): {map_count}")
    # Plot the class distribution
    plt.bar(map_count.keys(), map_count.values())
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title(f'Class Distribution ({info}) in Dataset')
    plt.savefig(f"./res/{info}")
    plt.close()