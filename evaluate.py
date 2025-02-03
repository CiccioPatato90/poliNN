from constants import *
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import utils as ut
import numpy as np
from db import Database

def calcu_mutual_information(db: Database):
    from sklearn.feature_selection import mutual_info_classif
    X,y = db.fetch_all_divide()
    # Calculate mutual information scores
    X_df = pd.DataFrame(X)
    mi_scores = mutual_info_classif(X, y, random_state=0)
    mi_scores_series = pd.Series(mi_scores, index=X_df.columns).sort_values(ascending=False)

    # Display the scores
    print(mi_scores_series)
    exit()


def cross_correlation(cluster_pairs, cache_dir):
        with open(cache_dir, 'rb') as f:
            data_instance = pickle.load(f)

            for cluster_a, cluster_b in cluster_pairs:
                # Retrieve the two clusters
                data_a = data_instance.total_values.values_dict[cluster_a].values
                data_b = data_instance.total_values.values_dict[cluster_b].values

                # Compute correlation matrices
                df1 = pd.DataFrame(data_a)
                df2 = pd.DataFrame(data_b)
                corr_series = df1.corrwith(df2)
    
                # Plot the correlation values
                plt.figure(figsize=(10, 6))
                corr_series.plot(kind='bar', color='skyblue')
                plt.title(f"Column-wise Correlation between Cluster {cluster_a} and Cluster {cluster_b}")
                plt.xlabel("Columns")
                plt.ylabel("Correlation")
                plt.xticks(rotation=45)
                plt.savefig(f"./res/img/linear_correlation_{cluster_a}_{cluster_b}_(no_outliers)")
            exit()

# Function to plot scatter plots for each column in df1 vs corresponding column in df2
def plot_scatter(cluster_pairs):
    with open(CACHE_DIR, 'rb') as f:
        data_instance = pickle.load(f)

        for cluster_a, cluster_b in cluster_pairs:
            # Retrieve the two clusters
            data_a = data_instance.total_values.values_dict[cluster_a].values
            data_b = data_instance.total_values.values_dict[cluster_b].values

            # Compute correlation matrices
            df1 = pd.DataFrame(data_a)
            df2 = pd.DataFrame(data_b)
            num_columns = df1.shape[1]
            fig, axes = plt.subplots(4, 6, figsize=(18, 12))
            axes = axes.ravel()
            
            for i in range(num_columns):
                axes[i].scatter(df1.iloc[:, i], df2.iloc[:, i], alpha=0.5)
                axes[i].set_title(f"Column {i}")
                axes[i].set_xlabel("df1")
                axes[i].set_ylabel("df2")
            
            plt.tight_layout()
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
    plt.title(f'Class Distribution ({info}) in Dataset')
    plt.savefig(f"./res/img/{info}")
    plt.close()

@ut.time_it
def bar_chart_one_hot(x: np.array, info: str):    
    label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    map_count = {label: 0 for label in label_map.values()}

    for one_hot in x:
        label = ut.to_binary(one_hot)
        map_count[label_map[int(label)]] += 1

    print(f"Class counts ({info}): {map_count}")
    
    # Set font to Times New Roman, size 11
    csfont = {'fontname':'Times New Roman'}

    # Plot the class distribution
    plt.figure(figsize=(6, 4))
    plt.bar(map_count.keys(), map_count.values(), color='skyblue', edgecolor='black')

    # Labels and title
    plt.xlabel('Class', **csfont, fontsize=11)
    plt.ylabel('Number of Samples', **csfont, fontsize=11,)
    plt.title(f'Class Distribution ({info}) in Dataset', **csfont, fontsize=11)

    # Grid in the background
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the figure
    plt.savefig(f"./res/img/{info}", dpi=300, bbox_inches='tight')
    plt.close()