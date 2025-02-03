import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import utils as ut
from typing import List, Dict

class ValueMetadata:
    def __init__(self):
        self.size: int = 0
        self.mean: float = 0.0
        self.median: float = 0.0
        self.std: float = 0.0
        self.var: float = 0.0
        self.min: float = 0.0
        self.max: float = 0.0
        self.range: float = 0.0
        self.skewness: float = 0.0
        self.sum_squared_errors: float = 0.0


class ValueInfo:
    def __init__(self):
        self.total: int = 0
        self.values: np.ndarray = np.array([])  # NumPy array for faster processing
        self.cluster_one_hot: np.ndarray = np.array([])
        self.cluster_binary: int = 0

    def find_outliers_IQR(self, df: pd.DataFrame):
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)

        IQR = q3 - q1

        #return a dataframe holding the outliers based on their position
        outliers = df[((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))]

        #print(f"IQR:", outliers)

        return outliers


    def analyze(self, metadata: ValueMetadata, clean: bool,file=None, min_outlier_threshold=3):
        # Focusing on outliers
        if file:
            #with open(file, 'w') as f:
                # Gather all the values for the cluster into a DataFrame
                df = pd.DataFrame(self.values)
                #f.write(f"Cluster {cluster_label} Outliers Analysis:\n")
                #f.write(f"(Cluster size -->  rows: {self.total}, col: 23)\n")
                
                outlier_dict = {}
                #print(df.index)
                # init cluster map all to zero
                for row_idx in df.index:
                    outlier_dict[row_idx] = 0 
                
                for col in df.columns:
                    # Find outliers for the current column
                    outliers = self.find_outliers_IQR(df[col])
                    # update row index for outliers
                    # IMPORTANT: INDICES ARE RELATIVE TO CLUSTER, NOT GLOBAL INDEX
                    for outlier_row_idx in outliers.index:
                        outlier_dict[outlier_row_idx] +=1


                
                # Convert outlier_dict to a DataFrame for easier display
                outlier_df = pd.DataFrame(list(outlier_dict.items()), columns=['Row Index', 'Outlier Count'])
                # Filter out rows where 'Outlier Count' is 0, sort by outlier_count
                outlier_df = outlier_df[outlier_df['Outlier Count'] > min_outlier_threshold]
                outlier_df = outlier_df.sort_values(by='Outlier Count', ascending=False)

                # Display the outlier DataFrame
                #print(outlier_df.head())
                outlier_df.to_csv(f'res/csv/outlier_analysis_cluster_{self.cluster_binary}.csv', index=False)

                if clean:
                    # Get the row indices to remove (from outlier_df)
                    rows_to_remove = outlier_df['Row Index']
                    print(f"Removed {len(rows_to_remove)} from cluster {self.cluster_binary}")
                    # Remove those rows from the original DataFrame (df)
                    df_cleaned = df.drop(rows_to_remove)
                    print(f"df_{self.cluster_binary}", df.shape)
                    print(f"df_cleaned_{self.cluster_binary}", df_cleaned.shape)
                    return df_cleaned
                else:
                    return []
        else:
            print("No file to write analisys!!")
            exit()


class ValuesDict:
    def __init__(self):
        #self.total: int = 0
        self.values_dict: Dict[int, ValueInfo] = {
            0: ValueInfo(),
            1: ValueInfo(),
            2: ValueInfo(),
            3: ValueInfo(),
        }
                                

class MetadataDict:
    def __init__(self):
        #self.total: int = 0
        self.values_dict: Dict[int, ValueMetadata] = {
            0: ValueMetadata(),
            1: ValueMetadata(),
            2: ValueMetadata(),
            3: ValueMetadata(),
        }

    def print(self, clusters, file=None):
       # Define the list of parameters you want to display for each cluster

        float_formatter = "{:.3f}".format
        np.set_printoptions(formatter={'float_kind':float_formatter})

        if file:
            with open(file, 'w') as f:
                for cluster in clusters:
                    # Gather all the values for the cluster into a dictionary
                    stats = {
                        'Mean': self.values_dict[cluster].mean,
                        'Median': self.values_dict[cluster].median,
                        'Standard Deviation': self.values_dict[cluster].std,
                        'Variance': self.values_dict[cluster].var,
                        'Min': self.values_dict[cluster].min,
                        'Max': self.values_dict[cluster].max,
                        'Range': self.values_dict[cluster].range,
                        'Skewness': self.values_dict[cluster].skewness,
                        'Sum of Squared Errors (SSE)': self.values_dict[cluster].sum_squared_errors
                    }

                    mean = np.array(stats['Mean'])
                    cluster_mean_across_cols = np.mean(mean)

                    # Write the cluster header
                    f.write(f"Cluster {cluster} Statistics (for each column):\n")
                    f.write(f"Cluster {cluster} size (num_records): { self.values_dict[cluster].size}\n")
                    f.write("-" * 80 + "\n")

                    f.write(pd.DataFrame(stats).to_string())

                    f.write(f"\nMean across cluster columns: {cluster_mean_across_cols} \n")
                    
                    f.write("\n"+"=" * 80 + "\n\n")



from tensorflow.python.keras.layers import Dense, Layer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import adam_v2

class Model:
    def __init__(self, input_dim, output_dim):
        self.model = Sequential()
        self.model.add(Dense(16, input_dim=input_dim, activation='relu'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(output_dim, activation='softmax'))
        
        optimizer = adam_v2.Adam(learning_rate=0.001)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


