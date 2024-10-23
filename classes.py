import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import utils as ut
from typing import List, Dict

class ValueMetadata:
    def __init__(self):
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

    def find_outliers_IQR(self, df: pd.DataFrame):
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)

        IQR = q3 - q1

        #return a dataframe holding the outliers based on their position
        outliers = df[((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))]

        return outliers
    

    def run_cleaned_analysis(self, df: pd.DataFrame, file: str, metadata: ValueMetadata,top_n_outliers=10, min_outlier_threshold=5):
        # Perform the analysis again on the cleaned DataFrame (without outliers)
            with open(file, 'w') as f:
                # Gather all the values for the cluster into a DataFrame
                #df = pd.DataFrame(self.values)
                f.write(f"Cluster {ut.to_binary(self.cluster_one_hot)} Outliers Analysis:\n")
                f.write(f"(Cluster size -->  rows: {self.total}, col: 23)\n")
                
                outlier_indices = []
                
                for col in df.columns:
                    # Find outliers for the current column
                    outliers_col = self.find_outliers_IQR(df[col])
                    
                    # Filter to get only non-null values (these are the actual outliers)
                    outliers_col = outliers_col.dropna()
                    
                    if len(outliers_col) > 0:
                        # Write summary for this column's outliers
                        f.write(f"COL {col}\n")
                        f.write("Total number of outliers: " + str(len(outliers_col)) + "\n")

                        # Get the top N outliers by value
                        top_outliers = outliers_col.nlargest(top_n_outliers)
                        f.write(f"Top {top_n_outliers} outlier values (max): {top_outliers.tolist()}\n")
                        f.write(f"Mean value for column : {metadata.mean[col]}\n")
                        f.write("Max outlier value: " + str(outliers_col.max()) + "\n")
                        f.write("Min outlier value: " + str(outliers_col.min()) + "\n")
                        
                        # Add the row indices of the outliers to a tracking list
                        outlier_indices.extend(outliers_col.index.tolist())

                        f.write("\n" + "=" * 80 + "\n\n")
                
                # Count the number of outliers for each row
                outlier_count_per_row = pd.Series(outlier_indices).value_counts()

                # Filter to include only rows with at least 'min_outlier_threshold' outliers
                significant_outliers = outlier_count_per_row[outlier_count_per_row >= min_outlier_threshold]
                
                # Identify and display rows with the most outliers
                f.write(f"Rows with more than {min_outlier_threshold - 1} outliers:\n")
                for index, count in significant_outliers.items():
                    f.write(f"Row {index}: {count} outliers\n")

                f.write("\n" + "=" * 80 + "\n\n")
                
                f.close()


    def analyze(self, metadata: ValueMetadata, file=None, top_n_outliers=10, min_outlier_threshold=5):
        # Focusing on outliers
        if file:
            with open(file, 'w') as f:
                # Gather all the values for the cluster into a DataFrame
                df = pd.DataFrame(self.values)
                f.write(f"Cluster {ut.to_binary(self.cluster_one_hot)} Outliers Analysis:\n")
                f.write(f"(Cluster size -->  rows: {self.total}, col: 23)\n")
                
                outlier_indices = []
                
                for col in df.columns:
                    # Find outliers for the current column
                    outliers_col = self.find_outliers_IQR(df[col])
                    
                    # Filter to get only non-null values (these are the actual outliers)
                    outliers_col = outliers_col.dropna()
                    
                    if len(outliers_col) > 0:
                        # Write summary for this column's outliers
                        f.write(f"COL {col}\n")
                        f.write("Total number of outliers: " + str(len(outliers_col)) + "\n")

                        # Get the top N outliers by value
                        top_outliers = outliers_col.nlargest(top_n_outliers)
                        f.write(f"Top {top_n_outliers} outlier values (max): {top_outliers.tolist()}\n")
                        f.write(f"Mean value for column : {metadata.mean[col]}\n")
                        f.write("Max outlier value: " + str(outliers_col.max()) + "\n")
                        f.write("Min outlier value: " + str(outliers_col.min()) + "\n")
                        
                        # Add the row indices of the outliers to a tracking list
                        outlier_indices.extend(outliers_col.index.tolist())

                        f.write("\n" + "=" * 80 + "\n\n")
                
                # Count the number of outliers for each row
                outlier_count_per_row = pd.Series(outlier_indices).value_counts()

                # Filter to include only rows with at least 'min_outlier_threshold' outliers
                significant_outliers = outlier_count_per_row[outlier_count_per_row >= min_outlier_threshold]
                
                # Identify and display rows with the most outliers
                f.write(f"Rows with more than {min_outlier_threshold - 1} outliers:\n")
                for index, count in significant_outliers.items():
                    f.write(f"Row {index}: {count} outliers\n")

                f.write("\n" + "=" * 80 + "\n\n")
                
                f.close()

                # Now remove the outliers and re-run the analysis
                if outlier_indices:
                    # Convert outlier indices to a set (to avoid duplicates)
                    unique_outliers = list(set(outlier_indices))
                    
                    # Remove rows with outliers from the DataFrame
                    cleaned_df = df.drop(index=unique_outliers)
                    
                    # Re-run analysis on the cleaned DataFrame
                    self.run_cleaned_analysis(cleaned_df, "cleaned_analysis_output.txt", metadata, 10, 5)

                    return cleaned_df
                    

                if False:
                    plt.figure(figsize=(12, 8))
                    sns.boxplot(data=df)
                    plt.title(f'Class Distribution ({ut.to_binary(self.cluster_one_hot)}) in Dataset')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(f"./res/{ut.to_binary(self.cluster_one_hot)}")
                    plt.close()
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
        parameters = ['Mean', 'Median', 'Standard Deviation', 'Variance', 'Min', 'Max', 'Range', 'Skewness', 'Sum of Squared Errors (SSE)']

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
                    f.write(f"Cluster {cluster} Statistics:\n")
                    f.write("-" * 80 + "\n")

                    f.write(pd.DataFrame(stats).to_string())

                    f.write(f"\nMean across cluster columns: {cluster_mean_across_cols} \n")
                    
                    f.write("\n"+"=" * 80 + "\n\n")
        else:
            # Print the output to the console instead
            for cluster in clusters:
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

                # Create the DataFrame for console output
                df = pd.DataFrame(stats, index=parameters)

                print(f"Cluster {cluster} Statistics:")
                print("-" * 40)
                print(df.to_string())  # Display the DataFrame in a table format
                print("=" * 40 + "\n")