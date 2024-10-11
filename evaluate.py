import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'df' is your cleaned DataFrame with features and labels
# Replace this with the actual DataFrame you created

# 1. Histogram of Feature Distributions
def plot_feature_distributions(df):
    feature_columns = df.columns[:-2]  # Assuming the last two columns are 'label' and 'used_training'
    df[feature_columns].hist(figsize=(16, 12), bins=20, edgecolor='black')
    plt.suptitle('Feature Distributions', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# 2. Correlation Heatmap
def plot_correlation_heatmap(df: pd.DataFrame):
    feature_columns = df.columns[:-2]  # Exclude 'label' and 'used_training'
    correlation_matrix = df[feature_columns].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap of Features', fontsize=16)
    plt.savefig('eval/correlation_heatmap.png')

# 3. Class Distribution
def plot_class_distribution(df):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='label', data=df, palette='viridis')
    plt.title('Class Distribution', fontsize=16)
    plt.xlabel('Class Labels')
    plt.ylabel('Frequency')
    plt.show()

# 4. Boxplots for Outlier Detection
def plot_boxplots(df):
    feature_columns = df.columns[:-2]  # Exclude 'label' and 'used_training'
    plt.figure(figsize=(20, 12))
    sns.boxplot(data=df[feature_columns], orient='h', palette='Set2')
    plt.title('Boxplots of Features for Outlier Detection', fontsize=16)
    plt.xlabel('Feature Value')
    plt.ylabel('Features')
    plt.show()
    
def plot_clustermap(df):
    plt.figure(figsize=(12, 10))
    sns.clustermap(df.corr(), cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5)
    plt.title('Clustermap of Correlation Matrix')
    plt.savefig('eval/clustermap.png')

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def apply_pca(df: pd.DataFrame, variance_threshold: float = 0.95) -> pd.DataFrame:
    """
    Perform PCA on the provided DataFrame and reduce its dimensionality 
    while maintaining the specified explained variance threshold.

    Parameters:
    - df: pd.DataFrame - The DataFrame containing feature columns and labels.
    - variance_threshold: float - The cumulative explained variance to retain (default is 0.95).

    Returns:
    - reduced_df: pd.DataFrame - DataFrame with principal components and the original labels.
    """
    
    # Step 1: Standardize the data (if not already standardized)
    feature_columns = df.columns[:-1]  # Exclude 'label' and 'used_training' columns
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_columns])

    # Step 2: Apply PCA
    pca = PCA()
    pca.fit(scaled_features)

    # Step 3: Plot the explained variance ratio to determine the optimal number of components
    explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance by Number of Principal Components')
    plt.grid(True)
    plt.show()

    # Step 4: Determine the number of components based on the variance threshold
    n_components = np.argmax(explained_variance_ratio >= variance_threshold) + 1
    print(f'Optimal number of components to retain {variance_threshold * 100}% variance: {n_components}')

    # Step 5: Transform the data using the optimal number of components
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(scaled_features)

    # Optional: Create a new DataFrame with the reduced components and the original labels
    reduced_df = pd.DataFrame(reduced_features, columns=[f'PC{i+1}' for i in range(n_components)])
    reduced_df['label'] = df['label'].values

    return reduced_df
