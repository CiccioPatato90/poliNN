import numpy as np
from numba import jit
import time
import json
from functools import wraps

def print_debug(data_list,info="[empty]", num_elements=1):
    if num_elements > len(data_list):
        print(f"Requested {num_elements} elements, but the list only has {len(data_list)}.")
        num_elements = len(data_list)  # Limit to the size of the list
    
    # Randomly select indices using numpy
    random_indices = np.random.choice(len(data_list), size=num_elements, replace=False)
    print(f"{info}:")
    for index in random_indices:
        print(f"idx {index}: {data_list[index]}")
    print("\n")

def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time
        print(f"Elapsed time for {func.__name__}: {elapsed_time:.4f} seconds")
        return result
    return wrapper

#@jit(nopython=True) in this case numbra made the fun slower
def to_binary(one_hot):    
    index = np.where(one_hot == 1)[0][0]
    return index

def to_one_hot(binary: int, len_labels) -> np.ndarray:
    """ Encode the label into one-hot format. """
    res = np.zeros(len_labels, dtype=int)
    res[binary] = 1
    return res

def divide_binary(data):
    dfs = np.split(data, [24], axis=1)
    # FORMAT: [0.06  0.097 0.064 0.059 0.059 0.097 0.061 0.057 0.057 0.095 0.058 0.059 0.07 0.087 0.054 0.055 0.089 0.067 0.068 0.066 0.09  0.058 0.056 0.087]
    X = dfs[0]
    # FORMAT: [1.] [1.] [1.] [2.] [3.]
    y = dfs[1]
    return X, y


def divide_one_hot(data):

    # Split data into features (X) and labels (y)
    X, y = np.split(data, [24], axis=1)
    
    X = X.astype(float)
    # Convert each label from JSON-like string to an integer array
    y = np.array([json.loads(label[0]) for label in y], dtype=int)
    
    return X, y