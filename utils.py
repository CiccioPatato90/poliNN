import numpy as np
from numba import jit
import time
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

def to_one_hot(binary, len_labels) -> np.ndarray:
    """ Encode the label into one-hot format. """
    res = np.zeros(len_labels, dtype=int)
    res[binary] = 1
    return res