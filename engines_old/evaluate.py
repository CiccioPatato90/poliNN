from imports import *
from collections import Counter
from matplotlib import pyplot
import utils as ut

def histogram_binary(x: np.array, info: str):    
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
    plt.savefig(f"./{info}")
    plt.close()

@ut.time_it
def histogram_one_hot(x: np.array, info: str):    
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
    plt.savefig(f"./{info}")
    plt.close()