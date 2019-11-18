import numpy as np
from sklearn.datasets import load_iris


def array_mergesort(target, data):
    merged_array = np.c_[target, data]
    merged_array = merged_array[merged_array[:,1].argsort()]
    return merged_array

def gini_impurity(target, data):
    pass





# iris = load_iris()
# data = iris.data
# target = iris.target
# data = data[:,0]
# print(len(array_mergesort(target,data)))

