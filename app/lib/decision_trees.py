# Work in progress
# Main decision tree class module
import numpy as np
from sklearn.datasets import load_iris
from sklearn.

class DecisionTree():
    def train(self):
        #TODO: placeholder wrapper for tree creation based on separated part of dataset
        pass

    def test(self):
        #TODO: placeholder wrapper for tree testing based on separated part of dataset
        pass

    def split_data(self):
        #TODO: wrapper for sklearn functions
        pass

    def create_tree(self, dataset=None, split=True, percentage=None):
        #TODO: If no dataset provided, create a tree based on Iris dataset. If provided, split data or do not split, let user decide the percentage
        if dataset is None:
            dataset = load_iris()
        
        if split:
            #Use 
            pass
        else:
            #No default
        pass

if __name__ == "__main__":
    pass

