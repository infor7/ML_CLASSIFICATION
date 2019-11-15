
# Main decision tree class module
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class DecisionTree():
    def fit(self):
        #TODO: placeholder wrapper for tree creation based on separated part of dataset
        pass

    def test(self):
        #TODO: placeholder wrapper for tree testing based on separated part of dataset
        pass

    def split_data(self):
        #TODO: wrapper for sklearn functions
        data = self.dataset.data
        target = self.dataset.target
        self.train_data, self.test_data, self.train_target, self.test_target = train_test_split(data, target, test_size = 0.20)
        pass

    def create_tree(self, dataset=None, test=True):
        #TODO: If no dataset provided, create a tree based on Iris dataset. If provided, split data or do not split, let user decide the percentage
        
        if dataset is None:
            dataset = load_iris()
        else:
            self.dataset = dataset

        if test:
            self.split_data()
            self.fit()
            self.test()
        else:
            self.train_data = dataset.data
            self.train_target = dataset.target
            self.fit()

        
class Node():
    def funcname1(self, parameter_list):
        pass
    
    def funcname2(self, parameter_list):
        pass

    def funcname3(self, parameter_list):
        pass


if __name__ == "__main__":
    pass

