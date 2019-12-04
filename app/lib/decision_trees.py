import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tree_node import Node



class DecisionTree():
    
    def __init__(self, targets, data, class_names = None, metric = "Gini", max_level = 10, min_split = 2, min_members = 1):
        self.targets = targets
        self.data = data
        self.class_names = class_names
        
        if metric is "Gini":
            self.metric = metric
        elif metric is "Entropy":
            self.metric = metric
        else:
            self.metric = "Gini" # def metric just in case
        self.max_level = max_level
        self.min_split = min_split
        self.min_members = min_members

        self.root = self.spawn_node(self.targets, self.data, self.metric, level = 0)
        pass



    def fit_id3(self):
        pass


    def fit_cart(self):
        pass


    def predict(self):
        pass

    def test_accuracy(self, targets, data):
        pass
    
     
    def spawn_node(self, targets, data, metric, class_names = None, level = 0,):
        return Node(targets,data,level, metric, class_names)


    def check_final(self, node):
        pass
    

# class DecisionTree():
#     def fit(self):
#         #TODO: placeholder wrapper for tree creation based on separated part of dataset
#         pass

#     def test(self):
#         #TODO: placeholder wrapper for tree testing based on separated part of dataset
#         pass

#     def split_data(self):
#         #TODO: wrapper for sklearn functions
#         data = self.dataset.data
#         target = self.dataset.target
#         self.train_data, self.test_data, self.train_target, self.test_target = train_test_split(data, target, test_size = 0.20)
#         pass

#     def create_tree(self, dataset=None, test=True):
#         #TODO: If no dataset provided, create a tree based on Iris dataset. If provided, split data or do not split, let user decide the percentage
        
#         if dataset is None:
#             dataset = load_iris()
#         else:
#             self.dataset = dataset

#         if test:
#             self.split_data()
#             self.fit()
#             self.test()
#         else:
#             self.train_data = dataset.data
#             self.train_target = dataset.target
#             self.fit()



if __name__ == "__main__":
    pass

