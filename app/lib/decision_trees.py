import numpy as np
import pandas as pd
from tree_node import Node
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits
from sklearn.datasets import load_wine




class DecisionTree():
    
    def __init__(self, targets, data, metric = "Gini", max_depth = 10, min_split = 2, min_members = 1):
        
        self.targets = targets
        self.data = data
        
        ## I changed 
        # if metric is "Gini":
        #     self.metric = metric
        # elif metric is "Entropy":
        #     self.metric = metric
        # else:
        #     self.metric = "Gini" # def metric just in case

        self.max_level = max_level
        self.min_split = min_split
        self.min_members = min_members

    
    def fit_id3(self):
        pass


    def fit_cart(self):
        pass


    def predict(self):
        pass

    def test_accuracy(self, targets, data):
        pass


    def spawn_node(self, targets, data, metric, class_names = None, level = 0):
        return Node(targets,data,level, metric, class_names)


    def check_final(self, node):
        if node.count == 1 or node.score == 0 or node.level == self.max_level:
            node.set_final()
        pass
    





if __name__ == "__main__":
    pass

