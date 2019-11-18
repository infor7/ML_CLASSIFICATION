import numpy as np


class Node:

    def __init__(self, target, data, level = 0, metric = "Gini", class_names = None):
        self.target = target
        self.data = data
        self.final = False
        
        if class_names is not None:
            self.class_names = class_names
        
        self.assign_class()
        
        if metric is "Gini":
            self.gini_score()
            self.score = self.gini
        elif metric is "Entropy":
            self.entropy_score()
            self.score = self.entropy
        else:
            self.score = None

            
    def assign_splitter(self, column, value):
        self.splitter_column = column
        self.splitter_value = value
        
    def gini_score(self):
        self.gini = 1 - np.sum(np.square(self.classes_count / self.count))
    
    def entropy_score(self):
        pi = self.classes_count / self.count
        self.entropy = np.sum(- pi * np.log2(pi))
    
    def set_final(self):
        self.final = True
    
    def assign_class(self):
        self.classes, self.classes_count = np.unique(self.target, return_counts = True)
        self.count = self.classes_count.sum()
        if self.class_names is None:
            self.classification = self.classes[self.classes_count.argmax()]
        else:
            self.classification = self.class_names[self.classes[self.classes_count.argmax()]]
            
    