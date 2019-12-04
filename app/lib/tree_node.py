import numpy as np


class Node:

    def __init__(self, target, data, metric = "Gini", class_names = None, level = 0):
        self.target = target
        self.data = data
        self.final = False
        
        if class_names is not None:
            self.class_names = class_names
        
        self.assign_class() # also assigns members count and classes count 
        
        if metric is "Gini":
            self.gini_score()
            self.score = self.gini
        elif metric is "Entropy":
            self.entropy_score()
            self.score = self.entropy
        else:
            self.score = None



    def assign_class(self):
        self.classes, self.class_count = np.unique(self.target, return_counts = True)
        self.count = self.class_count.sum()
        if self.class_names is None:
            self.classification = self.classes[self.class_count.argmax()]
        else:
            self.classification = self.class_names[self.classes[self.class_count.argmax()]]
            

    def assign_splitter(self, column, value):
        self.splitter_column = column
        self.splitter_value = value
        

    def gini_score(self):
        self.gini = 1 - np.sum(np.square(self.class_count / self.count))
    

    def entropy_score(self):
        pi = self.class_count / self.count
        self.entropy = np.sum(- pi * np.log2(pi))
    

    def set_final(self):
        self.final = True
    


            
    