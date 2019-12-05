import collections
import numpy as np

class Node:
    """ Node/Leaf for Decision Tree"""
    def __init__(self, target, data):
        self.target = target
        self.data = data
        self.leaf = False
        self.assign_class()
        pass


    def set_as_final(self):
        self.leaf = True


    def assign_class(self):
        self.classes, self.classes_count = np.unique(self.target, return_counts = True)
        self.count = self.classes_count.sum()
        self.result = self.classes[self.classes_count.argmax()]


    @staticmethod
    def entropy(target):
        classes, classes_count = np.unique(target, return_counts = True)
        count = classes_count.sum()
        p = classes_count / count
        entropy = np.sum(- p * np.log2())
        return entropy

    
    @staticmethod
    def gini(target):
        classes, classes_count = np.unique(target, return_counts = True)
        count = classes_count.sum()
        p = classes_count / count
        gini = 1 - np.sum(np.square(p))
        return gini


    @staticmethod
    def split_data(target, data, feature, value):
        t_target = target[data[:,feature] >= value]
        t_data = data[data[:,feature] >= value]
        f_target = target[data[:,feature] < value]
        f_data = data[data[:,feature] < value]
        return t_target, t_data, f_target, f_data


    @staticmethod
    def prune_feature(data, feature):
        return np.delete(data, feature, 1)



