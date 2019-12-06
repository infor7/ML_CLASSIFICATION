from IPython import get_ipython
# For interactive testing in IPython
get_ipython().magic('reset -f')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('aimport node')
get_ipython().magic('autoreload 1')

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits
from sklearn.datasets import load_wine

# Self implemented classes
from node import Node



class Tree:
    def __init__(self, target, data, metric = "gini", max_depth = 0, min_split = 2, min_members = 1, min_gain = 0):
        self.root = Node(target, data)
        self.metric = metric
        self.max_depth = max_depth
        self.min_split = min_split
        self.min_members = min_members
        self.min_gain = min_gain


    
    def fit_cart(self, target, data, depth):
        """ Basic CART algorithm using gini impurity"""
        parent_score = Node.gini(target)
        gain = 0.0
        split_feature = None
        split_value = None
        split_datasets = None
        features_count = data.shape[1]

        for feature in range(0, features_count):
            vals, counts = np.unique(data[:,feature], return_counts = True)
            for val in vals:
                t_target, t_data, f_target, f_data = Node.split_data(target, data, feature, val)
                p = (len(t_target) / (len(t_target) + len(f_target)))
                new_gain = parent_score - p*Node.gini(t_target) - (1-p)*Node.gini(f_target)

                if new_gain>gain and len(t_target)>=self.min_members and len(f_target)>=self.min_members:
                    split_datasets = [t_target, t_data, f_target, f_data]
                    gain = new_gain
                    split_feature = feature
                    split_value = val

        if gain > self.min_gain and len(target) > self.min_split and (depth < self.max_depth or self.max_depth == 0):
            t_children = self.fit_cart(split_datasets[0], split_datasets[1], depth + 1)
            f_children = self.fit_cart(split_datasets[2], split_datasets[3], depth + 1)
            return Node(target, data, False, depth, t_children, f_children, split_feature, split_value)
        else:
            return Node(target, data, True, depth)

    
    def fit_id3(self, target, data, depth):
        """Pseudo ID3 implementation
         - prunes used features, but can works with non-binary targets"""
        parent_score = Node.entropy(target)
        gain = 0.0
        split_feature = None
        split_value = None
        split_datasets = None
        features_count = data.shape[1]

        for feature in range(0, features_count):
            vals, counts = np.unique(data[:,feature], return_counts = True)
            for val in vals:
                t_target, t_data, f_target, f_data = Node.split_data(target, data, feature, val)
                p = (len(t_target) / (len(t_target) + len(f_target)))
                new_gain = parent_score - p*Node.entropy(t_target) - (1-p)*Node.entropy(f_target)

                if new_gain>gain and len(t_target)>=self.min_members and len(f_target)>=self.min_members:
                    t_data = Node.prune_feature(t_data, feature)
                    f_data = Node.prune_feature(f_data, feature)
                    split_datasets = [t_target, t_data, f_target, f_data]
                    gain = new_gain
                    split_feature = feature
                    split_value = val

        if gain > self.min_gain and len(target) > self.min_split and (depth < self.max_depth or self.max_depth == 0) and features_count != 0:
            t_children = self.fit_cart(split_datasets[0], split_datasets[1], depth + 1)
            f_children = self.fit_cart(split_datasets[2], split_datasets[3], depth + 1)
            return Node(target, data, False, depth, t_children, f_children, split_feature, split_value)
        else:
            return Node(target, data, True, depth)


    def fit_id3_simple(self, target, data, depth):
        """VERY Pseudo ID3 implementation
         - no prunning, but can works with non-binary targets"""
        parent_score = Node.entropy(target)
        gain = 0.0
        split_feature = None
        split_value = None
        split_datasets = None
        features_count = data.shape[1]

        for feature in range(0, features_count):
            vals, counts = np.unique(data[:,feature], return_counts = True)
            for val in vals:
                t_target, t_data, f_target, f_data = Node.split_data(target, data, feature, val)
                p = (len(t_target) / (len(t_target) + len(f_target)))
                new_gain = parent_score - p*Node.entropy(t_target) - (1-p)*Node.entropy(f_target)

                if new_gain>gain and len(t_target)>=self.min_members and len(f_target)>=self.min_members:
                    split_datasets = [t_target, t_data, f_target, f_data]
                    gain = new_gain
                    split_feature = feature
                    split_value = val

        if gain > self.min_gain and len(target) > self.min_split and (depth < self.max_depth or self.max_depth == 0):
            t_children = self.fit_cart(split_datasets[0], split_datasets[1], depth + 1)
            f_children = self.fit_cart(split_datasets[2], split_datasets[3], depth + 1)
            return Node(target, data, False, depth, t_children, f_children, split_feature, split_value)
        else:
            return Node(target, data, True, depth)


    def fit(self):
        if self.metric == "gini":
            return self.fit_cart(self.root.target, self.root.data, 0)
        elif self.metric == "entropy":
            return self.fit_id3(self.root.target, self.root.data, 0)
        else:
            return self.fit_id3_simple(self.root.target, self.root.data, 0)

    
    def classify(self, data, node):
        if self.metric == "gini":
            if node.leaf is True: 
                return node.result
            else:
                if data[node.feature] >= node.value:
                    next_node = node.t_children
                else:
                    next_node = node.f_children
                return self.classify(data, next_node)
        elif self.metric == "entropy":
            if node.leaf is True:
                return node.result
            else:
                if data[node.feature] <= node.value:
                    next_node = node.t_children
                else:
                    next_node = node.f_children
                new_data = Node.prune_feature(data, node.feature)
                return self.classify(new_data, next_node)
        else:
            if node.leaf is True: 
                return node.result
            else:
                if data[node.feature] >= node.value:
                    next_node = node.t_children
                else:
                    next_node = node.f_children
                return self.classify(data, next_node)
        

    @staticmethod
    def cross_validate(folds, target, data, test_size = 0.25, metric = "gini", max_depth = 0, min_split = 2, min_members = 1, min_gain = 0.0):
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        for fold in range(0,folds):
            data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=test_size)
            tree = Tree(target_train, data_train, metric, max_depth, min_split, min_members, min_gain)
            nodes = tree.fit()
            predictions = []
            
            for row in data_test:
                predictions.append(tree.classify(row, nodes))
            accuracy += accuracy_score(target_test, predictions)
            
            if len(np.unique(target)) == 2:
                precision += precision_score(target_test, predictions)
                recall += recall_score(target_test, predictions)

        accuracy = accuracy / folds
        precision = precision / folds
        recall = recall / folds
        return accuracy, precision, recall



# Testcode
def main():
    # """Cross-validation test for 3 sets and simple test for digits"""
    print("Running Decision Tree Example: iris")
    iris = load_iris()
    accuracy, prc, rc = Tree.cross_validate(100,iris.target, iris.data, 0.25, "gini")
    print("----------------")
    print(accuracy)
    print("----------------")
    accuracy, prc, rc = Tree.cross_validate(100,iris.target, iris.data, 0.25, "simple")
    print("----------------")
    print(accuracy)
    print("=================")
    print("Running Decision Tree Example: wine")
    wine = load_wine() 
    accuracy, prc, rc = Tree.cross_validate(20,wine.target, wine.data, 0.25, "gini")
    print("----------------")
    print(accuracy)
    print("----------------")
    accuracy, prc, rc = Tree.cross_validate(20,wine.target, wine.data, 0.25, "simple")
    print("----------------")
    print(accuracy)
    print("=================")
    print("Running Decision Tree Example: cancer")
    cancer = load_breast_cancer() 
    accuracy, prc, rc = Tree.cross_validate(10,cancer.target, cancer.data, 0.25, "gini")
    print("----------------")    
    print(accuracy)
    print(prc)
    print(rc)
    print("----------------")
    accuracy, prc, rc = Tree.cross_validate(10,cancer.target, cancer.data, 0.25, "simple")
    print(accuracy)
    print(prc)
    print(rc)
    print("=================")


if __name__ == "__main__":
    main()