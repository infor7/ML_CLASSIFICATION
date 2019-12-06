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

x = 0

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

                print("new_gain = " + str(new_gain))

                if new_gain>gain and len(t_target)>=self.min_members and len(f_target)>=self.min_members:
                    split_datasets = [t_target, t_data, f_target, f_data]
                    gain = new_gain
                    split_feature = feature
                    split_value = val
        print("---------------")
        print(gain)
        print("---------------")
        if gain > self.min_gain and len(target) > self.min_split and (depth < self.max_depth or self.max_depth == 0):
            t_children = self.fit_cart(split_datasets[0], split_datasets[1], depth + 1)
            f_children = self.fit_cart(split_datasets[2], split_datasets[3], depth + 1)
            return Node(target, data, False, depth, t_children, f_children, feature, val)
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
                    split_datasets = (t_target, t_data, f_target, f_data)
                    gain = new_gain
                    split_feature = feature
                    split_value = val
        
        if gain > self.min_gain and len(target) >= self.min_split and (depth < self.max_depth or self.max_depth == 0) and features_count != 0:
            t_children = fit_id3(t_target, t_data, depth + 1)
            f_children = fit_id3(f_target, f_data, depth + 1)
            return Node(target, data, False, depth, t_children, f_children, feature, val)
        else:
            return Node(target, data, True, depth)


    def fit(self):
        if self.metric == "gini":
            return self.fit_cart(self.root.target, self.root.data, 0)
        else:
            return self.fit_id3(self.root.target, self.root.data, 0)

    
    def classify(self, data, node):
        if self.metric == "gini":
            if node.leaf is True: 
                print("a")
                return node.result
            else:
                if data[node.feature] >= node.value:
                    print("b")
                    next_node = node.t_children
                else:
                    print("c")
                    next_node = node.f_children
                return self.classify(data, next_node)
        else:
            if node.leaf is True:
                return node.result
            else:
                if data[node.feature] >= node.value:
                    next_node = node.t_children
                else:
                    next_node = node.f_children
                new_data = np.delete(data, feature, 1)
                return self.classify(new_data, next_node)



        


    


# Testcode
def main():
    print("Running Decision Tree Example: iris")
    iris = load_iris()
    iris_tree_g = Tree(iris.target, iris.data)
    nodes = iris_tree_g.fit()

    for row in iris.data:
        test = iris_tree_g.classify(row, nodes)
        print(test)


    # print("Running Decision Tree Example: wine")
    # wine = load_wine()


    # print("Running Decision Tree Example: cancer")
    # cancer = load_breast_cancer()

    # # Warning - very slow
    # print("Running Decision Tree Example: digits")



if __name__ == "__main__":
    main()