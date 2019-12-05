from IPython import get_ipython
# For testing in IPython
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
    def __init__(self, target, data, metric = "gini", max_depth = 10, min_split = 2, min_members = 1, min_gain = 0):
        self.root = Node(target, data)
        self.metric = metric
        self.max_depth = max_depth
        self.min_split = min_split
        self.min_members = min_members
        self.min_gain = min_gain

        print("Root created")


    # def spawn_nodes(self, target, data, parent_score, metric ):
    #     pass



    def fit(self):
        if self.metric == "gini":
            parent_score = Node.gini(self.root.target)
        else:
            parent_score = Node.entropy(self.root.target)

        gain = 0.0
        split_feature = None
        split_value = None
        split_datasets = None

        features_count = self.root.data.shape[1]
        
        for feature in range(0, features_count):
            vals, counts = np.unique(self.root.data[:,feature], return_counts = True)
            for val in vals:
                t_target, t_data, f_target, f_data = Node.split_data(self.root.target, self.root.data, feature, val)
                p = (len(t_target) / (len(t_target) + len(f_target)))
                if self.metric == "gini":
                    new_gain = parent_score - p*Node.gini(t_target) - (1-p)*Node.gini(f_target)
                    
                else:
                    new_gain = parent_score - p*Node.entropy(t_target) - (1-p)*Node.entropy(f_target)

                if new_gain > gain and len(t_target) > 0 and len(f_target) > 0:
                    gain = new_gain
                    split_feature = feature
                    split_value = val
                    split_datasets = (t_target, t_data, f_target, f_data)


        if gain > self.min_gain:


        def classify(data):
            pass


    


# Testcode
def main():
    print("Running Decision Tree Example: iris")
    iris = load_iris()
    iris_tree_g = Tree(iris.target, iris.data)
    iris_tree_g.fit()

    print("Running Decision Tree Example: wine")
    wine = load_wine()


    print("Running Decision Tree Example: cancer")
    cancer = load_breast_cancer()

    # Warning - very slow
    print("Running Decision Tree Example: digits")


if __name__ == "__main__":
    main()