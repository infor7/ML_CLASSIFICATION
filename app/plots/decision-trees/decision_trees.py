import matplotlib.pyplot as plt
import numpy as np
import lib.node 
from sklearn.datasets import load_iris
from lib.tree import Tree

def execute(ax=None, **kwargs):
    ax = ax or plt.gca()
    line, = ax.plot (np.arange(0.0, 5.0, 0.02))
    return line

def plot():
    pass


if __name__ == "__main__":
    
    iris = load_iris()
	tree = Tree(iris.target, iris.data)
	nodes = tree.fit()
	print(tree.classify(iris.target[123],nodes))

    fig, [plot1, plot2] = plt.subplots(nrows=2)
    execute(plot1)
    execute(plot2)
    plt.show()