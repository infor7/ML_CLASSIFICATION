import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_blobs


def execute(ax=None, **kwargs):
    ax = ax or plt.gca()
    line, = ax.plot(np.arange(0.0, 5.0, 0.02))
    return line


class LinearSVM:
    def __init__(self):
        # nothing so far
        awesomness = 100

    def generate_data(self):
        X, y = make_blobs(n_samples=100, n_features=2, centers=None, cluster_std=1.0, center_box=(-10.0, 10.0),
                          shuffle=True, random_state=None)
        plt.scatter(X[:, 0], X[:, 1], c=y)


if __name__ == "__main__":
    fig, [plot1, plot2] = plt.subplots(nrows=2)
    execute(plot1)
    execute(plot2)
    plt.show()
    classifier = LinearSVM()
    classifier.generate_data()
