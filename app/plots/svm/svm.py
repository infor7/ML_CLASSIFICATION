import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
import cvxopt


def execute(ax=None, **kwargs):
    ax = ax or plt.gca()
    line, = ax.plot(np.arange(0.0, 5.0, 0.02))
    return line


class LinearSVM:
    def __init__(self):
        # nothing so far
        awesomness = 100

    def training_model(self, X, y):
        n_samples, n_features = X.shape
        # P = X^T X
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = np.dot(X[i], X[j])

        P = cvxopt.matrix(np.outer(y, y) * K)
        # q = -1 (1xN)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        # A = y^T
        A = cvxopt.matrix(y, (1, n_samples))
        # b = 0
        b = cvxopt.matrix(0.0)
        # -1 (NxN)
        G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        # 0 (1xN)
        h = cvxopt.matrix(np.zeros(n_samples))

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        # Lagrange multipliers
        a = np.ravel(solution['x'])
        # Lagrange have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
        self.b /= len(self.a)
        # Weights
        self.w = np.zeros(n_features)
        for n in range(len(self.a)):
            self.w += self.a[n] * self.sv_y[n] * self.sv[n]

    def generate_data(self):
        X, y = make_blobs(n_samples=250, centers=2, random_state=0, cluster_std=0.60)
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.show()
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    #fig, [plot1, plot2] = plt.subplots(nrows=2)
    #execute(plot1)
    #execute(plot2)
    #plt.show()
    classifier = LinearSVM()
    X_train, X_test, y_train, y_test = classifier.generate_data()
