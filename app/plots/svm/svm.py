import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
import cvxopt
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
import itertools as it


def execute(ax=None, **kwargs):
    ax = ax or plt.gca()
    line, = ax.plot(np.arange(0.0, 5.0, 0.02))
    return line


def generate_data():
    X, y = make_blobs(n_samples=200, centers=2, random_state=0, cluster_std=0.60)
    y[y == 0] = -1
    tmp = np.ones(len(X))
    y = tmp * y
    # plt.scatter(X[:, 0], X[:, 1], c=y)
    # plt.show()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_train, X_test, y_train, y_test


def boundaries(x, w, b, c=0):
    return (-w[0] * x - b + c) / w[1]


def plot_boundaries(a0, b0, w, b):
    # Boundary
    a1 = boundaries(a0, w, b)
    b1 = boundaries(b0, w, b)
    plt.plot([a0, b0], [a1, b1], 'k')
    # First support vector
    a1 = boundaries(a0, w, b, 1)
    b1 = boundaries(b0, w, b, 1)
    plt.plot([a0, b0], [a1, b1], 'k--')
    # Second support vector
    a1 = boundaries(a0, w, b, -1)
    b1 = boundaries(b0, w, b, -1)
    plt.plot([a0, b0], [a1, b1], 'k--')


def plot_features(features, labels):
    # nu_labels = np.unique(labels)
    nu_features = len(features[0])
    plt.figure()
    for i in range(nu_features):
        for j in range(nu_features):
            if not i == j:
                plt.subplot(nu_features, nu_features, nu_features * i + j + 1)
                plt.scatter(features[:, i], features[:, j], c=labels)
    plt.show()


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

        # Convex optimization
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

        # Silent solvers output
        cvxopt.solvers.options['show_progress'] = False
        # Find solution for qiven coefficients
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

    def project(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        return np.sign(self.project(X))


if __name__ == "__main__":
    # fig, [plot1, plot2] = plt.subplots(nrows=2)
    # execute(plot1)
    # execute(plot2)
    # plt.show()

    dataset = load_iris()

    labels = dataset.target
    features = dataset.data
    list_labels = np.unique(labels)
    nu_labels = len(list_labels)
    nu_features = len(features[0])
    print(nu_labels)
    print(list_labels)
    '''
    print(features)
    print(labels)
    permutation = labels.argsort()
    labels = permutation[labels]
    features = permutation[features]

    print(features)
    print(labels)
    '''

    # print(type(labels))
    # print(type(features))

    # plot_features(features, labels)
    """
    classifier = LinearSVM()

    fe = features[(labels == 1) | (labels == 2)]
    la = labels[(labels == 1) | (labels == 2)]
    la = np.where(la == 1, -1, la)
    la = np.where(la == 2, 1, la)
    tmp = np.ones(len(fe))
    la = tmp * la
    print(fe[:, [0, 1]], la)
    print(len(la))
    plt.figure()
    plt.scatter(fe[:, 0], fe[:, 1], c=la, cmap='summer')
    classifier.training_model(fe[:, [0, 1]], la)
    print(classifier.w)
    a0 = min(fe[:, 0])
    b0 = max(fe[:, 0])
    a1_min = np.amin(fe)
    b1_max = np.amax(fe)
    plot_boundaries(a0, b0, classifier.w, classifier.b)
    plt.show()
    y_pred = classifier.predict(fe[:, [0, 1]])
    out = confusion_matrix(la, y_pred)
    # I assume classifier doesn't work if the accuracy is <= 50%
    
    # print(accuracy)
    """

    """
    svc = LinearSVC()
    svc.fit(fe, la)
    plt.scatter(fe[:, 0], fe[:, 1], c=la, cmap='winter');
    ax = plt.gca()
    xlim = ax.get_xlim()
    w = svc.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(xlim[0], xlim[1])
    yy = a * xx - svc.intercept_[0] / w[1]
    plt.plot(xx, yy)
    yy = a * xx - (svc.intercept_[0] - 1) / w[1]
    plt.plot(xx, yy, 'k--')
    yy = a * xx - (svc.intercept_[0] + 1) / w[1]
    plt.plot(xx, yy, 'k--')
    """

    # do it
    classifier = LinearSVM()

    feature_pair = features[:, [0, 1]]
    labels_pairs = list(it.combinations(list_labels, 2))
    print(labels_pairs)
    plt.figure()
    plt.scatter(features[:, 0], features[:, 2], c=labels, cmap='summer')
    for pair in labels_pairs:
        print(pair)
        # Prepare feature arrat for given label pair
        # fe = features[(labels == 1) | (labels == 2)]
        features1 = features[labels == pair[0]]
        features2 = features[labels == pair[1]]
        feat = np.concatenate([features1, features2])
        
        # Prepare label array for given label pair
        # la = labels[(labels == 1) | (labels == 2)]
        # la = np.where(la == 1, -1, la)
        # la = np.where(la == 2, 1, la)
        labels1 = np.ones(len(features1))
        labels2 = np.zeros(len(features2))
        labels2[labels2 == 0] = -1
        tmp = np.ones(len(features2))
        labels2 = tmp * labels2
        lab = np.concatenate([labels1, labels2])

        # for every combination of features

        # create list of all combinations of feature pairs
        list_features = np.array(range(nu_features))
        print(list_features)
        print(list(it.combinations(list_features, 2)))

        # make training model for every feature combinations and print

        a0 = min(feat[:, 0])
        b0 = max(feat[:, 0])

        classifier.training_model(feat[:, [0, 1]], lab)
        # maybe print in different way? trzeba wrócić do poprzednich labeli?
        # plt.scatter(feat[:, 0], feat[:, 1], c=lab, cmap='summer')
        # Check if classification worked (how many values were assigned correctly)
        check = classifier.predict(feat[:, [0, 1]])
        out = confusion_matrix(lab, check)
        accuracy = (out[1][1] + out[0][0]) / (out[1][1] + out[0][0] + out[1][0] + out[0][1])
        if accuracy > 0.5:
            plot_boundaries(a0, b0, classifier.w, classifier.b)

    plt.show()
    









    '''
    X = feature_pair[:100]
    y = labels[:100]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    classifier.training_model(X_test, y_test)
    '''


    '''
    svc = LinearSVC()
    svc.fit(X_test, y_test)
    y_pred = svc.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    # 
    '''
    # plot_boundaries(classifier.w, classifier.b)

    '''
    # Predict values
    y_predict = classifier.predict(X_test)

    print('My implementation')
    out = confusion_matrix(y_test, y_predict)
    print('Good predictions 1 class - ' + str(out[1][1]))
    print('Bad predictions 1 class - ' + str(out[0][1]))
    print('Good predictions -1 class - ' + str(out[0][0]))
    print('Bad predictions -1 class - ' + str(out[1][0]))
    
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='summer')

    plt.show()

    svc = LinearSVC()
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    confusion_matrix(y_test, y_pred)

    print('Sklearn')
    out2 = confusion_matrix(y_test, y_predict)
    print('Good predictions 1 class - ' + str(out2[1][1]))
    print('Bad predictions 1 class - ' + str(out2[0][1]))
    print('Good predictions -1 class - ' + str(out2[0][0]))
    print('Bad predictions -1 class - ' + str(out2[1][0]))
    '''
