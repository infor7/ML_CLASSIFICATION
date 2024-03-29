import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
import cvxopt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
import itertools as it
from collections import Counter
from sklearn.decomposition import PCA


def two_to_two_example():
    classifier = LinearSVM()
    X_train, X_test, y_train, y_test = generate_data()
    classifier.training_model(X_train, y_train)

    plt.xlim(-1, 4)
    #plt.ylim(1, 5)
    plt.title("SVM - example")

    a0 = -4
    b0 = 4

    # Boundary
    a1 = boundaries(a0, classifier.w, classifier.b)
    b1 = boundaries(b0, classifier.w, classifier.b)
    plt.plot([a0, b0], [a1, b1], 'k')

    # First support vector
    a1 = boundaries(a0, classifier.w, classifier.b, 1)
    b1 = boundaries(b0, classifier.w, classifier.b, 1)
    plt.plot([a0, b0], [a1, b1], 'k--')

    # Second support vector
    a1 = boundaries(a0, classifier.w, classifier.b, -1)
    b1 = boundaries(b0, classifier.w, classifier.b, -1)
    plt.plot([a0, b0], [a1, b1], 'k--')

    # Plot training data
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='winter')

    # Predict values
    y_predict = classifier.predict(X_test)

    out = confusion_matrix(y_test, y_predict)
    print('Good predictions 1 class - ' + str(out[1][1]))
    print('Bad predictions 1 class - ' + str(out[0][1]))
    print('Good predictions -1 class - ' + str(out[0][0]))
    print('Bad predictions -1 class - ' + str(out[1][0]))

    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='summer')\

    plt.show()


def dimentionality_reduction_example():
    # Load dataset
    dataset = load_iris()
    print(dataset.target)
    print(len(dataset.target))
    print(len(dataset.data[0]))

    # Prepare data
    labels = dataset.target
    features = dataset.data
    list_labels = np.unique(labels)
    nu_labels = len(list_labels)
    nu_features = len(features[0])
    print('This dataset has ' + str(nu_features) + ' features and ' + str(nu_labels) + ' labels')

    # Perform dimentionality reduction - reduce number of features to 2
    # WARNING! With lots of features it's useless!
    # features_reduced = PCA(n_components=2).fit_transform(features)
    # print(X_reduced)
    # features = features_reduced
    # nu_features = len(features[0])

    # Divide data into train and test
    features, features_test, labels, labels_test = train_test_split(features, labels, random_state=0)

    # Create classifier
    classifier = LinearSVM()

    # Create list of all combinations of feature pairs
    list_features = np.array(range(nu_features))
    features_pairs = list(it.combinations(list_features, 2))

    # features_pairs = features_pairs[0::5]

    print(features_pairs)
    print(len(features_pairs))

    # Create list of all combinations of labels pairs
    labels_pairs = list(it.combinations(list_labels, 2))
    print(labels_pairs)
    print(len(labels_pairs))

    # if there is to much labels (to long calculations) you can take only pairs
    # labels_pairs = [(i, i+1) for i in range(nu_labels-1)]
    # print(labels_pairs)
    # plt.figure()

    # labels_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8), (8, 9), (0, 9)]
    '''
    # Determine subplot grid
    if nu_features % 2 == 0:
        columns = nu_features / 2
    elif nu_features % 2 == 1:
        columns = (nu_features - 1) / 2
    rows = len(features_pairs) / columns
    # features_pairs = f_pairs_extended
    '''
    
    result = []

    counter = 0
    # For every combination of features
    for pair_f in features_pairs:
        print(counter)
        # Determine proper subplot to plot on
        # index = features_pairs.index(pair_f)

        # Configure plots
        '''
        title = 'Features ' + str(pair_f)
        plt.subplot(rows, columns, index + 1)            #.set_aspect('equal')
        plt.xlim(min(features[:, pair_f[0]]) - 0.5, max(features[:, pair_f[0]]) + 0.5)
        plt.ylim(min(features[:, pair_f[1]]) - 0.5, max(features[:, pair_f[1]]) + 0.5)
        plt.title(title)

        # Plot data points
        plt.scatter(features[:, pair_f[0]], features[:, pair_f[1]], c=labels, cmap='summer')
        '''
        # For every combination of labels
        for pair_l in labels_pairs:
            # print(pair_l)
            # Prepare feature array for given label pair
            feat = features[(labels == pair_l[0]) | (labels == pair_l[1])]

            # Approach one-vs-therest
            #feat = features
            #lab = labels
            #lab = np.where(lab == counter, -1, lab)
            #lab = np.where((lab != counter) & (lab != -1), 1, lab)

            # Prepare label array for given label pair
            lab = labels[(labels == pair_l[0]) | (labels == pair_l[1])]
            lab = np.where(lab == pair_l[0], -1, lab)
            lab = np.where(lab == pair_l[1], 1, lab)
            tmp = np.ones(len(feat))
            lab = tmp * lab

            # Calculate range of boundary lines on x axis
            # a0 = min(feat[:, pair_f[0]])
            # b0 = max(feat[:, pair_f[0]])

            # List of currently unused labels
            # unused_labels = list_labels[(list_labels != pair_l[0]) & (list_labels != pair_l[1])]
            # print(unused_labels)

            # Run classifier
            classifier.training_model(feat[:, [pair_f[0], pair_f[1]]], lab)

            # Manage classification results
            labels_predicted = classifier.predict(features_test[:, [pair_f[0], pair_f[1]]])
            labels_predicted = np.where(labels_predicted == 1, pair_l[1], labels_predicted)
            labels_predicted = np.where(labels_predicted == -1, pair_l[0], labels_predicted)
            # print(labels_test)
            # print(labels_predicted)
            result.append(list(labels_predicted))

            # Check if classification worked (how many values were assigned correctly)
            # out = confusion_matrix(labels_test, labels_predicted)
            # accuracy = (out[1][1] + out[0][0]) / (out[1][1] + out[0][0] + out[1][0] + out[0][1])
            # print(accuracy)
            # if accuracy > 0.5:

            # plot_boundaries(a0, b0, classifier.w, classifier.b)
        counter = counter + 1

    # plt.show()

    # Calculate final result - the most common classification for each data point
    final = [most_common([item[i] for item in result]) for i in range(len(result[0]))]
    # print(final)

    out = confusion_matrix(labels_test, final)
    print(out)
    print('Result')
    for i in range(len(out[0])):
        print("Good predictions of " + str(i) + ' class - ' + str(out[i][i]))
        out[i][i] = 0
        print("Bad predictions of " + str(i) + ' class - ' + str(sum(out[i])))
    acc = accuracy_score(labels_test, final)
    print("Accuracy = " + str(acc))

    # Sklearn approach

    svc = LinearSVC()
    svc.fit(features, labels)
    y_pred = svc.predict(features_test)
    print(confusion_matrix(labels_test, y_pred))
    acc = accuracy_score(labels_test, y_pred)
    print("Accuracy = " + str(acc))




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
    '''
    # First support vector
    a1 = boundaries(a0, w, b, 1)
    b1 = boundaries(b0, w, b, 1)
    plt.plot([a0, b0], [a1, b1], 'k--')
    # Second support vector
    a1 = boundaries(a0, w, b, -1)
    b1 = boundaries(b0, w, b, -1)
    plt.plot([a0, b0], [a1, b1], 'k--')
    '''


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


def reverse(tuples):
    new_tup = ()
    for k in reversed(tuples):
        new_tup = new_tup + (k,)
    return new_tup


def most_common(lst):
    data = Counter(lst)
    return max(lst, key=data.get)


class LinearSVM:
    def __init__(self):
        self.C = 1

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
        # G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        # 0 (1xN)
        # h = cvxopt.matrix(np.zeros(n_samples))

        # G and h for C
        tmp1 = np.diag(np.ones(n_samples) * -1)
        tmp2 = np.identity(n_samples)
        G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
        tmp1 = np.zeros(n_samples)
        tmp2 = np.ones(n_samples) * self.C
        h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

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

    dimentionality_reduction_example()

    #two_to_two_example()


    '''
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
