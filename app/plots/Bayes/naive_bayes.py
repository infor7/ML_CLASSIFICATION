import matplotlib.pyplot as plt
from lib.naive_bayes import NaiveBayesGaussian, NaiveBayesMultinomial
import lib.tools as tools
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
import numpy as np
from itertools import permutations
from sklearn.naive_bayes import GaussianNB, MultinomialNB


def execute(ax=None, **kwargs):
    if kwargs.get("dataset") is not None:
        dataset = load_iris()
    else:
        dataset = load_iris()
    accuracy, indexes = accuracy_comparison(dataset.data, dataset.target)

    ax = ax or plt.gca()
    line, = ax.plot(indexes, accuracy, 'k', indexes, accuracy, 'bo')
    plt.title("Accuracy vs K value")
    return line


def accuracy_of_gaussian(data_split, labels_split):
    nb = NaiveBayesGaussian()
    accuracies = []
    clf = GaussianNB()
    accuracies, sklearn_accuracies = tools.accuracy_of_method(data_split, labels_split, nb, sklearn_class=clf)
    return np.mean(accuracies), np.mean(sklearn_accuracies)


def accuracy_of_multinomial(data_split, labels_split, num_of_bins=100):
    # folds = 3
    # data_split, labels_split = tools.cross_validation_split(dataset=dataset, labels=labels, folds=folds)
    nb = NaiveBayesMultinomial(num_of_bins=num_of_bins)
    clf = MultinomialNB(alpha=0.0001, fit_prior=True)

    accuracies, sklearn_accuracies = tools.accuracy_of_method(data_split, labels_split, nb, sklearn_class=clf)
    return np.mean(accuracies), np.mean(sklearn_accuracies)


def accuracy_for_letters():
    with open('../../data_sources/letter-recognition.data', 'r') as f:
        dataset, labels = tools.load_text_file(f, label_index=0, labels_numeric=False)
        folds = 3
        data_split, labels_split = tools.cross_validation_split(dataset=dataset, labels=labels, folds=folds)
        # iris = load_iris()
        print("Multinomial:")
        accuracy_of_multinomial(data_split,labels_split, num_of_bins=11)
        print("Gaussian:")
        accuracy_of_gaussian(data_split,labels_split)


def accuracy_for_wines():
    with open('../../data_sources/Wine.csv', 'r') as f:
        dataset, labels = tools.load_text_file(f, label_index=0, dtype=float, labels_numeric=True)
        # iris = load_iris()
        # labels -= 1
        folds = 3
        data_split, labels_split = tools.cross_validation_split(dataset=dataset, labels=labels, folds=folds)
        # iris = load_iris()
        print("Multinomial:")
        accuracy_of_multinomial(data_split,labels_split, num_of_bins=15)
        print("Gaussian:")
        accuracy_of_gaussian(data_split,labels_split)


def accuracy_for_trees():
    with open('../../data_sources/covtype.csv', 'r') as f:
        dataset, labels = tools.load_text_file(f, label_index=-1, dtype=float, labels_numeric=True)
        # iris = load_iris()
        # labels -= 1
        folds = 3
        data_split, labels_split = tools.cross_validation_split(dataset=dataset, labels=labels, folds=folds)
        # iris = load_iris()
        print("Multinomial:")
        accuracy_of_multinomial(data_split,labels_split)
        print("Gaussian:")
        accuracy_of_gaussian(data_split,labels_split)


def accuracy_for_cancer():
    with open('../../data_sources/kag_risk_factors_cervical_cancer.csv', 'r') as f:
        dataset, labels = tools.load_text_file(f, label_index=28,  dtype=float)
        # labels = np.array(dataset[:,28], dtype=int)
        # dataset=np.append(dataset[:,:28], dataset[:, 29:], axis=1)
        # iris = load_iris()
        # labels -= 1
        folds = 3
        data_split, labels_split = tools.cross_validation_split(dataset=dataset, labels=labels, folds=folds)
        # iris = load_iris()
        print("Multinomial:")
        accuracy_of_multinomial(data_split,labels_split)
        print("Gaussian:")
        accuracy_of_gaussian(data_split,labels_split)

def accuracy_for_iris():
    iris = load_iris()
    folds = 3
    data_split, labels_split = tools.cross_validation_split(dataset=iris.data, labels=iris.target, folds=folds)
    print("Multinomial:")
    accuracy_of_multinomial(data_split, labels_split)
    print("Gaussian:")
    accuracy_of_gaussian(data_split, labels_split)


if __name__ == "__main__":
    # iris = load_iris()
    # accuracy_of_gaussian(iris.data, iris.target)
    print("LETTERS: ")
    accuracy_for_letters()
    print()
    print("WINES: ")
    accuracy_for_wines()
    print()
    # print("TREES: ")
    # accuracy_for_trees()
    # print()
    print("CANCER: ")
    accuracy_for_cancer()
    print()
    print("IRIS: ")
    accuracy_for_iris()
    wait = input("PRESS ENTER TO CONTINUE.")
