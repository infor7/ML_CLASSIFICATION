import matplotlib.pyplot as plt
from lib.naive_bayes import NaiveBayes
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


def accuracy_of_gaussian(dataset, labels):
    folds = 3
    data_split, labels_split = tools.cross_validation_split(dataset=dataset, labels=labels, folds=folds)
    nb = NaiveBayes()
    accuracies = []
    clf = GaussianNB()
    accuracies, sklearn_accuracies = tools.accuracy_of_method(data_split, labels_split, nb.naive_bayes_gaussian, sklearn_class=clf)
    # for i in permutations(range(folds),2):
    #     predictions = nb.naive_bayes(data_split[i[0]], labels_split[i[0]], data_split[i[1]])
    #     accuracy = sum(predictions==labels_split[i[1]])/len(labels_split[i[1]])
    #     accuracies.append(accuracy)
    #     print("my: ", accuracy)
    #     clf.fit(data_split[i[0]], labels_split[i[0]])
    #     accuracy = sum(clf.predict(data_split[i[1]])==labels_split[i[1]])/len(labels_split[i[1]])
    #     print("sklearn: ", accuracy)
    return np.mean(accuracies), np.mean(sklearn_accuracies)


def accuracy_of_multinomial(dataset, labels):
    folds = 3
    data_split, labels_split = tools.cross_validation_split(dataset=dataset, labels=labels, folds=folds)
    nb = NaiveBayes()
    clf = MultinomialNB(alpha=0.0001, fit_prior=True)

    accuracies, sklearn_accuracies = tools.accuracy_of_method(data_split, labels_split, nb.naive_bayes_multinomial, sklearn_class=clf)
    return np.mean(accuracies), np.mean(sklearn_accuracies)


def accuracy_for_letters():
    with open('../../../datasets/letter-recognition.data', 'r') as f:
        dataset, labels = tools.load_text_file(f, first_column_labels=True, labels_numeric=False)
        # iris = load_iris()
        print("Multinomial:")
        accuracy_of_multinomial(dataset,labels)
        print("Gaussian:")
        accuracy_of_gaussian(dataset,labels)


def accuracy_for_wines():
    with open('../../../datasets/Wine.csv', 'r') as f:
        dataset, labels = tools.load_text_file(f, first_column_labels=True, dtype=float, labels_numeric=True)
        # iris = load_iris()
        labels -= 1
        print("Multinomial:")
        accuracy_of_multinomial(dataset, labels)
        print("Gaussian:")
        accuracy_of_gaussian(dataset, labels)


def accuracy_for_trees():
    with open('../../../datasets/covtype.csv', 'r') as f:
        dataset, labels = tools.load_text_file(f, last_column_labels=True, dtype=float, labels_numeric=True)
        # iris = load_iris()
        labels -= 1
        print("Multinomial:")
        accuracy_of_multinomial(dataset, labels)
        print("Gaussian:")
        accuracy_of_gaussian(dataset, labels)


def accuracy_for_cancer():
    with open('../../../datasets/kag_risk_factors_cervical_cancer.csv', 'r') as f:
        dataset, labels = tools.load_text_file(f, dtype=float)
        labels = np.array(dataset[:,28], dtype=int)
        dataset=np.append(dataset[:,:28], dataset[:, 29:], axis=1)
        # iris = load_iris()
        # labels -= 1
        print("Multinomial:")
        accuracy_of_multinomial(dataset, labels)
        print("Gaussian:")
        accuracy_of_gaussian(dataset, labels)

def accuracy_for_iris():
    iris = load_iris()
    print("Multinomial:")
    accuracy_of_multinomial(iris.data, iris.target)
    print("Gaussian:")
    accuracy_of_gaussian(iris.data, iris.target)


if __name__ == "__main__":
    # iris = load_iris()
    # accuracy_of_gaussian(iris.data, iris.target)
    # accuracy_for_letters()
    # accuracy_for_wines()
    # accuracy_for_trees()
    accuracy_for_cancer()
    # accuracy_for_iris()