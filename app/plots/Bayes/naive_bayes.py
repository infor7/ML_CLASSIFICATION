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
    data_split, labels_split = cross_validation_split(dataset=dataset, labels=labels, folds=folds)
    nb = NaiveBayes()
    accuracies = []
    clf = GaussianNB()
    # for i in permutations(range(folds),2):
    #     predictions = nb.naive_bayes(data_split[i[0]], labels_split[i[0]], data_split[i[1]])
    #     accuracy = sum(predictions==labels_split[i[1]])/len(labels_split[i[1]])
    #     accuracies.append(accuracy)
    #     print("my: ", accuracy)
    #     clf.fit(data_split[i[0]], labels_split[i[0]])
    #     accuracy = sum(clf.predict(data_split[i[1]])==labels_split[i[1]])/len(labels_split[i[1]])
    #     print("sklearn: ", accuracy)
    for i in range(folds):
        rest = [x for x in range(folds) if x != i]
        # works only for folds = 3
        # training_data = np.array(data_split)[rest].reshape(-1, np.array(data_split)[rest].shape[-1])
        training_data = np.append(data_split[rest[0]], data_split[rest[1]], axis=0)
        # training_labels = np.array(labels_split)[rest].flatten()
        training_labels = np.append(labels_split[rest[0]], labels_split[rest[1]], axis=0)
        testing_data = data_split[i]
        testing_labels = labels_split[i]
        predictions = nb.naive_bayes_gaussian(training_data, training_labels, testing_data)

        accuracy = sum(predictions==testing_labels)/len(testing_data)
        accuracies.append(accuracy)
        print("my: ", accuracy)
        clf.fit(training_data, training_labels)
        accuracy = sum(clf.predict(testing_data)==testing_labels)/len(testing_labels)
        print("sklearn: ", accuracy)

    print(np.mean(accuracies))

def cross_validation_split(dataset, labels, folds=3):
    dataset_copy = np.array(dataset)
    indices = np.random.permutation(np.arange(len(dataset)))
    dataset_copy = dataset_copy[indices]
    labels_copy = labels[indices]
    dataset_split = np.array_split(dataset_copy, folds)
    labels_split = np.array_split(labels_copy, folds)
    return dataset_split, labels_split

def accuracy_of_multinomial(dataset, labels):
    folds = 3
    data_split, labels_split = cross_validation_split(dataset=dataset, labels=labels, folds=folds)
    nb = NaiveBayes()
    clf = MultinomialNB(alpha=0.0001, fit_prior=True)
    accuracies = []
    for i in range(folds):
        rest = [x for x in range(folds) if x != i]
        # works only for folds = 3
        # training_data = np.array(data_split)[rest].reshape(-1, np.array(data_split)[rest].shape[-1])
        training_data = np.append(data_split[rest[0]], data_split[rest[1]], axis=0)
        # training_labels = np.array(labels_split)[rest].flatten()
        training_labels = np.append(labels_split[rest[0]], labels_split[rest[1]], axis=0)
        testing_data = data_split[i]
        testing_labels = labels_split[i]
        predictions = nb.naive_bayes_multinomial(training_data, training_labels, testing_data)
        accuracy = sum(predictions==testing_labels)/len(testing_data)
        accuracies.append(accuracy)
        print("my: ", accuracy)
        clf.fit(training_data, training_labels)
        accuracy = sum(clf.predict(testing_data)==testing_labels)/len(testing_data)
        print("sklearn: ", accuracy)

def accuracy_for_letters():
    with open('../../../datasets/letter-recognition.data', 'r') as f:
        dataset, labels = tools.load_text_file(f, first_columb_labels=True)
        # iris = load_iris()
        print("Multinomial:")
        accuracy_of_multinomial(dataset,labels)
        print("Gaussian:")
        accuracy_of_gaussian(dataset,labels)

def accuracy_for_wines():
    with open('../../../datasets/Wine.csv', 'r') as f:
        dataset, labels = tools.load_text_file(f, first_columb_labels=True, dtype=float, labels_numeric=True)
        # iris = load_iris()
        labels -= 1
        print("Multinomial:")
        accuracy_of_multinomial(dataset, labels)
        print("Gaussian:")
        accuracy_of_gaussian(dataset, labels)

if __name__ == "__main__":
    # iris = load_iris()
    # accuracy_of_gaussian(iris.data, iris.target)
    # accuracy_for_letters()
    accuracy_for_wines()