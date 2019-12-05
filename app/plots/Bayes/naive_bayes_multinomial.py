import matplotlib.pyplot as plt
from lib.naive_bayes import NaiveBayesGaussian, NaiveBayesMultinomial
import lib.tools as tools
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
import numpy as np
from itertools import permutations
from sklearn.naive_bayes import GaussianNB, MultinomialNB

def accuracy_of_multinomial(dataset, labels, num_of_bins=100):
    folds = 3
    data_split, labels_split = tools.cross_validation_split(dataset=dataset, labels=labels, folds=folds)
    nb = NaiveBayesMultinomial(num_of_bins=num_of_bins)
    clf = MultinomialNB(alpha=0.0001, fit_prior=True)

    accuracies, sklearn_accuracies = tools.accuracy_of_method(data_split, labels_split, nb, sklearn_class=clf)
    return np.mean(accuracies), np.mean(sklearn_accuracies)