import numpy as np
import random


<<<<<<< HEAD
class NaiveBayes(object):
=======
class NaiveBayesGaussian(object):
    def __init__(self, **kwargs):
        self.summary = []

    def fit(self, train_data, train_labels):
        self.summary = []
        self.summary = self.gaussian_calculate_summaries(train_data, train_labels)

    def predict(self, test_data):
        return self.gaussian_predict(self.summary, test_data)

>>>>>>> 9a3fc003c59d75666a69476982e0f8d4d69d531f
    def gaussian_calculate_summaries(self, dataset, labels):
        """
        Creates summary of dataset, first index - type of summary, second - no of class

        :param dataset:
        :return:
        """
        dataset_classed = self.split_by_class(dataset, labels)
        mean = np.array([np.mean(x, axis=0) for x in dataset_classed])
        std = np.array([np.std(x, axis=0) for x in dataset_classed])
        std[std==0] = 1
        summary = [mean, std, [len(x) for x in dataset_classed]]
        return summary

    def gaussian_calculate_label_probabilities(self, rows, summary):
        # if mean is None or std is None:
        feature_probabilities = [self.gaussian_calculate_probability(rows, summary[0][x], summary[1][x]) for x in
                                 range(len(summary[0]))]
        probabilities = np.prod(feature_probabilities, axis=2) / summary[2][0]
        return probabilities

    # Calculate the Gaussian probability distribution function for x
    def gaussian_calculate_probability(self, x, mean, stdev):
        exponent = np.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        return (1 / (np.sqrt(2 * np.pi) * stdev)) * exponent

    # gaussian_predict the class for a given row
    def gaussian_predict(self, summaries, rows):
        probabilities = self.gaussian_calculate_label_probabilities(rows, summaries)
        best_labels = np.argmax(probabilities, axis=0)
        return best_labels

    # Naive Bayes Algorithm
    def naive_bayes_gaussian(self, train_data, train_labels, test_data):
        summary = self.gaussian_calculate_summaries(train_data, train_labels)
        return self.gaussian_predict(summary, test_data)

    def split_by_class(self, dataset, labels):
        sorted_indices = np.argsort(labels)
        dataset = dataset[sorted_indices]
        labels = labels[sorted_indices]
        dataset_classed = np.split(dataset, np.where(labels[:-1] != labels[1:])[0] + 1)
        return dataset_classed


class NaiveBayesMultinomial(object):
    def __init__(self, **kwargs):
        self.hist_bins = list()
        self.summary_of_prob = list()
        if kwargs.get('alpha'):
            self.alpha = kwargs['alpha']
        else:
            self.alpha = 0.0001
        if kwargs.get('num_of_bins'):
            self.num_of_bins = kwargs['num_of_bins']
        else:
            self.num_of_bins = 100

    def fit(self, train_data, train_labels):
        dataset_classed = self.split_by_class(train_data, train_labels)
        hist_bins_min = [np.min(train_data[:, num_feature]) for num_feature in range(len(train_data[0]))]
        hist_bins_max = [np.max(train_data[:, num_feature]) for num_feature in range(len(train_data[0]))]
        self.num_of_bins = len(train_data)//len(dataset_classed)
        self.hist_bins = [np.linspace(np.floor(hist_bins_min[i]), np.ceil(hist_bins_max[i]) + 2, self.num_of_bins) for i in
                     range(len(train_data[0]))]
        self.summary_of_prob = []
        for class_f in dataset_classed:
            # summary_for_each_class = np.zeros((len(class_f[0]), len(hist_bins)-1))
            summary_for_each_class = []
            prob_of_class = np.log(len(class_f) / len(train_data))
            for feature_num in range(len(class_f[0])):
                feature = class_f[:, feature_num]
                hist, _ = np.histogram(feature, bins=self.num_of_bins,
                                       range=(hist_bins_min[feature_num], hist_bins_max[feature_num] + 2), density=True)
                # prob of each feature value in class to all values
                # summary_for_each_class[feature_num] =np.log(hist+alpha)
                summary_for_each_class.append(np.log(hist + self.alpha) + prob_of_class)
            # summary_for_each_class += np.log(len(class_f)/len(train_data))
            self.summary_of_prob.append(summary_for_each_class)

    def predict(self, test_data):
        sums_of_probs = np.zeros((len(test_data), len(self.summary_of_prob )))
        for class_f in range(len(self.summary_of_prob )):
            current_summary = self.summary_of_prob[class_f]
            sum_of_prob = np.zeros(len(test_data))
            for i in range(len(test_data[0])):
                # sum_of_prob+=current_summary[i,np.array(test_data[:,i], dtype=int)]
                indices = np.searchsorted(self.hist_bins[i], test_data[:, i]) - 1
                sum_of_prob += current_summary[i][indices]
            sums_of_probs[:, class_f] = sum_of_prob

        return np.argmax(sums_of_probs, axis=1)

    def split_by_class(self, dataset, labels):
        sorted_indices = np.argsort(labels)
        dataset = dataset[sorted_indices]
        labels = labels[sorted_indices]
        dataset_classed = np.split(dataset, np.where(labels[:-1] != labels[1:])[0] + 1)
        return dataset_classed
