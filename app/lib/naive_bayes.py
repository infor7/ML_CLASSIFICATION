import numpy as np
import random


class NaiveBayes(object):
    def calculate_summaries(self, dataset, labels):
        """
        Creates summary of dataset, first index - type of summary, second - no of class

        :param dataset:
        :return:
        """
        print(dataset, labels)
        sorted_indices = np.argsort(labels)
        dataset = dataset[sorted_indices]
        labels = labels[sorted_indices]
        dataset_classed = np.split(dataset, np.where(labels[:-1] != labels[1:])[0] + 1)
        mean = np.array([np.mean(x, axis=0) for x in dataset_classed])
        std = np.array([np.std(x, axis=0) for x in dataset_classed])
        summary = [mean, std, [len(x) for x in dataset_classed]]
        return summary

    def calculate_label_probabilities(self, rows, summary=None, mean=None, std=None):
        if mean is None or std is None:
            feature_probabilities = [self.calculate_probability(rows, summary[0][x], summary[1][x]) for x in
                                     range(len(summary[0]))]
        else:
            feature_probabilities = [self.calculate_probability(rows, mean, std) for x in
                                     range(len(summary[0]))]
        probabilities = np.prod(feature_probabilities, axis=2) / summary[2][0]
        return probabilities

    # Calculate the Gaussian probability distribution function for x
    def calculate_probability(self, x, mean, stdev):
        exponent = np.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        return (1 / (np.sqrt(2 * np.pi) * stdev)) * exponent

    # Predict the class for a given row
    def predict(self, summaries, rows):
        probabilities = self.calculate_label_probabilities(rows, summaries)
        best_labels = np.argmax(probabilities, axis=0)
        return best_labels

    # Naive Bayes Algorithm
    def naive_bayes(self, train_data, train_labels, test_data):
        summary = self.calculate_summaries(train_data, train_labels)
        return self.predict(summary, test_data)
        # predictions = list()
        # for row in test:
        #     output = self.predict(summarize, row)
        #     predictions.append(output)
        # return (predictions)
