import math
import numpy as np
from collections import Counter


class KNN():

    def set_up_train_data(self, training_samples, training_labels):
        self.trainig_samples = training_samples
        self.training_labels = training_labels


    def split_training(self, samples, labels, test_size):
        if len(samples) != len(labels):
            raise Exception("You must give same number of samples and labels")

        if test_size >= 1 and test_size <= 0:
            raise Exception("Test size must be greater than 0 and less than 1")

        training_samples, training_labels = [], []
        testing_sample, testing_labels = [], []

        for index, sample in enumerate(samples):
            if index == 0: # removes istuation when training sample is 0
                training_samples.append(sample)
                training_labels.append(labels[index])
                continue

            if index == 1: # removes istuation when training sample is 0
                testing_sample.append(sample)
                testing_labels.append(labels[index])
                continue

            if np.random.rand() > test_size:
                training_samples.append(sample)
                training_labels.append(labels[index])
            else:
                testing_sample.append(sample)
                testing_labels.append(labels[index])

        return training_samples, training_labels, testing_sample, testing_labels

    def distance(self, point_a, point_b):
        if len(point_a) != len(point_b):
            raise Exception("Points do not have same dimensions")

        sub_list = self.subtract_lists(point_a, point_b)
        powered_list = [x ** 2 for x in sub_list]
        return math.sqrt(sum(powered_list))

    def subtract_lists(self, list_a, list_b):
        result_list = []
        if len(list_a) != len(list_b):
            raise Exception("Points do not have same dimensions")

        for index, item in enumerate(list_a):
            result_list.append(item - list_b[index])

        return result_list

    def predict(self, test_samples, number_of_nearest_neighbours):
        predictions = []
        for test_sample in test_samples:
            distance = [self.distance(training_sample, test_sample) for training_sample in self.trainig_samples]
            sorted_indexes = np.argsort(distance)[:number_of_nearest_neighbours] #sort and convert value to its pre sort list index
            labels = [self.training_labels[x] for x in sorted_indexes]

            predictions.append(Counter(labels).most_common(1)[0][0])

        return predictions

if __name__ == "__main__":
    pass

