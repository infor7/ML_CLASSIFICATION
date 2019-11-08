import numpy as np
import re


def load_text_file(file, first_column_labels=False, last_column_labels=False, dtype=int, labels_numeric=True):
    lines = file.readlines()
    num_lines = sum(1 for line in lines)
    no_of_features = len(re.split('[, \t]', lines[1]))
    # first_columb_labels = int(first_columb_labels)
    if first_column_labels or last_column_labels:
        no_of_features -= 1
    if last_column_labels:
        labels_ind = -1
    else:
        labels_ind = 0
    data = np.zeros((num_lines - 1, no_of_features), dtype=dtype)
    labels = np.zeros((num_lines - 1), dtype=int)
    for index, line in enumerate(lines[1:]):
        words = np.array(re.split('[, \t]', line))
        words[words=='?'] = 0
        if len(words) > 1:
            if first_column_labels:
                data[index] = np.array(words[1:])
            elif last_column_labels:
                data[index] = np.array(words[:-1])
            else:
                data[index] = np.array(words)

            if not labels_numeric:
                labels[index] = ord(words[labels_ind])-65
            else:
                labels[index] = words[labels_ind]

    return data, labels


def cross_validation_split(dataset, labels, folds=3):
    dataset_copy = np.array(dataset)
    indices = np.random.permutation(np.arange(len(dataset)))
    dataset_copy = dataset_copy[indices]
    labels_copy = labels[indices]
    dataset_split = np.array_split(dataset_copy, folds)
    labels_split = np.array_split(labels_copy, folds)
    return dataset_split, labels_split


def accuracy_of_method(data_split, labels_split, model_class_instance, sklearn_class=None):
    """
    Function for training and testing algorithm on cross split database,
    training_method must have three arguments:
        training_data, training_labels, test_data
    sklearn_class - optional sklearn object, which have two methods: fit and predict
    for now works only with three folds

    :param np.array data_split:
    :param np.array labels_split:
    :param callable model_class_instance:
    :param optional object sklearn_class:
    :return: accuracies, (optional) sklearn_accuracies, for all folds
    """
    folds = 3
    accuracies = []
    sklearn_accuracies = []
    for i in range(len(data_split)):
        rest = [x for x in range(len(data_split)) if x != i]
        # works only for folds = 3
        # training_data = np.array(data_split)[rest].reshape(-1, np.array(data_split)[rest].shape[-1])
        training_data = np.append(data_split[rest[0]], data_split[rest[1]], axis=0)
        # training_labels = np.array(labels_split)[rest].flatten()
        training_labels = np.append(labels_split[rest[0]], labels_split[rest[1]], axis=0)
        testing_data = data_split[i]
        testing_labels = labels_split[i]
        model_class_instance.train(training_data, training_labels)
        predictions = model_class_instance.predict(testing_data)
        # predictions = model_class_instance.naive_bayes_gaussian(training_data, training_labels, testing_data)
        # predictions = training_method(training_data, training_labels, testing_data)
        accuracy = sum(predictions==testing_labels)/len(testing_data)
        accuracies.append(accuracy)
        print("my: ", accuracy)
        if sklearn_class is not None:
            sklearn_class.fit(abs(training_data), training_labels)
            accuracy = sum(sklearn_class.predict(abs(testing_data))==testing_labels)/len(testing_data)
            print("sklearn: ", accuracy)
            sklearn_accuracies.append(accuracy)
    return accuracies, sklearn_accuracies


# print(load_text_file('../../datasets/letter-recognition.data', first_columb_labels=True))
