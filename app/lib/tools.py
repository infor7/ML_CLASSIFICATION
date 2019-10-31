import numpy as np
import re


def load_text_file(file, first_column_labels=False, last_column_labels=False, dtype=int, labels_numeric=False):
    lines = file.readlines()
    num_lines = sum(1 for line in lines)
    no_of_features = len(re.split('[, \t]', lines[0]))
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
        words = re.split('[, \t]', line)
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

# print(load_text_file('../../datasets/letter-recognition.data', first_columb_labels=True))
