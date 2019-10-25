import numpy as np
import re


def load_text_file(file, first_columb_labels=False, dtype=int, labels_numeric=False):
    lines = file.readlines()
    num_lines = sum(1 for line in lines)
    no_of_features = len(re.split('[, \t]', lines[0]))
    # first_columb_labels = int(first_columb_labels)
    no_of_features -= first_columb_labels
    data = np.zeros((num_lines - 1, no_of_features), dtype=dtype)
    labels = np.zeros((num_lines - 1), dtype=int)
    for index, line in enumerate(lines[1:]):
        words = re.split('[, \t]', line)
        if len(words) > 1:
            data[index] = words[first_columb_labels:]
            if not labels_numeric:
                labels[index] = ord(words[0])-65
            else:
                labels[index] = words[0]

    return data, labels

# print(load_text_file('../../datasets/letter-recognition.data', first_columb_labels=True))
