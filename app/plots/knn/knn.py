import matplotlib.pyplot as plt
from lib.knn import KNN
from sklearn.datasets import load_iris

def execute(ax=None, **kwargs):
    accuracy, indexes = get_accuracy_for_range(1,50)

    ax = ax or plt.gca()
    line, = ax.plot(indexes, accuracy, 'k', indexes, accuracy, 'bo')
    plt.title("Accuracy vs K value")
    return line



def get_accuracy_for_range(start, stop):
    accuracy = []
    for index in range(start, stop):
        accuracy.append(get_accuracy_for_given_k_and_target_precentage(index, 0.33))
    return accuracy, range(start, stop)

def get_accuracy_for_given_k_and_target_precentage(k, percentage):
    iris = load_iris()
    knn = KNN()
    train_samples, train_labels, test_samples, test_labels = knn.split_training(iris.data, iris.target, percentage)
    knn.set_up_train_data(train_samples, train_labels)
    predictions = knn.predict(test_samples, 5)

    guessed_number = 0
    for index, prediction in enumerate(predictions):
        if prediction == test_labels[index]:
            guessed_number += 1
    return guessed_number/len(test_samples)


if __name__ == "__main__":
    fig, [plot1, plot2] = plt.subplots(nrows=2)
    execute(plot1)
    execute(plot2)
    plt.show()

