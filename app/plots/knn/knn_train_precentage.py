import matplotlib.pyplot as plt
from plots.knn.knn import get_accuracy_for_given_k_and_target_precentage


def execute(ax=None, **kwargs):
    accuracy, indexes = get_accuracy_for_target_precentage()

    ax = ax or plt.gca()
    line, = ax.plot(indexes, accuracy, 'k', indexes, accuracy, 'bo')
    return line

def get_accuracy_for_target_precentage():
    accuracy = []
    x, y = 1, 99
    for index in range(x,y):
        accuracy.append(get_accuracy_for_given_k_and_target_precentage(5, index/100))
    return accuracy, range(x, y)



if __name__ == "__main__":
    fig, [plot1, plot2] = plt.subplots(nrows=2)
    execute(plot1)
    execute(plot2)
    plt.show()

