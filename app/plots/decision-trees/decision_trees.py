# Work in progress
# Module for plotting results based on the dataset provided
import matplotlib.pyplot as plt
import numpy as np


# Mandatory execute function. I'm just writing those comments to guide my eyes 
def execute(ax=None, **kwargs):
    ax = ax or plt.gca()
    line, = ax.plot (np.arange(0.0, 5.0, 0.02))
    return line

def plot():
    #TODO: plotting results based on default and chosen datasets
    pass


if __name__ == "__main__":
    # Dunno why yet. Just to be able to execute as main
    fig, [plot1, plot2] = plt.subplots(nrows=2)
    execute(plot1)
    execute(plot2)
    plt.show()