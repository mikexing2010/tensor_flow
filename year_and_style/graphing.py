import matplotlib.pyplot as plt
import numpy as np


def plot_data(data, y_name, x_name, file_name):
    # data is a list of numbers
    plt.plot(data)
    plt.ylabel(y_name)
    plt.xlabel(x_name)
    plt.grid()
    plt.title(file_name)
    plt.savefig(file_name, bbox_inches='tight')


data = range(10, 0, -1)
plot_data(data, "loss", "step", "loss over time.png")