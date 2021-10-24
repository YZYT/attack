import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, integrate
from matplotlib.pyplot import MultipleLocator


def plot(ax, path, x, y, label):
    data = pd.read_csv(path)

    data_x = data[x]
    data1_y = data[y]

    if y == 'test_acc':
        data1_y = 1 - data1_y

    ax.plot(data_x, data1_y, label=label)


def plot_curve(files, labels, y, y_label, xlim, ylim, locator=10, save=False, **kwargs):
    # Train loss

    x = 'epoch'

    fig, ax = plt.subplots()
    for path, label in zip(files, labels):
        plot(ax, path, x, y, label)

    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel(y_label, fontsize=16)

    plt.xlim(*xlim)
    plt.ylim(*ylim)

    x_major_locator = MultipleLocator(locator)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.legend()

    plt.subplots_adjust(left=0.12, right=0.88, top=0.9, bottom=0.11)

    if kwargs.get('outname'):
        outname = kwargs['outname']
    else:
        outname = ''

    outname += y + '.png'
    if save:
        plt.savefig(fname=outname)
    plt.show()