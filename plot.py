import os.path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, integrate
from matplotlib.pyplot import MultipleLocator
from utils import *

"""
configure
"""
path = [
    [
        'alexnet_cifar10_v0', '7'
    ]
]

base_dir = f'logs'
log_name = 'history.csv'

files = [os.path.join(base_dir, dir, id, log_name) for dir, id in path]

labels = ['SGD', 'LA', 'SWA']

save = False
outname=''

if __name__ == '__main__':

    plot_curve(files, labels, y='train_loss', y_label='Train loss', xlim=[0, 200], ylim=[0, 1.5], locator=30, save=save)
    # plot_curve(files, labels, y='valid_loss', y_label='Test loss', xlim=[0, 200], ylim=[0.75, 1.9], locator=30, save=save)
    # plot_curve(files, labels, y='valid_acc', y_label='Test error(%)', xlim=[0, 200], ylim=[0.19, 0.5], locator=30, save=save)

