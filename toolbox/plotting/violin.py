import matplotlib.pyplot as plt
import numpy as np


def violin_overlay(data_list: list, relative_list:list=None, spread:int=0.4):
    if relative_list is None:
        relative_list = [True]*len(data_list)

    for index, (data, relative) in enumerate(zip(data_list, relative_list)):
        alpha, x = get_violin_x(data, index+1, spread=spread, relativ_hist=relative)
        plt.scatter(x, data, alpha=alpha, edgecolors='none')


def get_violin_x(data, offset, spread=0.4, relativ_hist=True):
    x = []

    hist_data, edges = np.histogram(data, bins=20)
    hist_data_abs = hist_data
    if relativ_hist:
        hist_data = np.divide(hist_data,np.max(hist_data))
    else:
        spread = 1/300

    for value in data:
        bin_num = len(edges[edges < value])-1
        x.append((np.random.random(1)-0.5)*spread*hist_data[bin_num]+offset)

    #set alpha from the space over which the data is spread out
    ylim = plt.gca().get_ylim()
    height = 1 - (edges[1] - edges[0])/(ylim[1] - ylim[0])
    num_points = np.max(hist_data_abs)
    alpha = 1/(np.max([1,num_points/10]))*height
    return alpha, x