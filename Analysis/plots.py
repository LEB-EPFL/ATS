import matplotlib.pyplot as plt
from toolbox.plotting import violin
from mpl_toolkits.axisartist.grid_finder import (FixedLocator, MaxNLocator,
                                                DictFormatter)
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
import numpy as np
import mpl_toolkits.axisartist.angle_helper as angle_helper
from matplotlib.projections import PolarAxes


def mini_panel(axs, time_unit='min', vert=True):
        ### Cum Info Plot
    for  ax in axs:
        ax.set_xlim([0, ax.get_xlim()[1]])
        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.minorticks_on()
        ax.yaxis.set_tick_params(which='minor', left=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.axes.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        ax.set_xlabel("Time [" + time_unit + "]")


def decay_plot(data, struct_type: str, fig_size, style, output_folder, colors, vert=True):
    """
    INPUT: struct_type: 'mito' / 'caulo'style
    """
    if style == "publication":
        fig, ax = plt.subplots(figsize=fig_size, tight_layout=True)
    else:
        fig, ax = plt.subplots()
    labels = ['slow', 'fast', 'EDA']
    box_dict = plt.boxplot(data, labels=labels, showfliers=False, whis=[5, 95], widths=0.5,
                           vert=vert)
    print(data)
    violin.violin_overlay(data, spread=0, vert=vert)
    # for index, item in enumerate(box_dict['boxes']):
    #     plt.setp(item, color=colors[index])
    # so it fits the size of the cumsum plot
    ax.set_xlabel(" ") if vert else ax.set_ylabel(None)
    ax.set_ylabel("Bleaching Decay [AU]") if vert else ax.set_xlabel("Bleaching Decay [AU]")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    PLOT_MODE = 'vert' if vert else 'hor'
    plt.savefig(output_folder + struct_type + "_decay" + PLOT_MODE + ".pdf", transparent=True)


def rotated_figure(rect=111):

    fig = plt.figure(figsize=(8, 4))
    tr = Affine2D().scale(2, 1).rotate_deg(90)

    grid_helper = floating_axes.GridHelperCurveLinear(
        tr, extremes=(80, 400, 0, 300),
        grid_locator1=MaxNLocator(nbins=4),
        grid_locator2=MaxNLocator(nbins=4))

    ax1 = fig.add_subplot(
        rect, axes_class=floating_axes.FloatingAxes, grid_helper=grid_helper)

    ax = ax1.get_aux_axes(tr)
    return ax

def main():
    ax = rotated_figure()
    ax.bar([100, 200], [100, 200])
    plt.show()


if __name__ == '__main__':
    main()