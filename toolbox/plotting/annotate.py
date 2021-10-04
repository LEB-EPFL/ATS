import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatch
import numpy as np
from scipy import stats


def significance_brackets(pairs:list, axes:plt.Axes = None, vert=True):
    if vert:
        significance_brackets_vert(pairs, axes)
    else:
        significance_brackets_hor(pairs, axes)

def significance_brackets_vert(pairs:list, axes:plt.Axes = None):
    """ Add significance bracket to a plot over the specified pairs. Initially only programmed
    for the boxplot + violin plot also present in this module. Could/Should be extended to at least
    only the boxplot."""

    if axes is None:
        axes = plt.gca()

    # Check if we are in the tight layout
    if axes.get_figure().get_tight_layout() is True:
        offset = 3
        y_lim_correction = 1.4
    else:
        offset = 7
        y_lim_correction = 2

    axes_objects = axes.get_children()

    # Find the scatter plots
    scatters = []
    for axes_object in axes_objects:
        if isinstance(axes_object, matplotlib.collections.PathCollection):
            scatters.append(axes_object)

    offset_list = np.ones(len(scatters))


    for (pos1, pos2) in pairs:

        #Look at all the y_data that the bracket will span and get the maximal y position

        y_data = [scatters[index].properties()["offsets"][:, 1] for index in range(pos1, pos2+1)]
        y_data = [item for sublist in y_data for item in sublist]
        y_pos = np.max(y_data)


        y_data1 = scatters[pos1].properties()["offsets"][:, 1]
        y_data2 = scatters[pos2].properties()["offsets"][:, 1]
        _, p_value = stats.ttest_ind(y_data1, y_data2)
        print("{} {}".format(pos1, pos2))
        print("p_value: {}".format(p_value))
        number_of_stars = min(int(np.floor(abs(np.log10(p_value)+1))), 5)
        if number_of_stars == 0:
            text = 'n.s.'
        else:
            text = '*' * number_of_stars

        props = {'arrowstyle':'-', 'shrinkA':0, 'shrinkB':0, 'linewidth':1}
        spacer_props = {'shrinkA':0, 'shrinkB':0, 'linewidth':1, 'headwidth':28, 'alpha':0}
        dx = np.abs(pos2-pos1)

        multiple_bracket_factor = (max(list(offset_list[(pos1+1):pos2]) + [1]) - 1)*offset*2
        shift_up = matplotlib.transforms.offset_copy(axes.transData, fig=axes.get_figure(),
                                                     y=offset*2 + multiple_bracket_factor,
                                                     units='points')

        bar = axes.annotate('', xy=(pos1+1, y_pos), xytext=(pos2+1, y_pos),
                            arrowprops=props, xycoords=('data', shift_up))
        axes.annotate('', xy=(0,0), xytext=(0, -offset), arrowprops=props,  xycoords=bar,
                      textcoords = 'offset points')
        axes.annotate('', xy=(1,0), xytext=(0, -offset), arrowprops=props,  xycoords=bar,
                      textcoords = 'offset points')

        # bar_box = bar.get_tightbbox(axes.get_figure().canvas.get_renderer())
        valign = 'bottom' if text == "n.s." else 'baseline'
        text = axes.annotate(text, xy=(0.5, 1), zorder=10, ha='center', va=valign,
                             xycoords=bar, fontsize=9)

        # Check if we have to adjust the ylim of the axes
        text_box = text.get_tightbbox(axes.get_figure().canvas.get_renderer())
        text_y = axes.transAxes.inverted().transform((0,text_box.y1))[1]
        if text_y > 1:
            top_y = (text_y - 1) * y_lim_correction + 1
            new_y_max = axes.transLimits.inverted().transform([0, top_y])[1]
            axes.set_ylim(axes.get_ylim()[0], new_y_max)

        offset_list[pos1] += 1
        offset_list[pos2] += 1

def significance_brackets_hor(pairs:list, axes:plt.Axes = None):
    """ Add significance bracket to a plot over the specified pairs. Initially only programmed
    for the boxplot + violin plot also present in this module. Could/Should be extended to at least
    only the boxplot."""

    if axes is None:
        axes = plt.gca()

    # Check if we are in the tight layout
    if axes.get_figure().get_tight_layout() is True:
        offset = 3
        x_lim_correction = 1.4
    else:
        offset = 7
        x_lim_correction = 2

    axes_objects = axes.get_children()

    # Find the scatter plots
    scatters = []
    for axes_object in axes_objects:
        if isinstance(axes_object, matplotlib.collections.PathCollection):
            scatters.append(axes_object)

    offset_list = np.ones(len(scatters))


    for (pos1, pos2) in pairs:

        #Look at all the y_data that the bracket will span and get the maximal y position

        x_data = [scatters[index].properties()["offsets"][:, 0] for index in range(pos1, pos2+1)]
        x_data = [item for sublist in x_data for item in sublist]
        x_pos = np.max(x_data)


        x_data1 = scatters[pos1].properties()["offsets"][:, 0]
        x_data2 = scatters[pos2].properties()["offsets"][:, 0]
        _, p_value = stats.ttest_ind(x_data1, x_data2)
        print("{} {}".format(pos1, pos2))
        print("p_value: {}".format(p_value))
        number_of_stars = min(int(np.floor(abs(np.log10(p_value)+1))), 5)
        if number_of_stars == 0:
            text = 'n.s.'
        else:
            text = '*' * number_of_stars

        props = {'arrowstyle':'-', 'shrinkA':0, 'shrinkB':0, 'linewidth':1}
        spacer_props = {'shrinkA':0, 'shrinkB':0, 'linewidth':1, 'headwidth':28, 'alpha':0}
        dx = np.abs(pos2-pos1)

        multiple_bracket_factor = (max(list(offset_list[(pos1+1):pos2]) + [1]) - 1)*offset*2
        shift_over = matplotlib.transforms.offset_copy(axes.transData, fig=axes.get_figure(),
                                                     x=offset*2 + multiple_bracket_factor,
                                                     units='points')

        print(x_pos)
        print(pos1)
        print(pos2)
        bar = axes.annotate('', xy=(x_pos, pos1+1), xytext=(x_pos, pos2+1),
                            arrowprops=props, xycoords=(shift_over, 'data'))
        axes.annotate('', xy=(0,0), xytext=(-offset, 0), arrowprops=props,  xycoords=bar,
                      textcoords = 'offset points')
        axes.annotate('', xy=(0,1), xytext=(-offset, 0), arrowprops=props,  xycoords=bar,
                      textcoords = 'offset points')

        # bar_box = bar.get_tightbbox(axes.get_figure().canvas.get_renderer())
        valign = 'bottom' if text == "n.s." else 'center'
        text = axes.annotate(text, xy=(1, 0.5), zorder=10, ha='left', va=valign,
                             xycoords=bar, fontsize=9, rotation=270)

        # # Check if we have to adjust the ylim of the axes
        # text_box = text.get_tightbbox(axes.get_figure().canvas.get_renderer())
        # text_y = axes.transAxes.inverted().transform((0,text_box.y1))[1]
        # if text_y > 1:
        #     top_y = (text_y - 1) * x_lim_correction + 1
        #     new_y_max = axes.transLimits.inverted().transform([0, top_y])[1]
        #     axes.set_ylim(axes.get_ylim()[0], new_y_max)

        offset_list[pos1] += 1
        offset_list[pos2] += 1



def main():
    import numpy as np
    from boxplot import boxplot
    from violin import violin_overlay

    cm = 1/2.54
    # plt.subplots(figsize=(9*cm, 5*cm), tight_layout=True)
    plt.subplots()
    data1 = np.random.normal(1, 1, 110)
    data2 = np.random.normal(1, 1, 100)
    data3 = np.random.normal(1.5, 1, 100)
    boxplot([data1,data2, data3])
    violin_overlay([data1, data2, data3])
    significance_brackets([(0,1), (1,2), (0,2)])
    plt.show()

def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    """
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)

if __name__ == "__main__":
    main()