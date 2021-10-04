from tkinter.constants import VERTICAL
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
from cycler import cycler

import data_locations
from tools import get_info, get_times, get_snr, make_fps, get_files, get_decay
from constriction_v1 import get_width
import plots
from toolbox.plotting import violin, annotate
from toolbox.plotting.style_mpl import set_mpl_style, set_mpl_font


style = "publication"
set_mpl_style(style)
set_mpl_font(9)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # type: ignore
colors = [colors[0], colors[3], colors[2], colors[5], colors[4]]
mpl.rcParams.update({"axes.prop_cycle": cycler('color', colors)})
cm = 1/2.54  # centimeters in inches
fig_size = (4*cm, 5*cm)


output_folder = "//lebsrv2.epfl.ch/LEB_Perso/Willi-Stepp/ATS_Figures/Figure3_caulo/"
VERTICAL_MODE = True
PLOT_MODE = 'vert' if VERTICAL_MODE else 'hor'
fig_size = (4*cm, 5*cm) if VERTICAL_MODE else (6*cm, 4*cm)


slow_folders = data_locations.caulo_folders['slow']
fast_folders = data_locations.caulo_folders['fast']
ats_folders = data_locations.caulo_folders['ats']


def main():
    if style == "publication" and VERTICAL_MODE:
        fig1, ax1 = plt.subplots(figsize=fig_size)
        fig2, ax2 = plt.subplots(figsize=fig_size)
    elif style == "publication":
        fig1, ax1 = plt.subplots(figsize=(6/2.54, 6/2.54))
        fig2, ax2 = plt.subplots(figsize=(6/2.54, 6/2.54))
    else:
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
    axs = [ax1, ax2]
    figs = [fig1, fig2]

    # Plotting of the cum_info versus time will be done inside the functions to limit outputs
    # Get the width from inside the ats function to enable filter for high/low fps
    fast_info, fast_decay = analyse_fast_data(axs)
    slow_info, slow_decay = analyse_slow_data(axs)

    high_info, low_info, ats_width, ats_decay = analyse_ats_data(axs)
    ats_info = high_info + low_info

    # ### MEAN INFO PLOT
    print('high info: {:.2f}'.format(np.mean(high_info)))
    print('low  info: {:.2f}'.format(np.mean(low_info)))
    labels = ["slow", "fast", 'EDA', 'EDA-fast', 'EDA-slow']
    data = [slow_info, fast_info, ats_info, high_info, low_info]  #, slow_info]
    if style == "publication":
        cm = 1/2.54  # centimeters in inches
        _, ax = plt.subplots(figsize=(9*cm, 5*cm), tight_layout=True)
    else:
        _, ax = plt.subplots()

    plt.boxplot(data, labels=labels, showfliers=False, whis=[5, 95])
    violin.violin_overlay(data)
    annotate.significance_brackets([(0,2), (3, 4)])
    plt.ylabel('Information per Frame [AU]')
    # plt.title("Caulobacter EDA")
    _, p_value = stats.ttest_ind(slow_info, ats_info)
    print(p_value)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_folder + "caulo_violin.pdf", facecolor=None, edgecolor=None,
                transparent=True)

    print("Slow: N = {}, n = {}".format(len(slow_info), len(slow_folders)))
    print("Fast: N = {}, n = {}".format(len(fast_info), len(fast_folders)))
    print(" EDA: N = {}, n = {}".format(len(ats_info), len(ats_folders)))


    ### CONSTRICTION plot
    # fast_width, *_ = constriction(fast_folders)
    # fast_width = list(filter(None, fast_width))
    # slow_width, *_ = constriction(slow_folders)
    # slow_width = list(filter(None, slow_width))

    # _, ax = plt.subplots()
    # data = [slow_width, fast_width, ats_width['all'], ats_width['high'], ats_width['low']]
    # data = [np.multiply(sublist,pixel_calib) for sublist in data]
    # plt.boxplot(data, labels=labels, showfliers=False, whis=[5,95])
    # violin.violin_overlay(data, bins=50)
    # annotate.significance_brackets([(0,2), (0,3), (1,2), (1,3), (3,4)])
    # plt.ylabel('Constriction FWHM [nm]')


    # ### DECAY plot
    slow_decay = [i for i in slow_decay if i < 10]
    data = [slow_decay, fast_decay, ats_decay]
    plots.decay_plot(data, '/caulo', fig_size, style, output_folder, colors,
                     vert=VERTICAL_MODE)
    print('bleaching decay fast/EDA: ', np.mean(fast_decay)/np.mean(ats_decay))
    print('bleaching decay EDA/slow: ', np.mean(ats_decay)/np.mean(slow_decay))

    # ### CUMSUM and CUMINFO plots
    plots.mini_panel(axs, 'h', vert=VERTICAL_MODE)

    ax1.set_ylabel("Cumulative Lightdose [AU]")
    plt.figure(figs[0].number)
    plt.tight_layout()
    ax1.set_xticks([0, 3])
    ax1.set_xlim([0, 3])
    ax1.xaxis.labelpad = - 7
    plt.savefig(output_folder + "/caulo_lightdose_"+PLOT_MODE+".pdf", transparent=True)

    ax2.set_ylabel("Cumulative Information [AU]")
    plt.figure(figs[1].number)
    plt.tight_layout()
    plt.savefig(output_folder + "/caulo_info_cum_"+PLOT_MODE+".pdf", transparent=True)

    plt.show()


def analyse_slow_data(axs):
    print('Slow data')
    folders = slow_folders
    info_all = []
    decay_all = []
    for folder in folders:
        print(folder)
        files, _ = get_files(folder)
        times = get_times(files['peaks'])
        snr, _ = get_snr(files['network'],  rect=[50, 974, 50, 974])
        # fig, ax1 = plt.subplots()
        snr_decay = get_snr_fast(files['network'], rect=[50, 974, 50, 974])
        decay_all.append(get_decay(times, snr_decay))

        info = get_info(files['nn'], times)
        info_all.extend(info)
        snr_cutoff_frame = np.sum([x > 1.02 for x in snr])

        times = np.divide(times, 1000*60*60)
        axs[1].plot(times[1:snr_cutoff_frame+1],
                    np.cumsum(info[:len(times[1:snr_cutoff_frame+1])]), color=colors[0], alpha=0.8)
        times = times[1:snr_cutoff_frame+1]
        light_exp_plot(axs[0], times, 0, 1)

    return info_all, decay_all


def analyse_fast_data(axs):
    print('Fast data')
    folders = fast_folders
    info_all = []
    decay_all = []
    for folder in folders:
        print(folder)
        files, _ = get_files(folder)
        snr = get_snr_fast(files['network'], rect=[50, 974, 50, 974])
        times = get_times(files['peaks'])

        # Calculate the decay constant from the SNR
        decay_all.append(get_decay(times[1:], snr[1:]))

        # fig, ax1 = plt.subplots()
        # ax1.plot(times[1:], snr[1:], color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])#, color = COLORS['slow'])
        # plt.show()

        info = get_info(files['nn'], times)
        info_all.extend(info)
        snr_cutoff_frame = np.sum([x > 1.02 for x in snr])

        times = np.divide(times, 1000*60*60)
        print('SNR cutoff fast:', times[snr_cutoff_frame-1])
        axs[1].plot(times[1:snr_cutoff_frame+1],
                    np.cumsum(info[:len(times[1:snr_cutoff_frame+1])]), color=colors[1], alpha=0.8)
        times = times[1:snr_cutoff_frame+1]
        light_exp_plot(axs[0], times, 1, 1)
    return info_all, decay_all


def analyse_ats_data(axs):
    print('EDA data')
    # folders = ['W:/Watchdog/microM_test/201208_cell_Int0s_30pc_488_50pc_561_band_6']
    # folder = 'C:/Users/stepp/Documents/02_Raw/Caulobacter_iSIM/210421_23'
    folders = ats_folders
    high_info_all = []
    low_info_all = []
    high_width_all = []
    low_width_all = []
    decay_all = []
    for folder in folders:
        print(folder)
        files, _ = get_files(folder)

        # Calculate the SNR for every bacteria image
        snr, _ = get_snr(files['network'])

        # This is the SNR plot, let's see which parts are high/low fps
        # get the timting of all the drp images
        times = get_times(files['peaks'])

        # The first frame was 5 seconds for a long time
        fps, fps_times = make_fps(times[1:], unit='h')

        snr_decay = get_snr_fast(files['network'], rect=[50, 974, 50, 974])
        decay_all.append(get_decay(times, snr_decay))

        # #Let's plot this data
        # fig, ax1 = plt.subplots()
        # ax1.plot(fps_times, fps)#, color = COLORS['ats'])
        # ax2 = ax1.twinx()
        # ax1.plot(times[1:], snr[1:], color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])#, color = COLORS['slow'])
        # plt.show()

        #Now, what is the mean 'info' in a frame that was taken at low fps vs high fps
        info, high_info, low_info = get_info(files['nn'], times, fps)
        # width, high_width, low_width = get_width_ats(files['network'], files['nn'], times, fps)

        # high_width = list(filter(None, high_width))
        # low_width = list(filter(None, low_width))
        # width = list(filter(None, width))

        # #Let's plot this data
        # fig, ax1 = plt.subplots()
        # ax1.plot(fps_times, fps)#, color = COLORS['ats'])
        # ax2 = ax1.twinx()
        # ax2.plot(times[1:], info, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])#, color = COLORS['slow'])
        # plt.show()

        high_info_all.extend(high_info[:])
        low_info_all.extend(low_info[:])

        # high_width_all.extend(high_width[:])
        # low_width_all.extend(low_width[:])

        # print('high info: {:.2f}'.format(np.mean(high_info)))
        # print('low  info: {:.2f}'.format(np.mean(low_info)))
        # labels = ['high fps', 'low fps']
        # plt.boxplot([high_info, low_info], labels = labels)
        # high_x = np.random.normal(1, 0.02, len(high_info))
        # low_x = np.random.normal(2, 0.02, len(low_info))
        # plt.scatter(high_x, high_info, alpha=0.5, edgecolors='none')
        # plt.scatter(low_x, low_info, alpha=0.5, edgecolors='none')

        # _, p_value = stats.ttest_ind(high_info, low_info)
        # print(p_value)
        snr_cutoff_frame = np.sum([x > 1.02 for x in snr])

        times = np.divide(times, 1000*60*60)
        print('SNR cutoff ats:', times[snr_cutoff_frame-1])
        axs[1].plot(times[1:snr_cutoff_frame+1],
                    np.cumsum(info[:len(times[1:snr_cutoff_frame+1])]), color=colors[2])
        times = times[1:snr_cutoff_frame+1]
        light_exp_plot(axs[0], times, 2)

    all_width = {'all': low_width_all + high_width_all,
                 'high': high_width_all,
                 'low':  low_width_all}

    return high_info_all, low_info_all, all_width, decay_all


def light_exp_plot(ax, times, color, zorder=0):
    if len(times) > 3:
        times = times - np.min(times)
        ax.plot(times, np.cumsum(np.ones_like(times)), color=colors[color], zorder=zorder)


def get_width_ats(bact_filelist, nn_filelist, times, fps):
    high_width = []
    low_width = []
    width_list = []
    high = np.max(fps)
    low = np.min(fps)

    for index, (bact_file, nn_file) in enumerate(zip(bact_filelist, nn_filelist)):
        width, _ = get_width(bact_file, nn_file)
        width_list.append(width)
        if fps is not None:
            factor = 1/low
            if round(1/(times[index]- times[index-1])*factor) == round(high*factor):
                high_width.append(width)
            elif round(1/(times[index]- times[index-1])*factor) == round(low*factor):
                low_width.append(width)
            else:
                print('fps does not match high or low level')
    return width_list, high_width, low_width


def get_snr_fast(filelist, rect=None):
    """ Get the mask just from the first image and reuse it for all frames. As the imaging is fast,
    the position of the bacteria should not change much."""
    if os.path.isfile(filelist[0]):
        image = cv2.imread(filelist[0],-1)
    else:
        image = filelist[0]

    if rect is not None:
        image = image[rect[0]:rect[1], rect[2]:rect[3]]
    blur = cv2.GaussianBlur(image, (5,5), 0)
    blur = cv2.medianBlur(blur, 5)
    ret3, mask = cv2.threshold(blur,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    invert_mask = cv2.bitwise_not(mask).astype(np.bool)
    mask = mask.astype(np.bool)
    snr = []
    for file in filelist:
        if os.path.isfile(file):
            image = cv2.imread(file,-1)
        else:
            image = file

        if rect is not None:
            image = image[rect[0]:rect[1], rect[2]:rect[3]]

        mean_signal = np.mean(image[mask])
        mean_noise = np.mean(image[invert_mask])
        snr.append(mean_signal/mean_noise)
    return snr


if __name__ == "__main__":
    main()