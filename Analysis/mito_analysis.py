"""
Get the real mito Data into a structure that can be plotted nicely and to get to the
blog post graphs that are in Doras last Basecamp post:
https://3.basecamp.com/4536561/buckets/17861748/messages/3288166766
"""
from constriction import VERTICAL_MODE
import glob
import json
import os
import re
from scipy import stats, optimize
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import numpy as np

import tifffile
from SmartMicro import NNio
from toolbox.plotting.style_mpl import set_mpl_style, set_mpl_font
from toolbox.plotting import violin, annotate
from toolbox.misc.speak import say_done
from constriction_v1 import get_width
from constriction_v1 import constriction, pixel_calib

import data_locations
from tools import get_info, get_snr, get_times, make_fps

#  Make a custom cycle so that slow/fast and EDA data is grouped in colors
style = "publication"
set_mpl_style(style)
set_mpl_font(size=9)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = [colors[0], colors[3], colors[2], colors[5], colors[4]]
mpl.rcParams.update({"axes.prop_cycle": cycler('color', colors)})
cm = 1/2.54  # centimeters in inches


TIME_ADJUST = 1_000*60
INFO_ADJUST = 10_000

output_folder = "//lebsrv2.epfl.ch/LEB_Perso/Willi-Stepp/ATS_Figures/Figure2_combined/archive/"
VERTICAL_MODE = True
PLOT_MODE = 'vert' if VERTICAL_MODE else 'hor'
fig_size = (4*cm, 5*cm) if VERTICAL_MODE else (6*cm, 4*cm)

slow_folders = data_locations.mito_folders['slow']
fast_folders = data_locations.mito_folders['fast']
ats_folders = data_locations.mito_folders['ats']


def main():
    # constriction_analysis()
    # say_done()
    # plt.show()
    info_analysis()
    plt.show()


def constriction_analysis():
    # EDA
    json_file = os.path.dirname(ats_folders[1]) + "/analysis.json"
    if os.path.isfile(json_file):
        with open(json_file, 'r') as in_file:
            all_width = json.load(in_file)
    else:
        all_width, _ = constriction(ats_folders, ats=True)

    # all_width = constriction(ats_folders, ats=True)
    #FIXED
    json_file = os.path.dirname(slow_folders[1]) + "/analysis.json"
    if os.path.isfile(json_file):
        with open(json_file, 'r') as in_file:
            slow_width = json.load(in_file)
    else:
        slow_width = constriction(slow_folders)

    # slow_width = constriction(slow_folders)
    slow_width = list(filter(None, slow_width))

    json_file = os.path.dirname(fast_folders[1]) + "/analysis.json"
    if os.path.isfile(json_file):
        with open(json_file, 'r') as in_file:
            fast_width = json.load(in_file)
    else:
        fast_width = constriction(fast_folders)
    # fast_width = constriction(fast_folders)
    fast_width = list(filter(None, fast_width))
    print(json_file)

    data =[slow_width, fast_width] + list(list(filter(None, x[1])) for x in all_width.items())
    data = [np.multiply(sublist, pixel_calib) for sublist in data]
    labels =["slow", "fast"] +  list(x[0] for x in all_width.items())
    plt.boxplot(data, labels=labels, showfliers=False, whis=[5,95])
    violin.violin_overlay(data)
    annotate.significance_brackets([(1,2)])


def info_analysis():

    if style == "publication" and VERTICAL_MODE:
        fig1, ax1 = plt.subplots(figsize=fig_size)
        fig2, ax2  = plt.subplots(figsize=fig_size)
    elif style == "publication":
        fig1, ax1 = plt.subplots(figsize=(6*cm, 6*cm))
        fig2, ax2  = plt.subplots(figsize=(6*cm, 6*cm))
    else:
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
    axes = [ax1, ax2]
    figs = [fig1, fig2]

    ats, ats_high, ats_low, ats_decay = get_ats_infos(ats_folders, axes)
    slow, slow_decay = get_fixed_infos(slow_folders, axes)
    fast, fast_decay = get_fixed_infos(fast_folders, axes)

    ### Cum Info Plot
    for fig, ax in zip(figs, axes):
        plt.figure(fig.number)
        ax.set_xlim([0, ax.get_xlim()[1]])
        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.minorticks_on()
        ax.yaxis.set_tick_params(which='minor', left=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.axes.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        ax.set_xlabel("Time [min]")

    ax1.set_ylabel("Cumulative Information [AU]")
    plt.figure(figs[0].number)
    plt.tight_layout()
    plt.savefig(output_folder + "mito_info_cum_" + PLOT_MODE + ".pdf", transparent=True)

    ax2.set_ylabel("Cumulative Lightdose [AU]")
    ax2.set_xticks([0, 10])
    ax2.set_xlim([0, 10])
    plt.figure(figs[1].number)
    plt.tight_layout()
    ax2.xaxis.labelpad = - 7
    plt.savefig(output_folder + "mito_lightdose_" + PLOT_MODE + ".pdf", transparent=True)
    # plt.show()
    # say_done()

    # ### Decay Plot
    if style == "publication":
        fig, ax = plt.subplots(figsize=fig_size, tight_layout=True)
    else:
        fig, ax = plt.subplots()
    labels = ['slow', 'fast', 'EDA']
    data = [slow_decay, fast_decay, ats_decay]
    box_dict = plt.boxplot(data, labels=labels, showfliers=False, whis=[5,95], widths=0.5,
                           vert=VERTICAL_MODE)
    violin.violin_overlay(data, spread=0, vert=VERTICAL_MODE)
    # for index, item in enumerate(box_dict['boxes']):
    #     plt.setp(item, color=colors[index])
    # so it fits the size of the cumsum plot
    ax.set_xlabel(" ") if VERTICAL_MODE else ax.set_ylabel(None)
    ax.set_ylabel("Bleaching Decay [AU]") if VERTICAL_MODE else ax.set_xlabel("Bleaching Decay [AU]")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_folder + "mito_decay_" + PLOT_MODE + ".pdf", transparent=True)

    print('bleaching decay fast/EDA: ', np.mean(fast_decay)/np.mean(ats_decay))
    print('bleaching decay EDA/slow: ', np.mean(ats_decay)/np.mean(slow_decay))

    # P VALUES
    _, p_value = stats.ttest_ind(ats_high, ats_low)
    print("\n\n___P Values:___")
    print('EDA: high vs low {:.0e}'.format(p_value))
    print('low vs EDA {:.0e}'.format(p_value))

    ### Average Info plot
    if style ==  "publication":
        fig, ax = plt.subplots(figsize=(9*cm, 5*cm), tight_layout=True)
    else:
        fig, ax = plt.subplots()

    labels = ['slow', 'fast', 'EDA', 'EDA-fast', 'EDA-slow']
    data = [slow, fast, ats, ats_high, ats_low]
    plt.boxplot(data, labels=labels, showfliers=False, whis=[5,95])
    violin.violin_overlay(data, [True, True, False, False, False])
    annotate.significance_brackets([(0,2), (3,4)])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.ylabel('Information per Frame [AU]')
    plt.tight_layout()
    plt.savefig(output_folder + "mito_violin.pdf", transparent=True)

    print("Slow: N = {}, n = {}".format(len(slow), len(slow_folders)))
    print("Fast: N = {}, n = {}".format(len(fast), len(fast_folders)))
    print(" EDA: N = {}, n = {}".format(len(ats), len(ats_folders)))

    # Rational variable plot

    # ats, ats_high, ats_low, ats_decay = get_ats_infos(ats_folders, axes)
    # # slow, slow_decay = get_fixed_infos(slow_folders, axes)
    # fast, fast_decay = get_fixed_infos(fast_folders, axes)


    plt.show()

def get_ats_infos(folders, axs):
    info_ats = []
    info_ats_high = []
    info_ats_low = []
    decays = []
    for folder in folders:
        print(folder)
        json_file = folder + '/analysis_json.txt'

        if os.path.isfile(json_file):
            with open(json_file, 'r') as in_file:
                data_dict = json.load(in_file)
            info = data_dict['nn']
            high_info = data_dict['nn_high']
            low_info = data_dict['nn_low']
        else:
            filelist = sorted(glob.glob(folder + '/img_*.tif'))
            re_odd = re.compile(".*time\d*[13579]_.*tif$")
            mito_filelist = [file for file in filelist if re_odd.match(file)]
            re_even = re.compile(".*time\d*[02468]_.*")
            drp_filelist = [file for file in filelist if re_even.match(file)]
            nn_filelist = sorted(glob.glob(folder + '/img_*_nn*'))

            snr, bleaching = get_snr(mito_filelist)
            times = get_times(drp_filelist)
            fps, fps_times = make_fps(times[1:])
            info, high_info, low_info = get_info(nn_filelist, times, fps)

            data_dict = {"folder": folder,
                         "times": times,
                         "snr": snr,
                         "bleaching": bleaching,
                         "nn": info,
                         "nn_high": high_info,
                         "nn_low": low_info}

            with open(json_file, 'w') as out_file:
                json.dump(data_dict, out_file)
        info_ats.extend(info)
        info_ats_high.extend(high_info)
        info_ats_low.extend(low_info)
        # info_ats.append(np.mean(info))

        times_decay = np.divide(data_dict['times'],10_000)
        times_decay = times_decay - np.min(times_decay)
        bleaching_decay = np.divide(data_dict['bleaching'],data_dict['bleaching'][0])
        # plt.plot(times, bleaching, color=colors[2])
        params, _ = optimize.curve_fit(exp_func, times_decay, bleaching_decay, maxfev=1000)
        decays.append(params[1])
        # A, K, C = params
        # plt.plot(times, exp_func(times, A, K, C), color=colors[2])
        # plt.show()

        ### SNR plot
        # snr_plot(data_dict)
        # plt.show()

        snr_cutoff_frame = np.sum([x > 1.1 for x in data_dict['snr']])
        times_int = data_dict['times'] - np.min(data_dict['times'])
        times = data_dict['times'][1:snr_cutoff_frame+1]
        times = times - np.min(times)
        print('SNR cutoff:', times[-1])
        axs[0].plot(np.divide(times, TIME_ADJUST),
                    np.cumsum(info)[:snr_cutoff_frame],
                    color=colors[2])
        axs[1].plot(np.divide(times_int, TIME_ADJUST),
                    np.cumsum(np.ones(len(times_int))),
                    color=colors[2])

    return info_ats, info_ats_high, info_ats_low, decays


def snr_plot(data_dict):
    crop_end = 280
    fig, ax = plt.subplots()
    times_in_s = np.divide(data_dict['times'][1:crop_end+1], 1000)
    fps, fps_times = make_fps(times_in_s)
    plt.plot(np.divide(data_dict['times'][:crop_end], 1000),
             data_dict['snr'][:crop_end], color=colors[0])
    plt.xlabel('Time [s]')
    plt.ylabel('SNR')
    ax2 = ax.twinx()
    ax2.set_ylabel('Imaging Speed [fps]', color=colors[2])
    # fps = fps - np.min(fps)
    # fps = fps/np.max(fps) * (np.max(data_dict['snr']) - np.min(data_dict['snr'])) + np.min(data_dict['snr'])
    ax2.plot(fps_times, fps, colors[2])
    fps_bin = [np.sign(x-1) for x in fps]
    steps = [fps_bin[idx]+fps_bin[idx+1] for idx in range(len(fps_bin)-1)]
    positions = np.insert([x for x, v in enumerate(steps) if v == 0], 0, 0)
    positions = np.append(positions, len(fps)-1)
    start = 0 if fps_bin[0] == 1 else 1
    for idx in range(start, len(positions)-1, 2):
        rect = mpl.patches.Rectangle((fps_times[positions[idx]], np.min(fps)),  # position
                                     fps_times[positions[idx+1]]-fps_times[positions[idx]],
                                     np.max(fps) - np.min(fps),  # height
                                     linewidth=0, edgecolor='none',
                                     facecolor=(colors[2] + '50'))
        ax2.add_patch(rect)
    ax2.spines["right"].set_edgecolor(colors[2])
    ax2.tick_params(axis='y', colors=colors[2])


def get_fixed_infos(folders, axs):
    nn_data_fast = []
    decays = []
    for folder in folders:
        filelist = sorted(glob.glob(folder + '*_crop.ome.tif'))

        for file in filelist:
            img_range = range(240) #510 for fast
            print(file)
            nn_file = file[:-8] + '_nn.ome.tif'
            cropped_file = file[:-8] + '_crop.ome.tif'
            json_file = file[:-8] + '_json.txt'

            data_dict = {"file": file,
                        "nn_file": nn_file}

            if os.path.isfile(json_file):
                with open(json_file, 'r') as in_file:
                    data_dict = json.load(in_file)
            else:
                # NNio.cropOMETiff(file, cropped_file, cropFrame=np.max(img_range)+1, cropRect=True)
                # file = cropped_file
                nn_file = file[:-8] + '_nn.ome.tif'
                cropped_file = file #[:-8] + '_crop.ome.tif'
                json_file = file[:-8] + '_json.txt'

            # first, calculate the NN frames that we will need for the analysis
            if not os.path.isfile(nn_file):
                print('Calculating the NN file for this stack')
                print(file)
                model_path = '//lebnas1.epfl.ch/microsc125/Watchdog/GUI/model_Dora.h5'
                NNio.calculateNNforStack(file, model = model_path, img_range = img_range)

            #get the bleaching curve for the mito channel
            if not ("snr" in data_dict.keys() and "times" in data_dict.keys()):
                drp, mito, times, _ = NNio.loadTifStack(file, outputElapsed=True,
                                                        img_range=img_range)
                data_dict['times'] = times
            times = data_dict['times']

            if not "snr" in data_dict.keys():
                snr, bleaching = get_snr(mito)
                data_dict['snr'] = snr
                data_dict['bleaching'] = bleaching
            else:
                snr = data_dict['snr']


            # what is the nn output for these frames?
            if not "nn" in data_dict.keys():
                nn_data = []
                nn = tifffile.imread(nn_file)
                for frame in range(nn.shape[0]):
                    nn_data.append(np.max(nn[frame]))
                data_dict['nn'] = nn_data
            else:
                nn_data = data_dict['nn']

            # save all data we have obtained to a json file so we don't have to recalculate next time
            with open(json_file,'w') as out_file:
                json.dump(data_dict, out_file)
            nn_data_fast.extend(nn_data)

            color = 1 if "fast" in file else 0

            # Calculate the decay parameter for the signal to noise data
            times_decay = np.divide(data_dict['times'],10_000)
            times_decay = list(times_decay-np.min(times_decay))
            bleaching_decay = list(np.divide(data_dict['bleaching'],data_dict['bleaching'][0]))


            ### Checking the Fit
            # plt.plot(times, bleaching, color=colors[color])
            # # plt.show()
            params, _ = optimize.curve_fit(exp_func, times_decay, bleaching_decay, maxfev=1000)
            decays.append(params[1])
            # A, K, C = params
            # plt.plot(times, exp_func(np.array(times), A, K, C), color=colors[color])
            # # plt.show()

            ### SNR plot
            # fig, ax = plt.subplots()
            # fps, fps_times = make_fps(data_dict['times'][1:])
            # plt.plot(np.divide(data_dict['times'],1000), data_dict['snr'], color='#002b36')
            # plt.xlabel('Time [s]')
            # plt.ylabel('SNR')
            # plt.show()

            # plt.plot(times, snr)
            # plt.show()

            times = times - np.min(times)
            snr_cutoff_frame = np.sum([x > 1.1 for x in snr])
            print('SNR cutoff:', times[snr_cutoff_frame-1])
            axs[0].plot(np.divide(times[:snr_cutoff_frame], TIME_ADJUST),
                        np.cumsum(nn_data)[:snr_cutoff_frame],
                        color=colors[color])
            axs[1].plot(np.divide(times, TIME_ADJUST),
                        np.cumsum(np.ones(len(times))),
                        color=colors[color])
            # nn_data_fast.append(np.mean(nn_data))
    return nn_data_fast, decays


def exp_func(t, A, K, C):
    return A * np.exp(-K*t) + C


def lin_func(t, C, K):
    return K*t + C



if __name__ == "__main__":
    main()
