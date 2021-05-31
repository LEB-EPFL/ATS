"""
Get the real mito Data into a structure that can be plotted nicely and to get to the
blog post graphs that are in Doras last Basecamp post:
https://3.basecamp.com/4536561/buckets/17861748/messages/3288166766
"""
import glob
import json
import os
import re
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import numpy as np
import pyttsx3
import tifffile
from SmartMicro import NNio
from toolbox.plotting.style_mpl import set_mpl_style, set_mpl_font
from toolbox.plotting import violin

from caulo_analysis import get_info, get_snr, get_times, make_fps

#  Make a custom cycle so that slow/fast and ATS data is grouped in colors
style = "publication"
set_mpl_style(style)
set_mpl_font(size=9)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = [colors[0], colors[3], colors[2], colors[5], colors[4]]
mpl.rcParams.update({"axes.prop_cycle":cycler('color',colors)})

output_folder = "//lebsrv2.epfl.ch/LEB_Perso/Willi-Stepp/ATS_Figures/"

slow_folder = '//lebnas1.epfl.ch/microsc125/iSIMstorage/Users/Dora/20201202_smartTest/analysis/'
slow_samples = ["cell0/", "cell3/", "cell4/"]
slow_folders = [slow_folder + sample for sample in slow_samples]

# The fast file are saved in stacks that are also not cropped very well. drp1 first
fast_folder = '//lebnas1.epfl.ch/microsc125/iSIMstorage/Users/Dora/20201213_fastMito/analysis/'
fast_samples = ['sample1/', 'sample2/']#, 'sample3/']
fast_folders = [fast_folder + sample for sample in fast_samples]
# exp = 'FOV_Int0_488nm_30mw_30pc_561nm_30mw_50pc_7/' # this has 3000 timepoint in tota
ats_folder = 'W:/Watchdog/microM_test/'
ats_samples = ['201208_cell_Int0s_30pc_488_50pc_561_band_4',
               '201208_cell_Int0s_30pc_488_50pc_561_band_5',
               '201208_cell_Int0s_30pc_488_50pc_561_band_6',
               '201208_cell_Int0s_30pc_488_50pc_561_band_10']
ats_folders = [ats_folder + sample for sample in ats_samples]

def main():
    fig, ax = plt.subplots()

    slow = get_fixed_infos(slow_folders)
    fast = get_fixed_infos(fast_folders)
    ats, ats_high, ats_low = get_ats_infos(ats_folders)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Cumulative Information [AU]")
    # plt.show()
    # say_done()

    # P VALUES
    _, p_value = stats.ttest_ind(ats_high, ats_low)
    print("\n\n___P Values:___")
    print('ATS: high vs low {:.0e}'.format(p_value))
    print('low vs ATS {:.0e}'.format(p_value))

    if style ==  "publication":
        cm = 1/2.54  # centimeters in inches
        plt.subplots(figsize=(9*cm, 5*cm))
    else:
        fig, ax = plt.subplots()

    labels = ['slow', 'fast', 'ATS', 'ATS-fast', 'ATS-slow']
    data = [slow, fast, ats, ats_high, ats_low]
    plt.boxplot(data, labels=labels, showfliers=False, whis=[5,95])
    violin.violin_overlay(data, [True, True, False, False, False])

    plt.ylabel('Information per Frame [AU]')
    plt.tight_layout()
    plt.savefig(output_folder + "mito_violin.pdf", facecolor=None, edgecolor=None,
                transparent=True)
    plt.show()


def get_violin_x(data, offset, relativ_hist=False):
    x = []
    alpha = 100
    hist_data, edges = np.histogram(data, bins=20)
    if relativ_hist:
        spread = 0.4
        hist_data = np.divide(hist_data,np.max(hist_data))
    else:
        spread = 1/300

    for value in data:
        bin_num = len(edges[edges < value])-1
        x.append((np.random.random(1)-0.5)*spread*hist_data[bin_num]+offset)

    alpha = alpha/len(data)
    return alpha, x

def get_ats_infos(folders):
    info_ats = []
    info_ats_high = []
    info_ats_low = []
    for folder in folders:
        print(folder)
        json_file = folder + 'analysis_json.txt'

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

            snr = get_snr(mito_filelist)
            times = get_times(drp_filelist)
            fps, fps_times = make_fps(times[1:])
            info, high_info, low_info = get_info(nn_filelist, times, fps)

            data_dict = {"folder": folder,
                        "times": times,
                        "snr": snr,
                        "nn": info,
                        "nn_high": high_info,
                        "nn_low": low_info}

            with open(json_file,'w') as out_file:
                json.dump(data_dict, out_file)
        info_ats.extend(info)
        info_ats_high.extend(high_info)
        info_ats_low.extend(low_info)
        # info_ats.append(np.mean(info))

        ### SNR plot
        # fig, ax = plt.subplots()
        # fps, fps_times = make_fps(data_dict['times'][1:])
        # plt.plot(np.divide(data_dict['times'],1000), data_dict['snr'], color='#002b36')
        # fps = fps - np.min(fps)
        # fps = fps/np.max(fps) * (np.max(data_dict['snr']) - np.min(data_dict['snr'])) + np.min(data_dict['snr'])
        # plt.plot(np.divide(fps_times,1000), fps, custom_color_cycle[2])
        # plt.xlabel('Time [s]')
        # plt.ylabel('SNR')
        # plt.show()


        snr_cutoff_frame = np.sum([x > 1.1 for x in data_dict['snr']])
        plt.plot(np.divide(data_dict['times'][1:snr_cutoff_frame+1],1000), np.cumsum(info)[:snr_cutoff_frame],
                 color=colors[2])

    return info_ats, info_ats_high, info_ats_low

def get_fixed_infos(folders):
    nn_data_fast = []
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
                snr = get_snr(mito)
                data_dict['snr'] = snr
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

            ### SNR plot
            # fig, ax = plt.subplots()
            # fps, fps_times = make_fps(data_dict['times'][1:])
            # plt.plot(np.divide(data_dict['times'],1000), data_dict['snr'], color='#002b36')
            # plt.xlabel('Time [s]')
            # plt.ylabel('SNR')
            # plt.show()


            # plt.plot(times, snr)
            # plt.show()
            color = 1 if "fast" in file else 0

            snr_cutoff_frame = np.sum([x > 1.1 for x in snr])
            plt.plot(np.divide(times[:snr_cutoff_frame],1000), np.cumsum(nn_data)[:snr_cutoff_frame],
                     color=colors[color])
            # nn_data_fast.append(np.mean(nn_data))
    return nn_data_fast

# def get_slow_infos(folders):


def say_done():
    engine = pyttsx3.init()
    engine.setProperty('volume',1.0)
    engine.say("Hey Willi, I'm done")
    engine.runAndWait()


if __name__ == "__main__":
    main()
