import glob
from SmartMicro.NNio import defineCropRect
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import tifffile
import json
import xmltodict
from scipy import stats
import os
import re

from ATSSim_analysis import makeStepSeries
from toolbox.plotting import violin, boxplot
from toolbox.plotting.style_mpl import set_mpl_style, set_mpl_font
from cycler import cycler

style = "publication"
set_mpl_style(style)
set_mpl_font(9)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = [colors[0], colors[3], colors[2], colors[5], colors[4]]
mpl.rcParams.update({"axes.prop_cycle":cycler('color',colors)})

output_folder = "//lebsrv2.epfl.ch/LEB_Perso/Willi-Stepp/ATS_Figures/"

def main():

    plt.subplots()
    # Plotting of the cum_info versus time will be done inside the functions to limit outputs
    high_info, low_info = analyse_ats_data()
    ats_info = high_info + low_info
    fast_info = analyse_fast_data()
    slow_info = analyse_slow_data()


    print('high info: {:.2f}'.format(np.mean(high_info)))
    print('low  info: {:.2f}'.format(np.mean(low_info)))
    labels = ["slow", "fast", 'ATS', 'ATS-fast', 'ATS-slow']
    data = [slow_info, fast_info, ats_info, high_info, low_info] #, slow_info]
    if style == "publication":
        cm = 1/2.54  # centimeters in inches
        plt.subplots(figsize=(9*cm, 5*cm))
    else:
        plt.subplots()

    plt.boxplot(data, labels=labels, showfliers=False, whis=[5,95])
    violin.violin_overlay(data)
    plt.ylabel('Information per Frame [AU]')
    # plt.title("Caulobacter ATS")
    _, p_value = stats.ttest_ind(slow_info, ats_info)
    print(p_value)
    plt.tight_layout()
    plt.savefig(output_folder + "caulo_violin.pdf", facecolor=None, edgecolor=None,
                transparent=True)
    plt.show()

def analyse_slow_data():
    print('Slow data')
    folder = "C:/Users/stepp/Documents/02_Raw/Caulobacter_iSIM/slow/"
    # samples = ["0", "1", "2", "3", "4", "6", "7", "8", "10", "11", "13"]
    samples = ["210526_FOV_5/Default0", "210526_FOV_5/Default1", "210526_FOV_5/Default2",
               "210526_FOV_5/Default3"]
    folders = [folder + sample + '/' for sample in samples]
    info_all = []
    for folder in folders:
        bact_filelist, ftsz_filelist, nn_filelist = get_files(folder)
        times = get_times(ftsz_filelist)
        snr = get_snr(bact_filelist,  rect=[50, 974, 50, 974])
        # fig, ax1 = plt.subplots()

        info = get_info(nn_filelist, times)
        info_all.extend(info)
        snr_cutoff_frame = np.sum([x > 1.02 for x in snr])
        plt.plot(times[1:snr_cutoff_frame+1], np.cumsum(info)[:snr_cutoff_frame],
                 color=colors[0])

    return info_all

def analyse_fast_data():
    print('Fast data')
    folder = "c:/Users/stepp/Documents/02_Raw/Caulobacter_iSIM/fast/"
    samples = ["FOV_5/Default0", "FOV_5/Default1", "FOV_5/Default2", "FOV_5/Default3",
               "FOV_6/Default0", "FOV_6/Default1", "FOV_6/Default2", "FOV_6/Default3"]
    folders = [folder + sample + '/' for sample in samples]
    info_all = []
    for folder in folders:
        bact_filelist, ftsz_filelist, nn_filelist = get_files(folder)
        snr = get_snr_fast(bact_filelist, rect=[50, 974, 50, 974])
        times = get_times(ftsz_filelist)
        # fig, ax1 = plt.subplots()
        # ax1.plot(times[1:], snr[1:], color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])#, color = COLORS['slow'])
        # plt.show()

        info = get_info(nn_filelist, times)
        info_all.extend(info)
        snr_cutoff_frame = np.sum([x > 1.02 for x in snr])
        plt.plot(times[1:snr_cutoff_frame+1], np.cumsum(info)[:snr_cutoff_frame],
                 color=colors[1])
    return info_all

def analyse_ats_data():
    print('ATS data')
    folders = ['C:/Users/stepp/Documents/02_Raw/Caulobacter_iSIM/210414_06',
               'C:/Users/stepp/Documents/02_Raw/Caulobacter_iSIM/210416_02',
               'C:/Users/stepp/Documents/02_Raw/Caulobacter_iSIM/210416_03']
    # folders = ['W:/Watchdog/microM_test/201208_cell_Int0s_30pc_488_50pc_561_band_6']
    #folder = 'C:/Users/stepp/Documents/02_Raw/Caulobacter_iSIM/210421_23'
    high_info_all = []
    low_info_all = []

    for folder in folders:
        bact_filelist, ftsz_filelist, nn_filelist = get_files(folder)

        # Calculate the SNR for every bacteria image
        snr = get_snr(bact_filelist)

        # This is the SNR plot, let's see which parts are high/low fps
        # get the timting of all the drp images
        times = get_times(ftsz_filelist)

        # The first frame was 5 seconds for a long time
        fps, fps_times = make_fps(times[1:])

        # #Let's plot this data
        # fig, ax1 = plt.subplots()
        # ax1.plot(fps_times, fps)#, color = COLORS['ats'])
        # ax2 = ax1.twinx()
        # ax1.plot(times[1:], snr[1:], color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])#, color = COLORS['slow'])
        # plt.show()

        #Now, what is the mean 'info' in a frame that was taken at low fps vs high fps
        info, high_info, low_info = get_info(nn_filelist, times, fps)


        # #Let's plot this data
        # fig, ax1 = plt.subplots()
        # ax1.plot(fps_times, fps)#, color = COLORS['ats'])
        # ax2 = ax1.twinx()
        # ax2.plot(times[1:], info, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])#, color = COLORS['slow'])
        # plt.show()

        high_info_all.extend(high_info[:])
        low_info_all.extend(low_info[:])

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
        plt.plot(times[1:snr_cutoff_frame+1], np.cumsum(info)[:snr_cutoff_frame],
                 color=colors[2])
        # plt.plot(times[1:], np.cumsum(info),
        #          color=custom_color_cycle[2])

    return high_info_all, low_info_all

def get_files(folder):

    if os.path.isfile(folder + '/img_channel001_position000_time000000000_z000.tif'):
        bact_filelist = sorted(glob.glob(folder + '/img_channel001*'))
        ftsz_filelist = sorted(glob.glob(folder + '/img_channel000*.tif'))
    else:
        print('No channels here')
        filelist = sorted(glob.glob(folder + '/img_*.tif'))
        re_odd = re.compile(".*time\d*[13579]_.*tif$")
        bact_filelist = [file for file in filelist if re_odd.match(file)]
        re_even = re.compile(".*time\d*[02468]_.*")
        ftsz_filelist = [file for file in filelist if re_even.match(file)]
    nn_filelist = sorted(glob.glob(folder + '/img_*_nn*'))

    return bact_filelist, ftsz_filelist, nn_filelist

def get_info(filelist, times, fps = None):
    high_info = []
    low_info = []
    info_list = []
    high = np.max(fps)
    low = np.min(fps)

    for index in range(1,len(filelist)):
        image = cv2.imread(filelist[index],-1)
        info = np.max(image)
        info_list.append(info)
        if fps is not None:
            factor = 1/low
            if round(1/(times[index]- times[index-1])*factor) == round(high*factor):
                high_info.append(info)
            elif round(1/(times[index]- times[index-1])*factor) == round(low*factor):
                low_info.append(info)
            else:
                print('fps does not match high or low level')

    if fps is None:
        return info_list
    else:
        return info_list, high_info, low_info

def get_snr(filelist, rect=None):
    snr = []
    for file in filelist:
        if os.path.isfile(file):
            image = cv2.imread(file,-1)
        else:
            image = file

        if rect is not None:
            image = image[rect[0]:rect[1], rect[2]:rect[3]]
        blur = cv2.GaussianBlur(image,(5,5),0)
        blur = cv2.medianBlur(blur, 5)
        ret3, mask = cv2.threshold(blur,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        invert_mask = cv2.bitwise_not(mask).astype(np.bool)
        mask = mask.astype(np.bool)
        # plt.imshow(mask)
        # plt.show()
        mean_signal = np.mean(image[mask])
        mean_noise = np.mean(image[invert_mask])
        snr.append(mean_signal/mean_noise)
    return snr

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

def get_times(filelist):
    times = []
    for file in filelist:
        with tifffile.TiffFile(file) as tif:
            try:
                times.append(json.loads(tif.imagej_metadata['Info'])['ElapsedTime-ms'])
            except TypeError:
                mdInfoDict = xmltodict.parse(tif.ome_metadata)
                times.append(float(mdInfoDict['OME']['Image']['Pixels']['Plane']['@DeltaT']))
    return times

def make_fps(times):
    fps = []
    x_axis = []
    last_fps = 0
    for index in range(1,len(times)):
        if not 1/(times[index] - times[index-1]) == last_fps and index > 1:
            fps.append(1/(times[index] - times[index-1]))
            x_axis.append(times[index-1])
        fps.append(1/(times[index] - times[index-1]))
        x_axis.append(times[index])
        last_fps = 1/(times[index] - times[index-1])
    return fps, x_axis

def split_frame(folder):
    from SmartMicro.NNio import cropOMETiff, defineCropRect, calculateNNforFolder
    import tifffile
    for i in range(4):
        os.mkdir(folder + str(i))
    bact_filelist, ftsz_filelist, _ = get_files(folder)
    for index, (bact_file, ftsz_file) in enumerate(zip(bact_filelist, ftsz_filelist)):
        bact_filename = os.path.basename(bact_file)
        ftsz_filename = os.path.basename(ftsz_file)
        for rect_index, rec_pos in enumerate([[0, 0], [0, 1], [1, 0], [1, 1]]):
            out_folder = folder + str(rect_index) + '/'
            bact_out = out_folder + bact_filename
            ftsz_out = out_folder + ftsz_filename
            # rect = (rec_pos[0][0]*1024, rec_pos[0][1]*1024, 1024, 1024)
            for file, file_out in zip([bact_file, ftsz_file], [bact_out, ftsz_out]):
                with tifffile.TiffFile(file) as tif:
                    ijInfo = tif.imagej_metadata
                    ijInfoDict = json.loads(ijInfo['Info'])
                    ijInfo['Info'] = json.dumps(ijInfoDict)
                    img = tifffile.imread(file)
                    rec = np.multiply(rec_pos, 1024)
                    cropped_img = img[rec[0]:rec[0]+1024,
                                      rec[1]:rec[1]+1024]
                    tifffile.imwrite(file_out, cropped_img, imagej=True,
                                     metadata={'Info': ijInfo['Info']})
    for i in range(4):
        nn_folder = folder + str(i)
        calculateNNforFolder(nn_folder)


if __name__ == "__main__":
    # folders = ["c:/Users/stepp/Documents/02_Raw/Caulobacter_iSIM/fast/FOV_5/Default",
    #            "c:/Users/stepp/Documents/02_Raw/Caulobacter_iSIM/fast/FOV_6/Default",
    #            "c:/Users/stepp/Documents/02_Raw/Caulobacter_iSIM/slow/210526_FOV_5/Default"]

    # for folder in folders:
    #     split_frame(folder)
    main()