import glob
import re
import tifffile
import json
import xmltodict
import numpy as np
import cv2
import os
from SmartMicro import NNio
from typing import List, Set, Dict, Tuple, Optional
from skimage.restoration import richardson_lucy
from skimage import filters, segmentation, feature, measure, morphology
from scipy import optimize, ndimage
import matplotlib.pyplot as plt
from tqdm import tqdm

import matplotlib as mpl
from toolbox.plotting.style_mpl import set_mpl_style, set_mpl_font
from cycler import cycler

# def get_files(folder):
#     filelist = sorted(glob.glob(folder + '/img_*.tif'))
#     re_odd = re.compile(".*time\d*[13579]_.*tif$")
#     mito_filelist = [file for file in filelist if re_odd.match(file)]
#     re_even = re.compile(".*time\d*[02468]_.*")
#     drp_filelist = [file for file in filelist if re_even.match(file)]
#     nn_filelist = sorted(glob.glob(folder + '/img_*_nn*'))
#     files = {'network': mito_filelist,
#              'peaks': drp_filelist,
#              'nn': nn_filelist}
#     return files


def get_files(folder):
    stack = False
    if os.path.isfile(folder + '/img_channel001_position000_time000000000_z000.tif'):
        bact_filelist = sorted(glob.glob(folder + '/img_channel001*_z*'))
        ftsz_filelist = sorted(glob.glob(folder + '/img_channel000*.tif'))
        nn_filelist = sorted(glob.glob(folder + '/img_*_nn*'))
        decon_filelist = sorted(glob.glob(folder + '/img_*_decon*'))
    elif os.path.isfile(folder + '/img_channel000_position000_time000000000_z000.tif'):
        print('No channels here')
        filelist = sorted(glob.glob(folder + '/img_*.tif'))
        re_odd = re.compile(r".*time\d*[13579]_.*tif$")
        bact_filelist = [file for file in filelist if re_odd.match(file)]
        re_even = re.compile(r".*time\d*[02468]_.*")
        ftsz_filelist = [file for file in filelist if re_even.match(file)]
        nn_filelist = sorted(glob.glob(folder + '/img_*_nn*'))
        decon_filelist = sorted(glob.glob(folder + '/img_*_decon*'))
    else:
        print("Image stacks")
        files = sorted(glob.glob(folder + '*_crop.ome.tif'))[0]
        print(files)
        nn_file = files[:-8] + '_nn.ome.tif'
        decon_file = files[:-8] + '_decon.tiff'
        cropped_file = files
        ftsz_filelist, bact_filelist = NNio.loadTifStack(cropped_file)
        nn_filelist = tifffile.imread(nn_file)
        try:
            decon_filelist = tifffile.imread(decon_file)
        except FileNotFoundError:
            decon_filelist = False
            print('No decon file present')
        stack = cropped_file

    files = {'network': bact_filelist,
             'peaks': ftsz_filelist,
             'nn': nn_filelist,
             'decon': decon_filelist}
    return files, stack


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


def make_fps(times: List[int], unit: str):
    if unit == 'h':
        times_for_y = times*60*60
    else:
        times_for_y = times
    fps = []
    x_axis = []
    last_fps = 0
    for index in range(1, len(times)):
        if not 1/(times[index] - times[index-1]) == last_fps and index > 1:
            fps.append(1/(times_for_y[index] - times_for_y[index-1]))
            x_axis.append(times[index-1])
        fps.append(1/(times_for_y[index] - times_for_y[index-1]))
        x_axis.append(times[index])
        last_fps = 1/(times[index] - times[index-1])
    return fps, x_axis


def get_info(filelist, times, fps=None):
    high_info = []
    low_info = []
    info_list = []
    high = np.max(fps)
    low = np.min(fps)

    for index in range(1, len(filelist)):
        image = cv2.imread(filelist[index], -1)
        info = np.max(image)
        info_list.append(info)
        if fps is not None:
            factor = 1/low
            if round(1/(times[index] - times[index-1])*factor) == round(high*factor):
                high_info.append(info)
            elif round(1/(times[index] - times[index-1])*factor) == round(low*factor):
                low_info.append(info)
            else:
                print('fps does not match high or low level')

    if fps is None:
        return info_list
    else:
        return info_list, high_info, low_info


def get_snr(filelist, rect=None):
    """Calculate the SNR for a list of images. Can be both stacks or a list of files as str.
    """
    snr = []
    bleaching = []
    start_area = None
    for idx, file in enumerate(filelist):
        if os.path.isfile(file):
            image = cv2.imread(file, -1)
        else:
            image = file

        if rect is not None:
            image = image[rect[0]:rect[1], rect[2]:rect[3]]

        blur = cv2.GaussianBlur(image, (5, 5), 0)
        blur = cv2.medianBlur(blur, 5)
        ret3, mask = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        invert_mask = cv2.bitwise_not(mask).astype(np.bool)
        mask = mask.astype(np.bool)
        start_area = np.sum(mask) if idx == 0 else start_area
        area = np.sum(mask)
        # plt.imshow(mask)
        # plt.show()
        mean_signal = np.mean(image[mask])
        mean_noise = np.mean(image[invert_mask])
        bleaching.append(mean_signal)
        snr.append(mean_signal/mean_noise*(area/start_area))
    return snr, bleaching


def split_frame(folder):
    """ This splits full 2048x2048 frames into four parts while copying the metadata.
    Used because the data in 1024x1024 frames is much closer to what has been recorded
    in ATS mode. Keeping the Metadata is important, because some later timing functions
    for example rely on the recorded time to be present in the metadata."""
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


def adjust_times(elapsed: list) -> Tuple[list, int]:
    if (elapsed[-1]-elapsed[0]) < 10_000*60:
        elapsed = np.array(elapsed)/1000
        timeUnit = 's'
    elif (elapsed[-1]-elapsed[0]) < 120_000*60:
        elapsed = np.array(elapsed)/1000/60
        timeUnit = 'min'
    else:
        elapsed = np.array(elapsed)/1000/60/60
        timeUnit = 'h'

    return elapsed, timeUnit


def deconvolve_stack(stack, intermediate: int = 0):
    """ Deconvolve all frames in a stack. First dimension is frames"""
    for i in tqdm(range(stack.shape[0])):
        stack[i] = deconvolve(stack[i], intermediate)
    return stack


def deconvolve(evt_image, intermediate: int = 0):
    x, y = np.meshgrid(np.linspace(-3, 3, 11), np.linspace(-3, 3, 11))
    d = np.sqrt(x*x+y*y)
    sigma, mu = 4.0/2.3548, 0.0
    psf = np.exp(-((d-mu)**2 / (2.0 * sigma**2)))
    psf = psf/np.sum(psf)  # *np.sum(evt_image)
    psf[psf < 0.000001] = 0
    # print(psf)
    # evt_image = (evt_image - np.median(evt_image)).astype(np.uint16)
    blur = cv2.medianBlur(evt_image, 5)
    blur = cv2.GaussianBlur(blur, (0, 0), 2)
    ret3, mask = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    idx = (mask == 1)
    evt_image = evt_image - np.min(evt_image[idx])
    idx = (mask == 0)
    evt_image[idx] = 0
    # plt.imshow(evt_image)
    # evt_image[evt_image < 0] = 0
    evt_image = evt_image.astype(np.uint16)

    if intermediate:
        return evt_image

    # plt.imshow(evt_image)
    # plt.show()

    evt_image = cv2.medianBlur(evt_image, 5)
    evt_image = cv2.GaussianBlur(evt_image, (0, 0), 2)
    if intermediate == 2:
        return evt_image
    evt_image = cv2.normalize(evt_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                              dtype=cv2.CV_32F)
    npad = np.ceil(3*sigma).astype(np.uint8)
    evt_image = cv2.copyMakeBorder(evt_image, npad, npad, npad, npad, cv2.BORDER_REPLICATE)
    evt_image = evt_image + 0.00001
    evt_decon = richardson_lucy(evt_image, psf)
    # evt_decon = (evt_decon*255).astype(np.uint8)
    evt_decon = evt_decon[npad+1:-(npad*1)+1, npad+1:-(npad*1)+1]
    return evt_decon


def get_decay(times, snr, cutoff=0.9):
    def exp_func(t, A, K, C):
        return A * np.exp(-K*t) + C
    times_decay = np.divide(times, 1_000_000)
    times_decay = times_decay - np.min(times_decay)
    bleaching_decay = snr
    # bleaching_decay = np.divide(snr,snr[0])
    # plt.plot(times, bleaching, color=colors[2])
    params, _ = optimize.curve_fit(exp_func, times_decay, bleaching_decay,
                                   p0=[0.1, 1, np.min(bleaching_decay)],
                                   maxfev=1000)

    params2, _ = optimize.curve_fit(exp_func, times_decay,
                                    bleaching_decay-np.max(bleaching_decay) + 1,
                                    p0=[0.1, 1, np.min(bleaching_decay)],
                                    maxfev=1000)
    cutoff_time = calc_cutoff_time(params2, cutoff)
    # plt.figure()
    # plt.plot(times_decay, bleaching_decay)
    # plt.plot(times_decay, exp_func(times_decay, params[0], params[1], params[2]))
    # print(cutoff_time)
    # plt.show()
    # print('fitting done')
    return params[1], cutoff_time


def calc_cutoff_time(params, cutoff=0.9):
    return np.log(params[0]/(cutoff-params[2]))/params[1]


def distance_watershed(img, coords=None, sigma=0.1):
    """Segmentation of events of interest based on Distance Transform Watershed of the Hessian
    probability map of divisions. Originally from Santiago Rodriguez
    https://github.com/LEB-EPFL/MitoSplit-Net/blob/main/mitosplit-net/preprocessing.py"""
    img_smooth = filters.gaussian(img, sigma)  # Smoothing so local maxima are well-defined
    distance = ndimage.distance_transform_edt(img_smooth)

    if coords is None:
        # Division sites as makers
        coords = feature.peak_local_max(distance, labels=img_smooth > 0)
        coords = tuple(coords.T)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[coords] = True
    markers = ndimage.label(mask)[0]
    # Watershed
    return segmentation.watershed(-distance, markers, mask=img)


def set_plotting_parameters():
    style = "publication"
    set_mpl_style(style)
    set_mpl_font(9)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # type: ignore
    colors = [colors[0], colors[3], colors[2], colors[5], colors[4]]
    mpl.rcParams.update({"axes.prop_cycle": cycler('color', colors)})
    cm = 1/2.54  # centimeters in inches
    fig_size = (5.5*cm, 5.5*cm)
    return fig_size


def limit_colors(color_list):
    color_dict = {'slow': 0, 'fast': 1, 'ats': 2, 'ats_slow': 3, 'ats_fast': 4}
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    new_colors = []
    for type_color in color_list:
        new_colors.append(colors[color_dict[type_color]])
    mpl.rcParams.update({"axes.prop_cycle": cycler('color', new_colors)})

    def plot(plotting_function):
        def plot_wrapper(*args, **kwargs):
            plotting_function(*args, **kwargs)
            mpl.rcParams.update({"axes.prop_cycle": cycler('color', colors)})
            return plotting_function
        return plot_wrapper
    return plot
