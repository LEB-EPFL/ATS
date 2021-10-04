import glob
import json
import os
import re
from math import e
from multiprocessing import Pool
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, interpolate
np.seterr(divide='ignore', invalid='ignore')
import tifffile
from scipy.optimize import curve_fit, differential_evolution
from scipy.spatial.distance import pdist, squareform

from skimage import measure, transform, morphology
from skimage.draw import line
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
import networkx as nx

from SmartMicro import NNio
from toolbox.plotting import annotate, violin

import pandas as pd

import data_locations
import tools

DATASET = "Mito"
PLOT = True

if DATASET == "Caulo":
    ##### Caulobacter
    slow_folders = data_locations.caulo_folders['slow']
    fast_folders = data_locations.caulo_folders['fast']
    ats_folders = data_locations.caulo_folders['ats']
else:
    #### Mito
    slow_folders = data_locations.mito_folders['slow']
    fast_folders = data_locations.mito_folders['fast']
    ats_folders = data_locations.mito_folders['ats']
    DETECTION_THRESHOLD = 80
    FRAME_SIZE = 20  # pixel around the event for analysis
    frame_size = 20


pixel_calib = 56  # nm per pixel
sig = 121.5/81  # in pixel
resize_param = pixel_calib/81


def main_v1():
    print('Slow Data')
    slow_width, slow_info, slow_intensity = constriction(slow_folders)
    slow_width = list(filter(None, slow_width))
    print('Fast Data')
    fast_width, fast_info, fast_intensity = constriction(fast_folders)
    fast_width  = list(filter(None, fast_width))
    print('ATS Data')
    ats_width, ats_info, ats_intensity = constriction(ats_folders)
    ats_width  = list(filter(None, ats_width))


    plt.subplots()
    # # full_length = len(all_width)

    # # info = list(filter(None, all_info))
    # perc_of_data = len(data)/full_length
    # print("percentage of data in the plot: {}".format(perc_of_data))
    labels = ["slow", "fast", 'ATS']
    data = [slow_width, fast_width, ats_width]
    data = [slow_intensity, fast_intensity, ats_intensity]
    # data = [slow_info, fast_info, ats_info]
    plt.boxplot(data, labels=labels, showfliers=False, whis=[5,95])
    violin.violin_overlay(data, bins=50)
    annotate.significance_brackets([(0,2), (1,2)])
    plt.title('Constriction')
    plt.show()


def constriction(folders, ats=False):
    all_widths = []
    all_width_high = []
    all_width_low = []
    all_infos = []
    all_intensities = []
    for folder in folders:
        files, stack = tools.get_files(folder)
        bact_filelist, ftsz_filelist, nn_filelist = files['network'], files['peaks'], files['nn']
        widths = []
        if ats:
            width_high = []
            width_low = []
            times = tools.get_times(ftsz_filelist)
            fps, fps_times = tools.make_fps(times[1:])
            high, low = np.max(fps), np.min(fps)
        infos = []
        intensities = []
        snr, _ = tools.get_snr(bact_filelist)
        # with Pool(11) as p:
        #      widths = p.map(get_width_pool, zip(bact_filelist, nn_filelist))

        # Check if filelist or tif-stack for the data
        # Stack is only true for the non-adaptive data in mitos
        if stack:
            for idx in range(bact_filelist.shape[0]):
                if PLOT:
                    fig, ax = plt.subplots(1,2, figsize=(12,7))
                else:
                    ax = None
                width, info = get_width(bact_filelist[idx], nn_filelist[idx], ax=ax)
                intensity = get_intensity(ftsz_filelist[idx], nn_filelist[idx])
                widths.append(width)
                intensities.append(intensity)
                print("intensity: ", intensity)
                if PLOT:
                    plt.show()
        else:
            for idx, (bact_file, ftsz_file, nn_file) in enumerate(zip(
                bact_filelist, ftsz_filelist, nn_filelist)):
                if PLOT:
                    fig, ax = plt.subplots(1,2, figsize=(12,7))
                else:
                    ax = None
                intensity = get_intensity(ftsz_file, nn_file)

                width, info = get_width(bact_file, nn_file, ax=ax)
                widths.append(width)
                if width is not None and intensity is not None:
                    infos.append(width*intensity)
                else:
                    infos.append(None)
                intensities.append(intensity)

                # ATS data will always be here, because it is not recorded in stacks
                if ats:
                    factor = 1/low
                    if round(1/(times[idx]- times[idx-1])*factor) == round(high*factor):
                        width_high.append(width)
                    elif round(1/(times[idx]- times[idx-1])*factor) == round(low*factor):
                        width_low.append(width)
                    else:
                        print('fps does not match high or low level')
                if PLOT:
                    plt.show()
            # print(widths)
        if ats:
            all_width_high.extend(width_high)
            all_width_low.extend(width_low)
        all_widths.extend(widths)
        all_infos.extend(list(filter(None, infos)))
        all_intensities.extend(list(filter(None, intensities)))

    if ats:
        all_widths = {'all': all_widths,
                      'high': all_width_high,
                      'low': all_width_low}
    json_file = os.path.dirname(folders[1]) + "/analysis.json"
    print(json_file)
    with open(json_file,'w') as out_file:
        json.dump(all_widths, out_file)
    return all_widths, all_infos, all_intensities


def get_width_pool(inputs):
    return get_width(inputs[0], inputs[1])


def get_intensity(ftsz_file, nn_file, ax=None):
    if isinstance(ftsz_file, (str, list)):
        nn_image = cv2.imread(nn_file, -1)
        ftsz_image = cv2.imread(ftsz_file, -1)
    else:
        nn_image = nn_file
        ftsz_image = ftsz_file

    nn_image = cv2.resize(nn_image, ftsz_image.shape)
    mean_ftsz_image = np.mean(ftsz_image)
    #20 for ftsz in Caulo
    size = 3
    evt_image, pos = get_event_frame(nn_image, ftsz_image, size)
    # plt.imshow(evt_image)
    # plt.show()
    # Activate this for Caulo intensity
    line = list(zip((np.ones(size*2+1)*(size+1)).astype(int),range(size*2+2)))
    intensity = line_gauss_fit(line, evt_image, limits=[1,6], ax=None)

    # This for Mito
    intensity = np.mean(evt_image)/mean_ftsz_image
    return intensity


def get_width(bact_file, nn_file, ax=None) -> Tuple[int, int]:
    if isinstance(bact_file, (str, list)):
        nn_image = cv2.imread(nn_file, -1)
        bact_image = cv2.imread(bact_file, -1)
    else:
        nn_image = nn_file
        bact_image = bact_file
    # plt.show()

    info = np.max(nn_image)

    nn_image = cv2.resize(nn_image, bact_image.shape)

    # evt_decon, pos = get_event_frame(nn_image, bact_decon)
    evt_image, pos = get_event_frame(nn_image, bact_image, frame_size)

    # Don't go on if the event is too close to the edge
    if pos == False:
        return None
    elif min(pos) < frame_size:
        return None

    # nn_crop, pos = get_event_frame(nn_image, nn_image)

    line, blur = get_line(evt_image, ax)
    width = line_gauss_fit(line, evt_image, ax=ax)

    return width, info

def values_along_line(line, evt_image):
    evt_image = tools.deconvolve(evt_image)
    values = []
    x = []
    for point in line:
        try:
            values.append(evt_image[point[1], point[0]])
            x.append(np.sqrt((point[1] - line[0][1])**2 + (point[0] - line[0][0])**2))
        except IndexError:
            pass
            # print('Point skipped because out of image')
            # print(point)

    values = values - min(values)
    if max(values) > 0:
        values = values/max(values)
    return values, x


def line_gauss_fit(line, evt_image, limits=[1, 6], ax=None):
    values, x = values_along_line(line, evt_image)

    # print(values)

    init_vals = [x[np.where(values == np.max(values))[0][0]], 1, 1]  # for [cen, wid, amp]
    param_bounds=([0,0,0.2],[np.inf,np.inf,np.inf])
    try:
        best_vals, covar = curve_fit(gaussian, x, values, p0=init_vals, bounds=param_bounds)
    except (RuntimeError, ValueError):
        best_vals = [0, 0, 0]
    gauss = gaussian(np.arange(0,max(x), 0.1), *best_vals)

    if ax is not None:
        # ax[0].imshow(evt_image)
        ax[1].plot(x,values)
        ax[1].plot(np.arange(0,max(x), 0.1), gauss)

    # width = best_vals[1]*2.3548
    # Translate to FWHM
    width = 2*np.sqrt(np.log(2)*best_vals[1])
    # print("width: ", width)

    # if width > 50:
    #     width = None
    if  width < limits[0] or width > limits[1] :
        width = None

    return width


def get_line(evt_image, ax=None, version=1, offset=0):

    # try:
    #     blur = cv2.GaussianBlur(evt_image,(5,5),0)
    # except:
    #     return None
    # blur = cv2.medianBlur(blur, 5)
    # ret, mask = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    evt_image = tools.deconvolve(evt_image, intermediate=True)
    ret, mask = cv2.threshold(evt_image, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Skeletonize this and get the list of points that are on the skeleton
    skel = morphology.skeletonize(mask)
    points = np.asarray(skel == 1).nonzero()


    # Fit a line through all of these points
    lm = LinearRegression(fit_intercept=True)
    axes0 = np.subtract(points, frame_size)
    axes1 = np.subtract(points, frame_size)
    lm.fit(axes1.reshape(-1, 1), axes0)
    preds = lm.predict(np.arange(-frame_size, frame_size, 0.1).reshape(-1,1))
    if PLOT:
        ax[0].imshow(evt_image)
        # ax[0].plot(list(zip(*points_fitted))[1], list(zip(*points_fitted))[0])
        ax[0].plot(np.arange(-frame_size, frame_size, 0.1) + frame_size, preds + frame_size)
        # ax[0].plot(points_unzip[1], points_unzip[0])

    slope_points = lm.predict(np.array([0, 1]).reshape(-1,1))
    slope = slope_points[1] - slope_points[0]

    if abs(slope) > 1000:
        slope = 1000*np.sign(slope)
    elif slope == 0:
        slope = 0.0001
    elif abs(slope) < 0.0001:
        slope = 0.0001*np.sign(slope)
    # print(slope)
    perp_slope = -1/slope

    perp_line = np.multiply(np.arange(-frame_size, frame_size), perp_slope)
    perp_line = perp_line + frame_size

    if PLOT:
        ax[0].plot(np.arange(-frame_size, frame_size) + frame_size, perp_line)

    start = (-frame_size, -frame_size*perp_slope)
    start = (int(start[0] + frame_size), int(start[1] + frame_size))
    # start = (int(x + frame_size) for x in start)
    end = (frame_size, frame_size*perp_slope)
    end = (int(end[0] + frame_size), int(end[1] + frame_size))
    # end = (int(x + frame_size) for x in end)
    discrete_line = list(zip(*line(*start, *end)))

    discrete_line_frame = []
    for idx, point in enumerate(discrete_line):
        if abs(point[0]) > frame_size*2-1  or abs(point[1]) > frame_size*2-1:
            pass
        elif point[0] < 0 or point[1] < 0:
            pass
        else:
            discrete_line_frame.append(point)

    if ax is not None:
        # plt.imshow(evt_image)  # was blur
        ax[0].scatter(axes1 + frame_size, axes0 + frame_size, alpha=0.3)
        ax[0].scatter(frame_size, frame_size)
        ax[0].scatter(*zip(*discrete_line_frame))
    return discrete_line_frame, evt_image

def sort_points(points):
    """ Sort points along the skeletonized line, so we can do a spline fit on this data
    afterwards
    https://stackoverflow.com/questions/37742358/sorting-points-to-form-a-continuous-line
    """
    # Create nearest neighbor graph
    clf = NearestNeighbors(2).fit(points)
    G = clf.kneighbors_graph()
    # Construct a graph
    T = nx.from_scipy_sparse_matrix(G)

    # Get all the possible paths
    paths = [list(nx.dfs_preorder_nodes(T, i)) for i in range(len(points))]

    # Get the path that has the shortest point to point distances
    mindist = np.inf
    minidx = 0

    for i in range(len(points)):
        p = paths[i]           # order of nodes
        ordered = points[p]    # ordered nodes
        # find cost of that order by the sum of euclidean distances between points (i) and (i+1)
        cost = (((ordered[:-1] - ordered[1:])**2).sum(1)).sum()
        if cost < mindist:
            mindist = cost
            minidx = i

    opt_order = paths[minidx]
    sorted_points = points[opt_order]

    return sorted_points


def closest_point(point, points):
    """Get the closest point in the line to the NN detected fission spot."""
    points = np.asarray(points)
    dist_2 = np.sum((points - point)**2, axis=1)
    return np.argmin(dist_2)

def box(x, *p):
    height, center, width = p
    return height*(center-width/2 < x)*(x < center+width/2)

def gaussian(x, cen, wid, amp):
    if wid == 0:
        wid = 0.00001
    return amp * np.exp(-(x-cen)**2 / wid)


def get_event_frame(bact_image, frame_size, nn_image=None, pos=None):
    if pos is None:
        pos = list(zip(*np.where(nn_image == np.max(nn_image))))[0]
    max0 = pos[0] + frame_size + 1
    max1 = pos[1] + frame_size + 1

    if max0 > bact_image.shape[0] or max1 > bact_image.shape[1]:
        return False, False

    frame = bact_image[pos[0] - frame_size:pos[0] + frame_size + 1,
                       pos[1] - frame_size:pos[1] + frame_size + 1]

    return frame, pos


def skeletonize(img):
    """ OpenCV function to return a skeletonized version of img, a Mat object"""

    #  hat tip to http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/

    img = img.copy() # don't clobber original
    skel = img.copy()

    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv2.countNonZero(img) == 0:
            break

    return skel

def make_plot():
    if PLOT:
        fig, ax = plt.subplots(1,2, figsize=(12,7))
    else:
        ax = None
    return ax

if __name__ == "__main__":
    main_v1()
