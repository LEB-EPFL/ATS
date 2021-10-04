
import warnings

import astropy.units as u
import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from fil_finder import FilFinder2D
from scipy import interpolate, ndimage
from scipy.optimize import curve_fit, differential_evolution
from scipy.spatial.distance import pdist, squareform
from skimage import measure, morphology, transform
from skimage.draw import line
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from SmartMicro import NNio
from toolbox.misc import speak
from toolbox.plotting import annotate, saving, unit_info, violin
from tqdm import tqdm
from events import detect_events, filter_events, get_event_frame

import data_locations
import data_handling
import tools

np.seterr(divide='ignore', invalid='ignore')


DATASET = "caulo"
PLOT = False  # plot substeps?

skip_events = []
if DATASET.lower() == "caulo":
    #
    # Caulobacter
    #
    slow_folders = data_locations.caulo_folders['slow']
    fast_folders = data_locations.caulo_folders['fast']
    ats_folders = data_locations.caulo_folders['ats']
    ats_folders.append('C:/Users/stepp/Documents/02_Raw/Caulobacter_iSIM/210506_01')
    DETECTION_THRESHOLD = 90
    save_file = "c:/Users/stepp/Documents/05_Software/Analysis/Constrictions/caulo_events"
    observation_time = 60*60  # seconds
    min_distance = 10  # pixel
    if PLOT:
        skip_events = [0]
        VMAX = 25
        OFFSETS = [-12, -2, 8]
elif DATASET.lower() == "mito":
    slow_folders = data_locations.mito_folders['slow']
    fast_folders = data_locations.mito_folders['fast']
    ats_folders = data_locations.mito_folders['ats']
    DETECTION_THRESHOLD = 80
    save_file = "c:/Users/stepp/Documents/05_Software/Analysis/Constrictions/mito_events"
    observation_time = 20  # seconds
    min_distance = 10  # pixel
    if PLOT:
        skip_events = [0]
        VMAX = 100
        OFFSETS = [-5, 5, 15]

figure_folder = "//lebsrv2.epfl.ch/LEB_PERSO/Willi-Stepp/ATS_Figures/Figure2_combined"
VERTICAL_MODE = False
PLOT_MODE = 'vert' if VERTICAL_MODE else 'hor'
FIG_SIZE = [4, 5] if VERTICAL_MODE else [8, 4]
FIG_SIZE = [i*unit_info.cm for i in FIG_SIZE]

FRAME_SIZE = 11  # pixel around the event for analysis

pixel_calib = 56  # nm per pixel
sig = 121.5/81  # in pixel
resize_param = pixel_calib/81

tools.set_plotting_parameters()


def main_v2():
    # ats_folders = ['W:/Watchdog/microM_test/201208_cell_Int0s_30pc_488_50pc_561_band_5']
    # FOR FIGURE:
    # ats_folders = ['C:/Users/stepp/Documents/02_Raw/SmartMito/'
    #                '201208_cell_Int0s_30pc_488_50pc_561_band_6']
    # slow_folders = ["C:/Users/stepp/Documents/02_Raw/Caulobacter_iSIM/slow/210526_FOV_5/Default2"]
    # constriction_v2(ats_folders, 'ats')
    # constriction_v2(fast_folders, 'fast')
    # constriction_v2(slow_folders, 'slow')
    # speak.say_done()
    plot_results()
    plt.show()
    # plt.show()


def constriction_v2(folders, mode, ats=False):
    """Detect interesting events in the data by looking at the network output. At those positions,
    Measure the constriction for a given time span. See what the smallest constriction measured
    at that position is for that time."""

    data = pd.DataFrame()
    for folder in folders:
        print(folder)
        files, stack = tools.get_files(folder)
        struct_filelist, foci_filelist, nn_filelist = files['network'], files['peaks'], files['nn']
        decon_filelist = files['decon']
        if stack:
            times = NNio.loadTifStackElapsed(stack)
        else:
            times = tools.get_times(foci_filelist)
            filelists = NNio.loadTifFolder(folder, resizeParam=resize_param,
                                           order=1, outputs=['decon'])
            struct_filelist, foci_filelist, nn_filelist, decon_filelist = filelists

        struct_filelist = decon_filelist

        nn_filelist = transform.resize(nn_filelist,
                                       struct_filelist.shape, preserve_range=True).astype(np.uint8)
        all_events = detect_events(nn_filelist, times, DETECTION_THRESHOLD)
        all_events['folder'] = folder
        events = filter_events(all_events, observation_time, min_distance)
        events = measure_events(events, struct_filelist, observation_time, times)
        print(events)
        data = data.append(events, ignore_index=True, sort=False)
    data.to_pickle(save_file + mode + '.pkl')


@tools.limit_colors(['slow', 'ats'])
@saving.saveplt(DATASET.lower() + '_constriction_' + PLOT_MODE, folder=figure_folder)
def plot_results():
    ats = np.load(save_file + 'ats.pkl', allow_pickle=True)
    slow = np.load(save_file + 'slow.pkl', allow_pickle=True)
    # fast = np.load(save_file + 'fast.pkl', allow_pickle=True)

    # filter out lists with only one or no successful measurement
    # ats = ats[ats.max_intensity > 90]
    # slow = slow[slow.max_intensity > 90]

    slow_widths = [np.nanmin(widths) for widths in slow.widths if len(widths) > 0]
    ats_widths = [np.nanmin(widths) for widths in ats.widths if len(widths) > 0]
    # fast_widths = [np.nanmin(widths) for widths in fast.widths if len(widths) > 0]

    slow_widths = [width*unit_info.nm_per_pixel_isim for width in slow_widths
                   if not np.isnan(width)]
    ats_widths = [width*unit_info.nm_per_pixel_isim for width in ats_widths if not np.isnan(width)]
    # fast_widths = [width*unit_info.nm_per_pixel_isim for width in fast_widths
    #                if not np.isnan(width)]

    data = [slow_widths, ats_widths]

    _, ax = plt.subplots(figsize=FIG_SIZE, tight_layout=True)

    ax.boxplot(data, labels=['slow', 'EDA'], showfliers=False,  whis=[5, 95], widths=[0.5, 0.5],
               vert=VERTICAL_MODE)
    violin.violin_overlay(data, bins=10, vert=VERTICAL_MODE, params={'s': 13, 'alpha': 0.6,
                                                                     'edgecolors': 'none'})
    annotate.significance_brackets([(0, 1)], vert=VERTICAL_MODE)
    plt.ylabel('Minimal Width [nm]') if VERTICAL_MODE else plt.xlabel('Minimal Width [nm]')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # plt.title(DATASET)
    plt.tight_layout()
    print('constriction slow/ats: ', np.mean(slow_widths)/np.mean(ats_widths))


def measure_events(events, struct_imgs, observation_time, times) -> pd.DataFrame:
    events['widths'] = None

    for idx, event in events.iterrows():
        if idx in skip_events:
            continue
        print(idx, "of", events.count()[0]-1)
        time = event['time']
        frame = int(event['frame'])
        events.at[idx, 'widths'] = []
        progress_bar = tqdm(total=int(observation_time*1000))
        while time < event['time'] + observation_time*1000 and frame < struct_imgs.shape[0]-1:
            ax = make_plot()
            # It,s not easy to update the position as there might not be a signal in the nn
            # anymore. We could make vertical lines on the spline and take the smallest
            # constriction as the tightest spot.
            position = [int(event['weighted_centroid-0']), int(event['weighted_centroid-1'])]
            event_frame, _ = get_event_frame(struct_imgs[frame], FRAME_SIZE, pos=position)
            frame += 1
            progress_bar.update(int(times[frame] - time))
            time = times[frame]
            # https://stackoverflow.com/questions/52014197/how-to-interpolate-a-2d-curve-in-python
            # Try different lines along the main line at try to get the width at each one
            offsets = OFFSETS  # np.arange(-50, 51, 1)
            # After test (-50, 51, 2) might reasonable before: (-30, 31, 1)
            widths = []

            backbone, event_frame, error = get_backbone(event_frame, ax=ax)

            for offset in offsets:
                if error:
                    width = np.nan
                else:
                    line, blur, error = get_line(backbone, ax=ax, version=2, offset=offset)

                if error:
                    width = np.nan
                else:
                    width = line_gauss_fit(line, event_frame, ax=ax, limits=[0, 1000])
                widths.append(width)
            try:
                widths = [i for i in widths if i and (i > 1.5 or np.isnan(i))]
                width = np.min(widths)
            except (TypeError, ValueError) as e:
                width = np.nan
                print('Widths list problem')
                print(e)
            if PLOT:
                print(f"final width {width}")
            events.at[idx, 'widths'].append(width)
            if PLOT:
                ax[0].set_xticks([])
                ax[0].set_yticks([])
                ax[1].set_xlabel('Distance [px]')
                plt.show()
        progress_bar.close()
    return events


def values_along_line(line, evt_image):
    # evt_image = tools.deconvolve(evt_image)
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
    # if max(values) > 0:
    #     values = values/max(values)
    return values, x


def line_box_fit(line, evt_image, ax=None):
    values, x = values_along_line(line, evt_image)
    # quadratic cost function
    # parameter bounds height, center, width
    res = differential_evolution(lambda p: np.sum((box(x, *p) - values)**2),
                                 [[0, 2], [0, FRAME_SIZE*2], [0.1, 10]])


def line_gauss_fit(line, evt_image, limits=[1, 6], ax=None):
    try:
        values, x = values_along_line(line, evt_image)
    except ValueError:
        values, x = np.zeros(20), np.arange(0, 20, 1)

    # for [cen, wid, amp]
    init_vals = [x[np.where(values == np.max(values))[0][0]], 5,  np.max(values)]
    param_bounds = ([2, 0, 0.1], [max(x) - 2, np.inf, np.inf])
    try:
        best_vals, covar = curve_fit(gaussian, x, values, p0=init_vals, bounds=param_bounds)
    except (RuntimeError, ValueError):
        best_vals = [0, 0, 0]
    gauss = gaussian(np.arange(0, max(x), 0.1), *best_vals)

    if ax is not None:
        # ax[0].imshow(evt_image)
        x = [i - best_vals[0] for i in x]
        ax[1].plot(x, values, alpha=0.5)
        ax[1].plot(np.arange(min(x), max(x), 0.1), gaussian(np.arange(min(x), max(x), 0.1), 0,
                                                            *best_vals[1:]),
                   alpha=0.5)
        ax[1].set_xlim([-10, 10])
        plt.tight_layout()
        print(best_vals[1])

    # width = best_vals[1]*2.3548
    # Translate to FWHM
    width = 2*np.sqrt(np.log(2)*2)*best_vals[1]
    # print("width: ", width)

    # if width > 50:
    #     width = None
    if width < limits[0] or width > limits[1]:
        width = None

    return width


def get_backbone(evt_image, ax=None):
    # Make a better mask as what we had before
    # evt_image = cv2.GaussianBlur(evt_image,(5, 5), 0)
    if DATASET == 'Mito':
        ret, mask = cv2.threshold(evt_image, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        mask = evt_image > 20

    elif DATASET == 'Caulo':
        ret, mask = cv2.threshold(evt_image, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        mask = evt_image > 3  # Was 1 for calc
    # print(ret)

    mask = ndimage.binary_fill_holes(mask)
    d = morphology.disk(3)
    mask = morphology.binary_closing(mask, selem=d)
    mask = morphology.binary_dilation(mask, selem=d)
    mask = morphology.binary_erosion(mask).astype(np.uint8)

    # Skeletonize this and get the list of points that are on the skeleton
    skel = morphology.skeletonize(mask)
    orig_skel = skel
    skel = clean_skeleton(skel)
    if skel is False:
        # There was more that one filament. Maybe already divided
        # print('More than one filament')
        if PLOT:
            ax[0].imshow(orig_skel)
            ax[1].imshow(evt_image)
        return mask, evt_image, True

    points = np.asarray(skel == 1).nonzero()

    # Spline fit along those points, first order to do that
    points = np.array(points).T

    if len(points) == 0:
        # print('No skeleton')
        if PLOT:
            ax[0].imshow(orig_skel)
            ax[1].imshow(evt_image)
        return orig_skel, evt_image, True

    if PLOT:
        ax[0].imshow(evt_image, cmap='gray', vmax=VMAX)

    try:
        sorted_points = sort_points(points)
        fitted_points = spline_fit_points(sorted_points)
    except ValueError:
        return points, evt_image, True

    return fitted_points, evt_image, False


def get_line(fitted_points, ax=None, version=1, offset=0):
    # This was used before the full frames where deconvolved
    # evt_image = tools.deconvolve(evt_image, intermediate=2)

    points_range = 5
    points = get_central_points(fitted_points, offset, points_range)
    points_unzip = list(zip(*points))

    # Fit a line through all of these points
    lm = LinearRegression(fit_intercept=True)

    try:
        axes0 = np.subtract(points_unzip[0], points[points_range][0])
        axes1 = np.subtract(points_unzip[1], points[points_range][1])
    except IndexError:
        # plt.imshow(evt_image)
        # print('Filament too short!')
        return False, None, True
    lm.fit(axes1.reshape(-1, 1), axes0)
    preds = lm.predict(np.arange(-FRAME_SIZE, FRAME_SIZE, 0.1).reshape(-1, 1))
    if PLOT:

        ax[0].plot(list(zip(*fitted_points))[1][30:], list(zip(*fitted_points))[0][30:],
                   color='#0293a4')
        # ax[0].plot(np.arange(-frame_size, frame_size, 0.1) + points[points_range][1],
        # preds + points[points_range][0])
        # ax[0].plot(points_unzip[1], points_unzip[0])

    slope_points = lm.predict(np.array([0, 1]).reshape(-1, 1))
    slope = slope_points[1] - slope_points[0]

    if abs(slope) > 1000:
        slope = 1000*np.sign(slope)
    elif slope == 0:
        slope = 0.0001
    elif abs(slope) < 0.0001:
        slope = 0.0001*np.sign(slope)
    # print(slope)
    perp_slope = -1/slope

    perp_line = np.multiply(np.arange(-FRAME_SIZE*2, FRAME_SIZE*2), perp_slope)
    perp_line = perp_line + points[points_range][0]

    if PLOT:
        ax[0].plot(np.arange(-FRAME_SIZE*2, FRAME_SIZE*2) + points[points_range][1], perp_line,
                   color='#d22a26')

    start = (-FRAME_SIZE*2, -FRAME_SIZE*perp_slope*2)
    start = (int(start[0] + points[points_range][1]), int(start[1] + points[points_range][0]))
    # start = (int(x + points[points_range][1]) for x in start)
    end = (FRAME_SIZE*2, FRAME_SIZE*perp_slope*2)
    end = (int(end[0] + points[points_range][1]), int(end[1] + points[points_range][0]))
    # end = (int(x + points[points_range][1]) for x in end)
    discrete_line = list(zip(*line(*start, *end)))

    discrete_line_frame = []
    for idx, point in enumerate(discrete_line):
        if abs(point[0]) > FRAME_SIZE*2-1 or abs(point[1]) > FRAME_SIZE*2-1:
            pass
        elif point[0] < 0 or point[1] < 0:
            pass
        else:
            discrete_line_frame.append(point)

    if ax is not None:
        # plt.imshow(evt_image)  # was blur
        # ax[0].scatter(axes1 + points[points_range][1], axes0 + points[points_range][0], alpha=0.3)
        # ax[0].scatter(points[points_range][1], points[points_range][0])
        # ax[0].scatter(*zip(*discrete_line_frame))
        ax[0].set_xlim(0, FRAME_SIZE*2+.5)
        ax[0].set_ylim(0, FRAME_SIZE*2+.5)
    return discrete_line_frame, None, False


def clean_skeleton(skeleton):
    fil = FilFinder2D(skeleton, distance=250 * u.pc, mask=skeleton, beamwidth=0*u.pix)
    fil.preprocess_image(flatten_percent=85)
    # fil.create_mask(border_masking=True, verbose=False, use_existing_mask=True)
    fil.medskel(verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fil.analyze_skeletons(branch_thresh=40*u.pix, skel_thresh=10 * u.pix,
                                  prune_criteria='length')
        except ValueError:
            print('Circular skeleton?')
            return False
    if len(fil.filaments) > 1:
        return False
    return fil.skeleton_longpath
    # Show the longest path
    plt.imshow(fil.skeleton, cmap='gray')
    plt.contour(fil.skeleton_longpath, colors='r')
    plt.axis('off')
    plt.show()


def sort_points(points):
    """ Sort points along the skeletonized line, so we can do a spline fit on this data
    afterwards. !! If there are seperated parts of the skeleton, only one will be fit.
    https://stackoverflow.com/questions/37742358/sorting-points-to-form-a-continuous-line
    """
    # Create nearest neighbor graph
    clf = NearestNeighbors(n_neighbors=2).fit(points)
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


def spline_fit_points(sorted_points, final_n_points=200):
    """https://stackoverflow.com/questions/53481596/python-image-finding-largest
    -branch-from-image-skeleton"""
    # Linear length along the line:
    distance = np.cumsum(np.sqrt(np.sum(np.diff(sorted_points, axis=0)**2, axis=1)))
    distance = np.insert(distance, 0, 0)/distance[-1]
    # Build a list of the spline function, one for each dimension:
    k_param = 3
    if len(distance) > k_param:
        splines = [interpolate.UnivariateSpline(distance, coords, k=k_param, s=2)
                   for coords in sorted_points.T]
    else:
        raise ValueError
    # Computed the spline for the asked distances:
    alpha = np.linspace(0, 1, final_n_points)
    points_fitted = np.vstack([spl(alpha) for spl in splines]).T
    return points_fitted


def get_central_points(fitted_points, offset, points_range):
    # Get point in line closest to the center
    center_point = closest_point([FRAME_SIZE+1, FRAME_SIZE+1], fitted_points)
    # Get points around that point to fit the direction
    center_point = center_point + offset
    points_range = 5
    points = fitted_points[center_point - points_range:center_point + points_range + 1]
    return points


def closest_point(point, points):
    """Get the closest point in the line to the NN detected fission spot."""
    points = np.asarray(points)
    dist_2 = np.sum((points - point)**2, axis=1)
    return np.argmin(dist_2)


def box(x, *p):
    height, center, width = p
    return height*(center-width/2 < x)*(x < center+width/2)


def gaussian(x, cen, c, amp):
    if c == 0:
        c = 0.00001
    return amp * np.exp(-(x-cen)**2 / (2 * c**2))


def make_plot(figsize=(1.8/1.2, 3/1.2)):
    if PLOT:
        fig, ax = plt.subplots(2, 1, figsize=figsize,
                               gridspec_kw={'height_ratios': [2, 1]})
    else:
        ax = None
    return ax


if __name__ == "__main__":
    main_v2()
