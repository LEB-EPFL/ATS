""" Generates gifs with a nice visualization of the slow-motion effect of ATS """

#%%
# from tensorflow import keras
from enum import Enum
from layeris.layer_image import LayerImage
import glob
import json
import os
import time
import cairo
from multiprocessing import Pool
import re
from typing import Tuple
import cv2
import imageio
import numpy as np
import tifffile
import xmltodict
from img_rust import screen_stack_wrap
import matplotlib.colors as mplcolors
from matplotlib import cm
from matplotlib import pyplot as plt
from skimage import exposure, filters, io, transform
from tqdm import tqdm
import pyttsx3
from toolbox.image_processing import prepare
import pdb
# Copy here from nnio so tensorflow is not loaded for the whole Pool


cdict = {'red': ((0.0, 0.0, 0.0),
                 (1.0, 1.0, 1.0)),
         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
         'blue': ((0.0, 0.0, 0.0),
                  (1.0, 0.0, 0.0))}
dark_red = mplcolors.LinearSegmentedColormap('my_colormap', cdict, 256)


def main():
    # caulo_slow_mo()
    mito_slow_mo()


# GIFMode = "slow_only"
GIFMode = "slow_mo"
GIFMode = "slow_only_nopeak"


# FOLDER = "W:/Watchdog/microM_test/cell_IntSmart_30pc_488_50pc_561_band_5"
FOLDER = "W:/Watchdog/microM_test/cell_IntSmart_30pc_488_50pc_561_band_4"


def load_data(folder: os.PathLike, crop: Tuple[int] = None, roi: Tuple[int] = None):
    filelist = sorted(glob.glob(folder + '/img_*.tif'))
    re_odd = re.compile(r".*time\d*[13579]_.*tif$")
    mito_filelist = [file for file in filelist if re_odd.match(file)]
    re_even = re.compile(r".*time\d*[02468]_.*")
    drp1_filelist = [file for file in filelist if re_even.match(file)]
    nn_filelist = sorted(glob.glob(folder + '/img_*_nn*'))
    print('got filelists')
    # for file in nn_filelist:
    #     os.remove(file)

    # return 0

    pixel_size = io.imread(mito_filelist[0]).shape
    mito_imgs = np.zeros((len(mito_filelist), pixel_size[0], pixel_size[1]))
    drp1_imgs = np.zeros((len(drp1_filelist), pixel_size[0], pixel_size[1]))
    pixel_size = io.imread(nn_filelist[0]).shape
    nn_imgs = np.zeros((len(nn_filelist), pixel_size[0], pixel_size[1]))

    frame = 0
    for mito_file, drp1_file, nn_file in tqdm(zip(mito_filelist, drp1_filelist, nn_filelist)):
        mito_imgs[frame] = io.imread(mito_file)
        drp1_imgs[frame] = io.imread(drp1_file)
        nn_imgs[frame] = io.imread(nn_file).astype(np.uint8)
        frame += 1

    times = loadElapsedTime(folder)
    times = sorted(times)
    times = times[::2]
    times = np.round(np.diff(times)/100)

    nn_imgs = resize_network(nn_imgs, drp1_imgs.shape)

    # Crop the movie
    if crop is None:
        start, end = 0, nn_imgs.shape[0]
    else:
        start, end = crop
    data = {}
    x, y, width, height = roi
    data['nn_imgs'] = nn_imgs[start:end, x:x+width, y:y+height]
    data['drp1_imgs'] = drp1_imgs[start:end, x:x+width, y:y+height]
    data['mito_imgs'] = mito_imgs[start:end, x:x+width, y:y+height]
    data['times'] = times[start:end]
    return data


def mito_slow_mo(data: dict, peaks: str = 'drp1_imgs'):
    bg = 0.9 if GIFMode == 'slow_only_nopeaks' else 0.93

    mito_prep = prepare_gif_images(data['mito_imgs'], bg, 0.9)  # , int_range=(0, 120)
    if peaks == 'drp1_imgs':
        peaks_prep = prepare_gif_images(data[peaks], 1.1, 0.99)
    else:
        peaks_prep = data[peaks]

    if 'slow_only' in GIFMode:
        subset_nn, subset_times = subset_fast(peaks_prep, data['times'])
        subset_struct, _ = subset_fast(mito_prep, data['times'])
    else:
        subset_nn = peaks_prep
        subset_times = data['times']
        subset_struct = mito_prep

    print("Applying LUTs")
    nn_lut = apply_nn_colormap(subset_nn, subset_times, non_diff=True)
    mito_lut = apply_struct_colormap(subset_struct, subset_nn, subset_times)

    print('Starting Rust')
    if GIFMode != "slow_only_nopeak":
        imgs = screen_stack_wrap(nn_lut.astype(np.uint8), mito_lut.astype(np.uint8))
    else:
        imgs = mito_lut.astype(np.uint8)

    if GIFMode == "slow_mo":
        overlay = make_overlay(mito_lut.shape, data['times'], scale=3)
        imgs = screen_stack_wrap(imgs, overlay.astype(np.uint8))
    # nn_lut = nn_lut[:, :, :, 0:3]
    # mito_lut = mito_lut[:, :, :, 0:3]

    # imgs = np.zeros(nn_lut.shape)
    # for idx, (nn, mito) in enumerate(zip(nn_lut, mito_lut)):
    #     nn_interm = LayerImage.from_array(nn)
    #     imgs[idx] = nn_interm.screen(mito).get_image_as_array()

    imgs = np.array(imgs).astype(np.uint8)

    file_path = 'c:/Users/stepp/Documents/05_Software/Analysis/2106_Publication/bigframe_mito_' + GIFMode + '.gif'
    # frame_times = np.ones(len(data['times']))
    frame_times = np.divide(subset_times, 8)
    if 'slow_only' in GIFMode:
        savegif(imgs, frame_times, 4, file_path, GIFMode)
    elif GIFMode == 'slow_mo':
        savegif(imgs, frame_times, 2, file_path, GIFMode)


def make_overlay(shape, times, scale=1):
    overlay = np.zeros(shape)
    for frame, time in enumerate(times):
        if time == np.max(times):
            pass
        elif time == np.min(times):
            overlay[frame] = create_slow_sign(overlay.shape[1:3], scale)
        else:
            print('look at the times!')
    return overlay


def create_slow_sign(shape, scale=1):
    width, height = shape
    shape = np.multiply(shape, scale)
    data = np.zeros((width, height, 4), dtype=np.uint8)
    surface = cairo.ImageSurface.create_for_data(data, cairo.FORMAT_ARGB32, width, height)
    cr = cairo.Context(surface)
    cr.set_source_rgb(1, 1, 1)
    cr = triangle(cr, shape)
    cr = line(cr, shape)

    return data


POS = 1/30
LINE_WIDTH = 1/70


def triangle(cr, shape):
    width, height = shape
    start = POS + LINE_WIDTH + 1/200
    cr.move_to(width*start, height*POS)
    cr.line_to(width*start, height*POS*2)
    cr.line_to(width*start*1.4, height*POS + height*POS/2)
    cr.fill()
    return cr


def line(cr, shape):
    width, height = shape
    pos = POS
    line_width = width*LINE_WIDTH
    cr.move_to(width*pos, height*pos)
    cr.line_to(width*pos + line_width, height*pos)
    cr.line_to(width*pos + line_width, height*pos*2)
    cr.line_to(width*pos, height*pos*2)
    cr.fill()
    return cr


def caulo_slow_mo():
    """ Make a Caulobacter slow-motion like gif """
    MODEL_PATH = '//lebnas1.epfl.ch/microsc125/Watchdog/GUI/model_Dora.h5'
    FOLDER = 'W:/Watchdog/bacteria/210507_Syncro/FOV_2/Default'
    FOLDER = 'W:/Watchdog/bacteria/210512_Syncro/FOV_3/Default'

    # MODEL = keras.models.load_model(MODEL_PATH, compile=True)

    #%% Load the individual files
    bact_filelist = sorted(glob.glob(FOLDER + '/img_channel001*.tif'))
    ftsz_filelist = sorted(glob.glob(FOLDER + '/img_channel000*.tif'))
    nn_filelist = sorted(glob.glob(FOLDER + '/img_*_nn*'))

    pixel_size = io.imread(bact_filelist[0]).shape
    bact_imgs = np.zeros((len(bact_filelist), pixel_size[0], pixel_size[1]))
    ftsz_imgs = np.zeros((len(ftsz_filelist), pixel_size[0], pixel_size[1]))
    pixel_size = io.imread(nn_filelist[0]).shape
    nn_imgs = np.zeros((len(nn_filelist), pixel_size[0], pixel_size[1]))


    frame = 0
    for bact_file, ftsz_file, nn_file in zip(bact_filelist, ftsz_filelist, nn_filelist):
        bact_imgs[frame] = io.imread(bact_file)
        ftsz_imgs[frame] = io.imread(ftsz_file)
        nn_imgs[frame] = io.imread(nn_file).astype(np.uint8)
        frame += 1


    #%% Get the timing for this array
    times = loadElapsedTime(FOLDER)
    times = np.round(np.diff(times)/10_0000)

    # Crop the movie
    start = 10
    end = 22
    nn_imgs = nn_imgs[start:end]
    ftsz_imgs = ftsz_imgs[start:end]
    bact_imgs = bact_imgs[start:end]
    times = times[start:end]

    # %% Prepare all the images
    frame = 0
    bact_prep = np.zeros(nn_imgs.shape)
    ftsz_prep = np.zeros(nn_imgs.shape)
    # imgs = np.array(imgs).astype(np.uint8)
    for bact_img, ftsz_img in zip(bact_imgs, ftsz_imgs):
        # bact_img_prep, ftsz_img_prep = prepare_gif_image(bact_img, ftsz_img)
        bact_img_prep= prepare_gif_image(bact_img)
        bact_prep[frame] = bact_img_prep.astype(np.uint8)
        # ftsz_prep[frame] = ftsz_img_prep.astype(np.uint8)
        frame += 1
    #%% Apply luts to the frames

    nn_lut = apply_nn_colormap(nn_imgs, times)
    bact_lut = apply_struct_colormap(bact_prep, nn_imgs, times)

    #%% Mix the channels

    # imgs = np.zeros(nn_lut.shape)

    # print('Starting pool')
    # with Pool(12) as p:
    #     imgs = p.map(screen_frame, zip(nn_lut, bact_lut))
    # imgs = np.array(imgs).astype(np.uint8)


    ### No pool
    # imgs = list(map(screen_frame, zip(nn_lut, bact_lut)))
    # imgs = np.array(imgs).astype(np.uint8)

    #This calls a very fast function in rust_img to do the screening
    print('Starting Rust')
    imgs = screen_stack_wrap(nn_lut.astype(np.uint8), bact_lut.astype(np.uint8))
    imgs = np.array(imgs).astype(np.uint8)

    # file_path = 'c:/Users/stepp/Documents/05_Software/Analysis/210525_GroupMeeting/ATS_210512.gif'
    # # frame_times = np.ones(len(times))*4
    # frame_times = times
    # savegif(imgs, frame_times, 5, file_path)


def savegif(stack, times, fps, out_file, gif_mode='eda'):
    """ Save a gif that uses the right frame duration read from the files. This can be sped up
    using the fps option"""
    times = np.divide(times, fps).tolist()
    if gif_mode == 'eda':
        times = times[:-1] + [5]
    elif gif_mode == 'slow_mo':
        times = list(np.ones(len(times))*1/fps)[:-1] + [5]
    print(times)
    print(stack.shape)
    imageio.mimsave(out_file, stack, duration=times)


def resize_network(stack, target_shape):
    stack = transform.resize(stack, target_shape, preserve_range=True)
    return stack.astype(np.uint8)


def screen(color1, color2):
    r = np.round((1 - (1 - color1[0] / 255) * (1 - color2[0] / 255)) * 255)
    g = np.round((1 - (1 - color1[1] / 255) * (1 - color2[1] / 255)) * 255)
    b = np.round((1 - (1 - color1[2] / 255) * (1 - color2[2] / 255)) * 255)
    alpha = 255 - ((255 - color1[3]) * (255 - color2[3]) / 255)
    return (r, g, b, alpha)


def screen_frame(frames):
    frame1 = frames[0]
    frame2 = frames[1]
    screened = np.zeros(frame1.shape)
    for row in range(frame1.shape[0]):
        for column in range(frame1.shape[1]):
            screened[row, column] = screen(frame1[row, column], frame2[row, column])

    return screened


def loadElapsedTime(folder, progress=None, app=None):
    """ get the Elapsed time for all .tif files in folder or from a stack """

    elapsed = []
    # Check for folder or stack mode
    # if not re.match(r'*.tif*', folder) is None:
    #     # get microManager elapsed times if available
    #     with tifffile.TiffFile(folder) as tif:
    #         for frame in range(0, len(tif.pages)):
    #             elapsed.append(
    #                 tif.pages[frame].tags['MicroManagerMetadata'].value['ElapsedTime-ms'])
    # else:
    if os.path.isfile(folder + '/img_channel001_position000_time000000000_z000.tif'):
        fileList = sorted(glob.glob(folder + '/img_channel001*'))
        numFrames = len(fileList)
    else:
        fileList = glob.glob(folder + '/img_*[0-9].tif')
        numFrames = int(len(fileList)/2)


    if progress is not None:
        progress.setRange(0, numFrames*2)
    i = 0
    for filePath in fileList:
        with tifffile.TiffFile(filePath) as tif:
            try:
                mdInfo = tif.imagej_metadata['Info']  # pylint: disable=E1136  # pylint/issues/3139
                if mdInfo is None:
                    mdInfo = tif.shaped_metadata[0]['Infos']  # pylint: disable=E1136
                mdInfoDict = json.loads(mdInfo)
                elapsedTime = mdInfoDict['ElapsedTime-ms']
            except (TypeError, KeyError) as _:
                mdInfoDict = xmltodict.parse(tif.ome_metadata, force_list={'Plane'})
                elapsedTime = float(mdInfoDict['OME']['Image']['Pixels']['Plane'][0]['@DeltaT'])

            elapsed.append(elapsedTime)
        if app is not None:
            app.processEvents()
        # Progress the bar if available
        if progress is not None:
            progress.setValue(i)
        i = i + 1

    return elapsed


def prepare_gif_image(bact_img):
    """ Prepare images for slow-mo gifs """
    # Set iSIM specific values
    pixelCalib = 56  # nm per pixel
    sig = 121.5/81  # in pixel
    resizeParam = pixelCalib/81  # no unit

    bact_prep = transform.rescale(bact_img, resizeParam)
    # Contrast settings
    contrastMax = 255

    # Contrast
    bact_prep = exposure.rescale_intensity(bact_prep, (np.mean(bact_prep), np.max(bact_prep)),
                                           out_range=(0, contrastMax))

    bact_prep = cv2.medianBlur(bact_prep.astype(np.uint8), 5)

    # ret, mask = cv2.threshold(bact_prep,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # print(ret)
    # bact_prep = exposure.rescale_intensity(
    #     bact_prep, (np.mean(bact_prep), np.max(bact_prep)), out_range=(0, contrastMax)).astype(np.uint8)


    # gaussian and background subtraction
    bact_prep = filters.gaussian(bact_prep, sig, preserve_range=True)

    bact_prep = exposure.rescale_intensity(bact_prep, (np.mean(bact_prep), np.max(bact_prep)),
                                           out_range=(0, contrastMax))

    return bact_prep


def prepare_gif_images(orig_imgs, background=0.85, over_exp=1, int_range=None):
    """ Prepare images for slow-mo gifs """
    image_int_range = int_range
    imgs_prep = np.zeros_like(orig_imgs)
    for idx, mito_img in enumerate(orig_imgs):
        mito_img = prepare.prepare_image(mito_img, background)
        if int_range is None:
            image_int_range = (np.min(mito_img), np.max(mito_img)*over_exp)
        imgs_prep[idx] = exposure.rescale_intensity(
            mito_img, image_int_range, out_range=(0, 255))
    return imgs_prep


def apply_nn_colormap(nn_imgs, times, non_diff=False):
    nn_imgs_normalized = nn_imgs/np.max(nn_imgs)*1.5
    shape = [len(nn_imgs), nn_imgs[0].shape[0], nn_imgs[0].shape[1]]
    shape.append(4)
    shape = tuple(shape)
    nn_lut = np.zeros(shape)
    for frame, nn_img in enumerate(nn_imgs_normalized[1:]):
        if non_diff:
            colormap = dark_red
        elif times[frame] == np.max(times):
            colormap = dark_red
        elif times[frame] == np.min(times):
            colormap = cm.gnuplot2
        else:
            print('look at the times!')
        nn_lut[frame] = (colormap(nn_img)*255).astype(np.uint8)
    return nn_lut


def apply_struct_colormap(struct_prep, nn_imgs, times):
    # pdb.set_trace()
    struct_imgs_normalized = np.divide(struct_prep, 255)
    struct_lut = (cm.gray(struct_imgs_normalized)*255).astype(np.uint8)
    ###
    ### Make Bacteria with changing color
    ###
    # shape = list(nn_imgs.shape)
    # shape.append(4)
    # shape = tuple(shape)
    # struct_lut = np.zeros(shape)
    # for frame, struct_img in enumerate(bact_imgs_normalized[1:]):
    #     if times[frame] == np.max(times):
    #         colormap = cm.gray
    #     elif times[frame] == np.min(times):
    #         colormap = cm.bone
    #     else:
    #         print('look at the times!')
        # struct_lut[frame] = (colormap(struct_img)*255).astype(np.uint8)

    return struct_lut


def subset_fast(bact_imgs, times):
    frame = 0
    fast_count = 0

    new_times = []
    bact_slow_only = []
    while frame < len(times):
        # pdb.set_trace()
        print(f'frame {frame} with fast_count {fast_count}')
        if times[frame] == np.max(times):
            bact_slow_only.append(bact_imgs[frame])
            new_times.append(np.max(times))
            print('added image')
            fast_count = 0
        elif times[frame] == np.min(times) and fast_count < np.max(times)/np.min(times):
            fast_count += 1
        elif times[frame] == np.min(times):
            bact_slow_only.append(bact_imgs[frame])
            new_times.append(np.max(times))
            print('added subimage from fast')
            fast_count = 0
        else:
            print('look at the times!')
        frame += 1
    return bact_slow_only, new_times


def say_done():
    engine = pyttsx3.init()
    engine.setProperty('volume',1.0)
    engine.say("Hey, I'm done")
    engine.runAndWait()


if __name__ == "__main__":
    main()
    say_done()