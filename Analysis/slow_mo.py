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
from data_handling import ATS_Data
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
from toolbox.plotting import unit_info
import pdb
# Copy here from nnio so tensorflow is not loaded for the whole Pool
from toolbox.image_processing.overlay import Overlay
import datetime
import ffmpeg


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
# GIFMode = "slow_only_nopeak"
GIFMode = "real_time"
# GIFMode = "slow_mo"
print(GIFMode)
# FOLDER = "W:/Watchdog/microM_test/cell_IntSmart_30pc_488_50pc_561_band_5"
FOLDER = "W:/Watchdog/microM_test/cell_IntSmart_30pc_488_50pc_561_band_4"


def load_data(folder: os.PathLike, crop: Tuple[int] = None, roi: Tuple[int] = None):
    filelist = sorted(glob.glob(folder + '/img_*.tif'))
    re_odd = re.compile(r".*time\d*[13579]_.*tif$")
    mito_filelist = [file for file in filelist if re_odd.match(file)]
    re_even = re.compile(r".*time\d*[02468]_.*")
    drp1_filelist = [file for file in filelist if re_even.match(file)]
    nn_filelist = sorted(glob.glob(folder + '/img_*_nn*'))
    if len(nn_filelist) == 0:
        nn_filelist = sorted(glob.glob(folder + '/nn/*neural.tif'))

    decon_list_net = sorted(glob.glob(os.path.join(folder, 'decon_net3/img*decon.tiff')))
    decon_list_peaks = sorted(glob.glob(os.path.join(folder, 'decon_peaks3/img*decon.tiff')))
    print('got filelists')
    # for file in nn_filelist:
    #     os.remove(file)

    # return 0

    pixel_size = io.imread(mito_filelist[0]).shape
    mito_imgs = np.zeros((len(mito_filelist), pixel_size[0], pixel_size[1]))
    drp1_imgs = np.zeros((len(drp1_filelist), pixel_size[0], pixel_size[1]))
    decon_imgs_net = np.zeros((len(mito_filelist), pixel_size[0], pixel_size[1]))
    decon_imgs_peaks = np.zeros((len(mito_filelist), pixel_size[0], pixel_size[1]))
    pixel_size = io.imread(nn_filelist[0]).shape
    nn_imgs = np.zeros((len(nn_filelist), pixel_size[0], pixel_size[1]))

    all_files = zip(mito_filelist, drp1_filelist, nn_filelist, decon_list_net, decon_list_peaks)

    frame = 0
    for mito_file, drp1_file, nn_file, decon_file_net, decon_file_peaks in tqdm(all_files):
        mito_imgs[frame] = io.imread(mito_file)
        drp1_imgs[frame] = io.imread(drp1_file)
        nn_imgs[frame] = io.imread(nn_file).astype(np.uint8)
        decon_imgs_net[frame] = io.imread(decon_file_net)
        decon_imgs_peaks[frame] = io.imread(decon_file_peaks)
        frame += 1

    system, name = name_file(folder)
    times = loadElapsedTime(folder)

    if 'caulo' in system:
        orig_times = times
        times = np.round(np.diff(times)/5_0000)
    else:
        times = sorted(times)
        times = times[::2]
        orig_times = times
        times = np.round(np.diff(times)/100)

    nn_imgs = resize_network(nn_imgs, drp1_imgs.shape)

    # Crop the movie
    if crop is None:
        start, end = 0, mito_imgs.shape[0]
    else:
        start, end = crop
    if roi is None:
        x = y = 0
        width, height = decon_imgs_net[0].shape
    else:
        x, y, width, height = roi

    data = {}
    data['folder'] = folder
    # data['nn_imgs'] = nn_imgs[start:end, x:x+width, y:y+height]
    # data['drp1_imgs'] = drp1_imgs[start:end, x:x+width, y:y+height]
    # data['mito_imgs'] = mito_imgs[start:end, x:x+width, y:y+height]
    data['times'] = times[start:end]
    data['orig_times'] = orig_times[start:end]
    data['decon_net'] = decon_imgs_net[start:end, x:x+width, y:y+height]
    data['decon_peaks'] = decon_imgs_peaks[start:end, x:x+width, y:y+height]
    return data


def mito_slow_mo(data: dict, peaks: str = 'drp1_imgs'):
    bg = 0.9 if GIFMode == 'slow_only_nopeaks' else 0.93
    system, name = name_file(data['folder'])
    vmin = 8 if 'mito' in system else 1
    vmin = 3 if name == "_3" else vmin
    peak_bg = 2 if 'mito' in system else 0

    mito_prep = prepare_gif_images(data['decon_net'], bg, 1, prepared=True, netw=True, vmin=vmin)
    # mito_prep = prepare_gif_images(data['decon_net'], bg, 0.9)  # , int_range=(0, 120)
    if peaks in ['drp1_imgs', 'decon_peaks']:
        print('Preparing peak data')
        peaks_prep = prepare_gif_images(data[peaks], background=peak_bg, over_exp=1, netw=False)
    else:
        peaks_prep = data[peaks]

    if 'slow_only' in GIFMode:
        subset_nn, subset_times = subset_fast(peaks_prep, data['times'])
        subset_struct, _ = subset_fast(mito_prep, data['times'])
    else:
        subset_nn = peaks_prep
        subset_times = data['times']
        subset_orig_times = data['orig_times']
        subset_struct = mito_prep

    print("Applying LUTs")
    nn_lut = apply_nn_colormap(subset_nn, subset_times, non_diff=True)
    mito_lut = apply_struct_colormap(subset_struct, subset_nn, subset_times)

    print('Starting Rust')
    if GIFMode != "slow_only_nopeak":
        imgs = screen_stack_wrap(nn_lut.astype(np.uint8), mito_lut.astype(np.uint8))
    else:
        imgs = mito_lut.astype(np.uint8)

    # resize if too small for overlay
    min_size = 256
    if imgs.shape[1] < min_size:
        new_size = imgs.shape[1] * (min_size // imgs.shape[1])
        resize_param = new_size/imgs.shape[1]
        imgs = transform.resize(imgs, (imgs.shape[0], new_size, new_size), order=0,
                                preserve_range=True, anti_aliasing=False)
        imgs = imgs.astype(np.uint8)
    else:
        resize_param = 1

    # pdb.set_trace()
    if GIFMode in ["slow_mo", "real_time"]:
        overlay = make_overlay(imgs.shape, subset_times, subset_orig_times, scale=3,
                               resize_param=resize_param)
        imgs = screen_stack_wrap(imgs, overlay.astype(np.uint8))
    imgs = np.array(imgs).astype(np.uint8)

    # folder = 'c:/Users/stepp/Documents/05_Software/Analysis/2106_Publication/add_slowmo/'
    folder = '//lebsrv2.epfl.ch/LEB_PERSO/Willi-Stepp/ATS_Figures/Movies/'
    os.makedirs(folder, exist_ok=True)
    sub_number = '0' if GIFMode == 'real_time' else '1'
    file_path = folder + "Suppl_Video" + name + '_' + sub_number + '.gif'

    imgs = np.concatenate([make_titleslide(imgs.shape, "Suppl_Video" + name + '_' + sub_number,
                                           description(name, sub_number)), imgs]).astype(np.uint8)
    # file_path = folder + 'overlay_test.gif'
    # frame_times = np.ones(len(data['times']))
    frame_times = np.divide(subset_times, 8)
    frame_times = np.append([4], frame_times)
    speed = 2 if 'mito' in system else 1
    if 'slow_only' in GIFMode:
        savegif(imgs, frame_times, 4, file_path, GIFMode)
    elif GIFMode == 'real_time':
        savegif(imgs, frame_times, speed, file_path, GIFMode)
    elif GIFMode == 'slow_mo':
        savegif(imgs, frame_times, speed, file_path, GIFMode)

    stream = ffmpeg.input(file_path)
    mp4_file = '.'.join(file_path.split('.')[:-1]) + '.mp4'
    os.system(f'ffmpeg -i {file_path} -y -vf scale=-4:720 -vcodec libx264 -pix_fmt yuv420p {mp4_file}')


def name_file(folder):
    system = 'mito_' if 'cell' in folder else 'caulo_'
    if system == 'mito_':
        if 'cell_Int0s_30pc_488_50pc_561_band_4' in folder:
            name = '_1'
        elif 'IntSmart_30pc' in folder:
            name = '_2'
        elif 'Int0s_30pc_488_50pc_561_band_10' in folder:
            name = '_3'
        # name = folder[-2:] if 'IntSmart_30pc' not in folder else '_4_2'
    else:
        if '210506_FOV_1' in folder:
            name = '_4'
        if '210602' in folder:
            name = '_sync2'
    return system, name


def description(name, mode):
    if name in ['_1', '_2', '_3']:
        description = ("Mitochondria (grey) Drp1 (red)\n"
                       "original fast frame rate 3.8 fps\n"
                       "original slow frame rate 0.2 fps")
    else:
        description = ("C.crescentus (grey) FtsZ (red)\n"
                       "original fast frames 3 min\n"
                       "original slow frames 9 min")
    if mode == "0":
        description = "Real_time\n" + description
    return description


def make_titleslide(shape, title, description):
    overlay = np.zeros([1, *shape[1:4]])
    context = Overlay(shape[1:3])
    context.title_slide(title, description)
    overlay[0] = context.get_image()
    return overlay


def make_overlay(shape, times, orig_times, scale=1, resize_param=1):
    overlay = np.zeros(shape)
    orig_times = [time - np.min(orig_times) for time in orig_times]
    if max(orig_times) > 60*60*1000:
        hours = True
        seconds = False
        milliseconds = False
    else:
        hours = False
        seconds = True
        milliseconds = False

    for frame, (time, orig_time) in enumerate(zip(times, orig_times)):
        context = Overlay(shape[1:3])
        context.timestamp(time_str(orig_time, hours, seconds, milliseconds))
        context.scale_bar(unit_info.px_per_micron_isim*resize_param)
        if time == np.max(times):
            pass
        elif time == np.min(times):
            context.slow_sign(scale=scale)
        else:
            print('look at the times!')
        overlay[frame] = context.get_image()
    return overlay


def time_str(time, hours: bool = False, seconds: bool = True, milliseconds: bool = False):
    time_str = str(datetime.timedelta(milliseconds=time))
    time_def = 'h:mm:ss.ms'

    if not hours:
        time_str = time_str.split(':')[1:]
        time_def = time_def.split(':')[1:]
        time_str = ':'.join(time_str)
        time_def = ':'.join(time_def)

    if not milliseconds:
        time_str = time_str.split('.')[0]
        time_def = time_def.split('.')[0]

    if not seconds:
        time_str = time_str.split(':')[:-1]
        time_def = time_def.split(':')[:-1]
        time_str = ':'.join(time_str)
        time_def = ':'.join(time_def)

    return time_str + ' ' + time_def


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


def savegif(stack, times, fps, out_file, gif_mode='real_time'):
    """ Save a gif that uses the right frame duration read from the files. This can be sped up
    using the fps option"""
    times = np.divide(times, fps).tolist()
    if gif_mode == 'real_time':
        times = times[:-1] + [5]
    elif gif_mode == 'slow_mo':
        times = [3] + list(np.ones(len(times))*1/fps)[:-1] + [5]
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


def prepare_gif_images(orig_imgs, background=0.85, over_exp=1, int_range=None, prepared=False,
                       netw=True, vmin=None):
    """ Prepare images for slow-mo gifs """
    image_int_range = int_range
    imgs_prep = np.zeros_like(orig_imgs)
    for idx, mito_img in enumerate(orig_imgs):
        if not prepared:
            mito_img = prepare.prepare_image(mito_img, background)
        if int_range is None:
            if netw:
                image_int_range_max = np.max(mito_img)*over_exp
            else:
                image_int_range_max = np.max(orig_imgs)*over_exp
            if vmin is None:
                image_int_range_min = np.min(mito_img)
            else:
                image_int_range_min = vmin
            image_int_range = (image_int_range_min, image_int_range_max)
        imgs_prep[idx] = exposure.rescale_intensity(
            mito_img, image_int_range, out_range=(0, 255))
    return imgs_prep


def apply_nn_colormap(nn_imgs, times, non_diff=False):
    # pdb.set_trace()
    nn_imgs_normalized = nn_imgs/np.max(nn_imgs)
    shape = [len(nn_imgs), nn_imgs[0].shape[0], nn_imgs[0].shape[1]]
    shape.append(4)
    shape = tuple(shape)
    nn_lut = np.zeros(shape)
    for frame, nn_img in enumerate(nn_imgs_normalized):
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


# mito_data = slow_mo.load_data("W:/Watchdog/microM_test/cell_IntSmart_30pc_488_50pc_561_band_4", crop=(13,93), roi=(25,591,128,128))
# mito_data = slow_mo.load_data(r"W:/Watchdog/microM_test\201208_cell_Int0s_30pc_488_50pc_561_band_4", crop=(126,356), roi=(364,90,299,299))

# folder = r'W:\Watchdog\microM_test\201208_cell_Int0s_30pc_488_50pc_561_band_10'
# mito_data = slow_mo.load_data(folder, crop=(15, 130), roi=(368, 361, 128, 128))
