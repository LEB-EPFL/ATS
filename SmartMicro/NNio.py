'''
Input/Ouput module for data generated using the network_watchdog approach for adaptive temporal
sampling on the iSIM
'''

import contextlib
import glob
import json
import os
import re
import threading
from pathlib import Path
from tkinter import messagebox

import h5py
import imageio
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import xmltodict
from matplotlib.widgets import RectangleSelector
from skimage import io
from tensorflow.python.training.tracking.util import streaming_restore
from tqdm import tqdm

from SmartMicro import ImageTiles, NNfeeder


def loadiSIMmetadata(folder):
    """  Load the information written by matlab about the generated DAQ signals for all in folder
    """
    delay = []
    fileList = sorted(glob.glob(folder + '/iSIMmetadata*.txt'))
    for name in fileList:
        txtFile = open(name)
        data = txtFile.readline().split('\t')
        try:
            data = [float(i) for i in data]
            txtFile.close()
            # set minimum cycle time
            if data[1] == 0:
                data[1] = 0.2
            numFrames = int(data[2])
            delay = np.append(delay, np.ones(numFrames)*data[1])
        except ValueError:
            # This means that delay was > 240 and an empty frame was added
            delay[-1] = delay[-1] + 240
    return delay


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


def loadNNData(folder):
    """ load the csv file written by NetworkWatchdog when processing live ATS data """
    file = 'output.txt'
    filePath = folder + '/' + file
    try:
        nnData = np.genfromtxt(filePath, delimiter=',')
    except OSError:
        print('There is no output.txt here!')
        nnData = []
    return nnData


def loadRationalData(folder, order=0):
    """ Get the maxima of the Drp/FtsZ channel to compare to the neural network output """
    fileList = glob.glob(folder + '/img_*.tif')
    channelMode = False
    for file in fileList:
        if re.findall(r'.*channel001.*', file):
            channelMode = True
            fileList = glob.glob(folder + '/img_*channel00{}*.tif'.format(order))
            break
    fileList = sorted(fileList)

    if not channelMode:
        fileList = fileList[order::2]
        print(fileList)

    rationalOutput = np.zeros(len(fileList))
    for index, file in tqdm(enumerate(fileList)):
        image = tifffile.imread(file)
        rationalOutput[index] = np.max(image)
    plt.imshow(image)
    plt.show()
    return rationalOutput


def resaveNN(folder):
    """ Function to resave files that have been written in float format """
    for filePath in glob.glob(folder + '/img_*_nn.tiff'):
        img = tifffile.imread(filePath)
        newFile = filePath[:-5] + '_fiji.tiff'
        tifffile.imsave(newFile, img.astype(np.uint8))


def resaveH5(file, newFile=None):
    """ Resave an h5 file that was not compressed into a compressed format """
    if newFile is None:
        fileHandle = h5py.File(file, 'a')
        newFileHandle = h5py.File(file[:-3] + '_temp.h5', 'w')
    else:
        fileHandle = h5py.File(file, 'r')
        newFileHandle = h5py.File(newFile, 'w')

    for item in fileHandle.keys():
        oldData = np.array(fileHandle.get(item))
        if newFile is None:
            del fileHandle[item]
        newFileHandle.create_dataset(item, data=oldData, compression='gzip')

    fileHandle.close()
    newFileHandle.close()
    os.remove(file)
    os.rename(file[:-3] + '_temp.h5', file)


def makeOuputTxt(folder):
    """ function that will make an output.txt as NetworkWatchdog would for a folder that already
    has the neural network images in it."""
    fileList = glob.glob(folder + '/img_*_nn.tiff')
    txtFile = (folder + '/output.txt')
    for file in tqdm(fileList):
        nnImage = tifffile.imread(file)
        frameNum = int(file[-17:-8])
        output = np.max(nnImage)
        file = open(txtFile, 'a')
        file.write('%d, %d\n' % (frameNum, output))
        file.close()


def makePrepImages(folder, model):
    """ Prepare the _prep images to enable virtual mode on an ATS folder """
    fileList = glob.glob(folder + '/img_*z000.tif')
    baseName = '/img_channel000_position000_time'
    for frame in tqdm(range(int(len(fileList)/2))):
        mitoFile = (folder + baseName + str((frame*2 + 1)).zfill(9) + '_z000.tif')
        mitoFilePrep = (folder + baseName + str((frame*2 + 1)).zfill(9) + '_z000_prep.tif')
        drpFile = (folder + baseName + str((frame*2)).zfill(9) + '_z000.tif')
        drpFilePrep = (folder + baseName + str((frame*2)).zfill(9) + '_z000_prep.tif')

        mito = tifffile.imread(mitoFile)
        drp = tifffile.imread(drpFile)

        inputData, positions = NNfeeder.prepareNNImages(mito, drp, model)
        if model.layers[0].input_shape[0][1] is None:
            # Just copy the full frame if a full frame network was used
            mitoDataFull = inputData[0, :, :, 0, 0]
            drpDataFull = inputData[0, :, :, 1, 0]
        else:
            # Stitch the tiles made back together if model needs it
            i = 0
            st0 = positions['stitch']
            st1 = None if st0 == 0 else -st0
            mitoDataFull = np.zeros((positions['px'][-1][-1], positions['px'][-1][-2]))
            drpDataFull = np.zeros_like(mitoDataFull)
            for pos in positions['px']:
                mitoDataFull[pos[0]+st0:pos[2]-st0, pos[1]+st0:pos[3]-st0] = \
                    inputData[i, st0:st1, st0:st1, 0]
                drpDataFull[pos[0]+st0:pos[2]-st0, pos[1]+st0:pos[3]-st0] = \
                    inputData[i, st0:st1, st0:st1, 1]
                i += 1

        tifffile.imwrite(mitoFilePrep, mitoDataFull)
        tifffile.imwrite(drpFilePrep, drpDataFull)


def loadTifFolder(folder, resizeParam=1, order=0, progress=None, cropSquare=True) -> np.ndarray:
    """Function to load SATS data from a folder with the individual tif files written by
    microManager. Inbetween there might be neural network images that are also loaded into
    an array. Mainly used with NN_GUI_v2.py

    Args:
        folder ([type]): Folder with the data inside
        resizeParam (int, optional): Parameter that the loaded images are shrunken to later. This
        is also the size of the neural network images. Defaults to 1.
        order (int, optional): Order of the data. In the original case mito/Drp. Defaults to 0.
        progress ([type], optional): Link to a progress bar in the calling app. Defaults to None.
        app ([type], optional): Link to the calling app in order to keep it responsive using
        app.processEvents after each frame. Defaults to None.

    Returns:
        stack1 (numpy.array): stack of data depending on order
        stack2 (numpy.array): stack of data depending on order
        stackNN (numpy.array): stack of available network output, with zeros where no file was found
    """
    fileList = sorted(glob.glob(folder + '/img_*.tif'))
    numFrames = int(len(fileList)/2)
    pixelSize = io.imread(fileList[0]).shape
    stack1 = np.zeros((numFrames, pixelSize[0], pixelSize[1]))
    stack2 = np.zeros((numFrames, pixelSize[0], pixelSize[1]))
    postSize = round(pixelSize[1]*resizeParam)
    stackNN = np.zeros((numFrames,  postSize, postSize))
    frame = 0
    if progress is not None:
        progress.setRange(0, numFrames*2)
    # Check if there is data in this folder that has channels
    channelMode = False
    for file in fileList:
        if re.findall(r'.*channel001.*', file):
            channelMode = True
            print('Channel Mode')
            break

    for filePath in fileList:
        splitStr = re.split(r'img_channel\d+_position\d+_time', os.path.basename(filePath))
        splitStr = re.split(r'_z\d+', splitStr[1])
        frameNum = int(splitStr[0])

        if splitStr[1] == '_prep.tif':
            continue

        if channelMode:
            splitStr = re.split(r'img_channel', os.path.basename(filePath))
            splitStr = re.split(r'_position\d+_time\d+_z\d+', splitStr[1])
            channelNum = int(splitStr[0])
            if not channelNum % 2:
                stack1[frameNum] = io.imread(filePath)
                nnPath = filePath[:-8] + 'nn.tiff'
                try:
                    stackNN[frameNum] = io.imread(nnPath)
                except FileNotFoundError:
                    pass
            else:
                stack2[frameNum] = io.imread(filePath)
                frame = frame + 1
        else:
            if not frameNum % 2:
                # odd
                stack1[frame] = io.imread(filePath)
                nnPath = filePath[:-8] + 'nn.tiff'
                try:
                    stackNN[frame] = io.imread(nnPath)
                except FileNotFoundError:
                    pass
            else:
                stack2[frame] = io.imread(filePath)
                frame = frame + 1

        # Progress the bar if available
        # if progress is not None:
            # progress.setValue(frameNum)
    stack1 = np.array(stack1)
    stack2 = np.array(stack2)

    # Check if this data is rectangular and crop if necessary
    if cropSquare:
        stack1 = cropToSquare(stack1)
        stack2 = cropToSquare(stack2)

    if order == 0:
        return stack1, stack2, stackNN
    else:
        return stack2, stack1, stackNN


def cropToSquare(stack):
    """ Crop a rectangular stack to a square. Should also work for single images """
    dimensions = len(stack.shape)
    print(stack.shape)
    diffShape = stack.shape[dimensions - 2] - stack.shape[dimensions - 1]
    start = int(np.floor(abs(diffShape/2)))
    if diffShape > 0:
        print('Shape is not rectangular going to crop')
        end = int(np.floor(stack.shape[dimensions-2]-abs(diffShape/2)))
        if dimensions == 3:
            stack = stack[:, start:end, :]
        elif dimensions == 2:
            stack = stack[start:end, :]
        else:
            print('Image has wrong dimensions')
    elif diffShape < 0:
        print('Shape is not rectangular going to crop')
        end = int(np.floor(stack.shape[dimensions-1]-abs(diffShape/2)))
        if dimensions == 3:
            stack = stack[:, :, start:end]
        elif dimensions == 2:
            stack = stack[:, start:end]
        else:
            print('Image has wrong dimensions')

    return stack


def loadTifStack(stack, order=None, outputElapsed=False, cropSquare=True, img_range=None):
    """ Load a tif stack and deinterleave depending on the order (0 or 1). Also get the
    elapsed time on the images and give them back as list."""
    imageMitoOrig = io.imread(stack, key=img_range)

    print(imageMitoOrig.shape)
    with tifffile.TiffFile(stack, fastij=False) as tif:
        # try to get dataOrder from file
        if order is None:
            mdInfo = tif.ome_metadata  # pylint: disable=E1136  # pylint/issues/3139
            # This should work for single series ome.tifs
            mdInfoDict = xmltodict.parse(mdInfo)
            try:
                order = int(mdInfoDict['OME']['Image']['Description']['@dataOrder'])
                print('order parameter was read from file: ', order)
            except KeyError:
                print('dataOrder not found in file using ', order)

        start1 = order
        start2 = np.abs(order-1)
        if len(imageMitoOrig.shape) == 4:
            # This file probably has channels for the different data
            channels = imageMitoOrig.shape[1]
            stack1 = imageMitoOrig[:, start1, :, :]
            stack2 = imageMitoOrig[:, start2, :, :]
        else:
            channels = 1
            stack1 = imageMitoOrig[start1::2]
            stack2 = imageMitoOrig[start2::2]

        # This could be done also more flexible with a matrix or dict

        # get elapsed from tif file or just put the frames
        elapsed = []
        try:
            mdInfoDict = xmltodict.parse(tif.ome_metadata)
            for frame in range(0, imageMitoOrig.shape[0]*channels):
                # Check which unit the DeltaT is
                if mdInfoDict['OME']['Image']['Pixels']['Plane'][frame]['@DeltaTUnit'] == 's':
                    unitMultiplier = 1000
                else:
                    unitMultiplier = 1
                elapsed.append(unitMultiplier*float(
                    mdInfoDict['OME']['Image']['Pixels']['Plane'][frame]['@DeltaT']))
        except (TypeError, KeyError):
            print('No timing metadata just writing frames')
            elapsed = range(0, imageMitoOrig.shape[0]*channels)

    elapsed1 = elapsed[start1::2]
    elapsed2 = elapsed[start2::2]

    # Check if this data is rectangular and crop if necessary
    if cropSquare:
        stack1 = cropToSquare(stack1)
        stack2 = cropToSquare(stack2)

    return (stack1, stack2, elapsed1, elapsed2) if outputElapsed else (stack1, stack2)


def loadTifStackElapsed(file, numFrames=None, skipFrames=0):
    """ this should ideally be added to loadElapsed """
    elapsed = []
    with tifffile.TiffFile(file) as tif:
        if numFrames is None:
            numFrames = len(tif.pages)

        try:
            mdInfo = tif.ome_metadata  # pylint: disable=E1136  # pylint/issues/3139
            # This should work for single series ome.tifs
            mdInfoDict = xmltodict.parse(mdInfo)
            for frame in range(0, numFrames):
                # Check which unit the DeltaT is
                if mdInfoDict['OME']['Image']['Pixels']['Plane'][frame]['@DeltaTUnit'] == 's':
                    unitMultiplier = 1000
                else:
                    unitMultiplier = 1
                if (frame % (skipFrames+1)) != 0:
                    continue
                elapsed.append(unitMultiplier*float(
                    mdInfoDict['OME']['Image']['Pixels']['Plane'][frame]['@DeltaT']))
        except (TypeError, KeyError):
            elapsed = np.arange(0, numFrames)
    return elapsed


def savegif(stack, times, fps):
    """ Save a gif that uses the right frame duration read from the files. This can be sped up
    using the fps option"""
    filePath = 'C:/Users/stepp/Documents/02_Raw/SmartMito/presentation.gif'
    times = np.divide(times, fps).tolist()
    print(stack.shape)
    imageio.mimsave(filePath, stack, duration=times)


def extractTiffStack(file, frame, target):
    """ It should be possible to get the elapsed time also for a single image. Therefore especially
    when writing from ATSSim, I want to preserve both the OME metadata and the imagej metadata.
    Unfortunately tifffile can't write both at the same time. So the imagej metadata is transferred
    to the shaped_metadata. This also makes it easy to account for both possibilities when loading
    data elsewhere. """
    with tifffile.TiffFile(file, fastij=False) as tif:
        mdInfo = tif.ome_metadata
        ijInfo = tif.imagej_metadata

    image = tifffile.imread(file, pages=frame)
    mdInfoDict = xmltodict.parse(mdInfo)
    ijInfoDict = json.loads(ijInfo['Info'])  # pylint: disable=E1136  # pylint/issues/3139

    if mdInfoDict['OME']['Image']['Pixels']['Plane'][frame]['@DeltaTUnit'] == 's':
        unitMultiplier = 1000
    else:
        unitMultiplier = 1

    elapsed = mdInfoDict['OME']['Image']['Pixels']['Plane'][frame]['@DeltaT']*unitMultiplier

    ijInfoDict['ElapsedTime-ms'] = elapsed
    ijInfo['Info'] = json.dumps(ijInfoDict)  # pylint: disable=E1136,E1137  # pylint/issues/3139

    # Save the metadata for this page
    planeMD = mdInfoDict['OME']['Image']['Pixels']['Plane'][frame]
    # empty out the metadata for all the planes
    mdInfoDict['OME']['Image']['Pixels']['Plane'] = []
    # Set the old metadata as single object in a list
    mdInfoDict['OME']['Image']['Pixels']['Plane'] = [planeMD]

    # retranslate the dict to an xml string
    mdInfo = xmltodict.unparse(mdInfoDict)
    tifffile.imwrite(target, image, description=mdInfo.encode(encoding='UTF-8', errors='strict'),
                     metadata={'Info': ijInfo['Info']})  # pylint: disable=E1136


def calculateNNforStack(file, model=None, nnPath=None, img_range=None):
    """ calculate neural network output for all frames in a stack and write to new stack """
    # dataOrder = 1  # 0 for drp/foci first, 1 for mito/structure first
    if model is None:
        from tensorflow import keras
        modelPath = '//lebnas1.epfl.ch/microsc125/Watchdog/GUI/model_Dora.h5'
        modelPath = '//lebnas1.epfl.ch/microsc125/Watchdog/Model/paramSweep5/f32_c07_b08.h5'
        model = keras.models.load_model(modelPath, compile=True)
    elif isinstance(model, str):
        if os.path.isfile(model):
            from tensorflow import keras
            model = keras.models.load_model(model, compile=True)
        else:
            raise FileNotFoundError

    # drpOrig, mitoOrig, elapsed, _ = loadTifStack(file, dataOrder,  outputElapsed=True)
    if nnPath is None:
        nnPath = file[:-8] + '_nn.ome.tif'

    # Prepare the Metadata for the new file by extracting the ome-metadata
    with tifffile.TiffFile(file, fastij=False) as tif:
        try:
            mdInfo = tif.ome_metadata.encode(encoding='UTF-8', errors='strict')
            # extract only the planes that was written to
            mdInfoDict = xmltodict.parse(mdInfo)
            mdInfoDict['OME']['Image']['Pixels']['Plane'] = mdInfoDict['OME']['Image']['Pixels']['Plane'][0::2]

            dataOrder = int(mdInfoDict['OME']['Image']['Description']['@dataOrder'])
            mdInfo = xmltodict.unparse(mdInfoDict).encode(encoding='UTF-8', errors='strict')
        except (TypeError, AttributeError, KeyError):
            dataOrder = dataOrderMetadata(file, write=True)
            # Restart now that the mdInfo should be set
            calculateNNforStack(file, model=model, nnPath=nnPath, img_range=img_range)
            return
        print('Data order:', dataOrder)

    drpOrig, mitoOrig = loadTifStack(file, dataOrder, cropSquare=True, img_range=img_range,
                                     )

    # Get each frame and calculate the nn-output for it
    inputData, positions = NNfeeder.prepareNNImages(mitoOrig[0, :, :],
                                                    drpOrig[0, :, :], model)
    if model.layers[0].input_shape[0][1] is None:
        nnSize = inputData.shape[1]
    else:
        nnSize = positions['px'][-1][-1]

    nnImage = np.zeros((int(drpOrig.shape[0]), nnSize, nnSize))
    for frame in tqdm(range(drpOrig.shape[0])):
        inputData, positions = NNfeeder.prepareNNImages(mitoOrig[frame, :, :],
                                                        drpOrig[frame, :, :], model)
        outputPredict = model.predict_on_batch(inputData)
        if model.layers[0].input_shape[0][1] is None:
            nnImage[frame] = outputPredict[0, :, :, 0]
        else:
            nnImage[frame, :, :] = ImageTiles.stitchImage(outputPredict, positions)

    # Write the whole stack to a tif file with description
    tifffile.imwrite(nnPath, nnImage, description=mdInfo)


def calculateNNforFolder(folder, model=None):
    if model is None:
        from tensorflow import keras
        modelPath = '//lebnas1.epfl.ch/microsc125/Watchdog/GUI/model_Dora.h5'
        modelPath = '//lebnas1.epfl.ch/microsc125/Watchdog/Model/paramSweep5/f32_c07_b08.h5'
        model = keras.models.load_model(modelPath, compile=True)
    elif isinstance(model,str):
        if os.path.isfile(model):
            from tensorflow import keras
            model = keras.models.load_model(model, compile=True)
        else:
            raise FileNotFoundError
    if os.path.isfile(folder + '/img_channel001_position000_time000000000_z000.tif'):
        bact_filelist = sorted(glob.glob(folder + '/img_channel001*'))
        ftsz_filelist = sorted(glob.glob(folder + '/img_channel000*.tif'))
    else:
        print('No channels here')
        filelist = sorted(glob.glob(folder + '/img_*.tif'))
        re_odd = re.compile(r'.*time\d*[13579]_.*')
        bact_filelist = [file for file in filelist if re_odd.match(file)]
        re_even = re.compile(".*time\d*[02468]_.*")
        ftsz_filelist = [file for file in filelist if re_even.match(file)]

    for index, bact_file in tqdm(enumerate(bact_filelist)):
        ftsz_file = ftsz_filelist[index]
        nn_file = bact_file[:-8] + 'nn.tiff'
        if os.path.isfile(nn_file):
            continue
        else:
            bact = io.imread(bact_file)
            ftsz = io.imread(ftsz_file)
            inputData, positions = NNfeeder.prepareNNImages(bact, ftsz, model)
            outputPredict = model.predict_on_batch(inputData)
        if model.layers[0].input_shape[0][1] is None:
            nnImage = outputPredict[0, :, :, 0]
        else:
            nnImage = ImageTiles.stitchImage(outputPredict, positions)
        tifffile.imwrite(nn_file, nnImage)


def dataOrderMetadata(file, dataOrder=None, write=True):
    """ Add a Tag to the tif that states with dataOrder it has """
    reader = tifffile.TiffFile(file)
    writeMetadata = False
    mdInfo = None
    if dataOrder is None:
        try:
            mdInfo = xmltodict.parse(reader.ome_metadata)
            dataOrder = mdInfo['OME']['Image']['Description']['@dataOrder']
            if dataOrder == 'False':
                raise TypeError('data Order should not be False')
            print(file)
            print('dataOrder already there: ' + dataOrder)
        except (KeyError, TypeError) as _:
            if threading.current_thread() is threading.main_thread():
                plt.imshow(reader.pages[0].asarray())
                plt.show()
                answer = messagebox.askyesno(title='dataOrder', message='Is this a mito image?')
                dataOrder = 1 if answer else 0
            else:
                dataOrder = None
                print('Not in main thread, could not GUI detect dataOrder')
            if write:
                writeMetadata = True
        print(dataOrder)
    else:
        writeMetadata = True

    if writeMetadata:
        try:
            mdInfo = xmltodict.parse(reader.ome_metadata)
            mdInfo['OME']['Image']['Description']['@dataOrder'] = dataOrder
            print('dataOrder was already there overwritten')
        except (TypeError, KeyError) as _:
            try:
                mdInfo['OME']['Image']['Description'] = {'@dataOrder': dataOrder}
                print('Struct for dataOrder generated')
            except TypeError:
                try:
                    mdInfo['OME']['Image'] = {'Description': {'@dataOrder': dataOrder}}
                except TypeError:
                    mdInfo = {'OME': {'Image': {'Description': {'@dataOrder': dataOrder}}}}
                    print('No OME metadata in this file?')
        mdInfo = xmltodict.unparse(mdInfo).encode(encoding='UTF-8', errors='strict')
        tifffile.tifffile.tiffcomment(file, comment=mdInfo)
        reader = tifffile.TiffReader(file)
        print(xmltodict.parse(reader.ome_metadata)['OME']['Image']['Description'])

    return int(dataOrder) if dataOrder is not None else dataOrder


def cropOMETiff(file, outFile=None, cropFrame=None, cropRect=None):
    """ Crop a tif while conserving the metadata. Micromanager seems to save data in a rather weird
    format if it is supposed to save them to stacks. It limits the file size to ~4GB and splits the
    series up into several files _1, _2, _3 etc. The OME metadata is however written to the first
    file with TiffData tags that link to the other files for higher frames. This leads to a
    different behavior if loaded in imagej via drag or via bio-formats import. For a drag, only
    the frames that are in that actual file will be loaded, while for an import, all frames will be
    loaded also from the other files. This function makes one file that has all the frames with the
    correct metadata."""

    if isinstance(file, str):
        file = Path(file)
    if isinstance(outFile, str):
        outFile = Path(outFile)

    if cropFrame is None:
        cropFrame, fullLength = checkblackFrames(file.as_posix())
        cropFrame = cropFrame if cropFrame % 2 else cropFrame - 1
        cropFrame = False if cropFrame > fullLength - 5 else cropFrame
        print('crop at Frame:', str(cropFrame))

    if cropRect is True:
        # Ask for where the file should be cropped
        cropRect = defineCropRect(file.as_posix())

    if outFile is None:
        outFile = Path(file.as_posix()[:-8] + '_combine.ome.tif')

    bfconvert = 'set BF_MAX_MEM=12g & bfconvert -bigtiff -overwrite'
    compression = '-compression LZW'
    if not cropFrame:
        frames = ''
    else:
        frames = '-range 0 ' + str(cropFrame-1)

    if cropRect is None:
        rect = ''
    else:
        rect = '-crop ' + ','.join([str(i) for i in cropRect])

    files = file.as_posix() + ' ' + outFile.as_posix() + ' & timeout 15'
    command = ' '.join([bfconvert, compression, frames, rect, files])
    print(command)
    os.system(command)

    if cropFrame:
        print('adjusting metadata')
        with tifffile.TiffReader(outFile.as_posix()) as reader:
            mdInfo = xmltodict.parse(reader.ome_metadata)
            mdInfo['OME']['Image']['Pixels']['Plane'] = \
                mdInfo['OME']['Image']['Pixels']['Plane'][0:cropFrame]
            mdInfo['OME']['Image']['Pixels']['@SizeT'] = str(cropFrame)
            mdInfo = xmltodict.unparse(mdInfo).encode(encoding='UTF-8', errors='strict')
        tifffile.tifffile.tiffcomment(outFile.as_posix(), comment=mdInfo)
        print('transfered metadata')
    return outFile


def checkblackFrames(file):
    """ Test if there are black frames at the end of a tiff file """
    with open(os.devnull, "w") as fileHandle, contextlib.redirect_stderr(fileHandle):
        stack = io.imread(file)
    print(stack.shape)
    print(len(tifffile.TiffFile(file).pages))
    frame = stack.shape[0] - 1
    maxImage = 0
    while maxImage == 0:
        maxImage = np.max(stack[frame])
        frame = frame - 1
    print('first frame with data: ' + str(frame+1) + '\n\n')
    # if frame + 1 < stack.shape[0]:
    #    cropOMETiff(file, frame)
    return frame, stack.shape[0]


def defineCropRect(file):
    """ from a file get a rectangle that could be cropped to """
    def on_select(*_):
        pass

    with tifffile.TiffFile(file) as tif:
        image = tif.pages[0].asarray()
        plt.figure()
        axes = plt.axes()
        plt.imshow(image)
    rectprops = dict(facecolor='red', edgecolor='black', alpha=0.2, fill=False)
    rectHandle = RectangleSelector(axes, on_select, drawtype='box', useblit=False, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels', rectprops=rectprops,
                                   interactive=True, state_modifier_keys={'square': 'shift'})
    plt.show()
    rectProp = rectHandle._rect_bbox
    rectProp = tuple(int(x) for x in rectProp)
    print(rectProp)
    return rectProp


def main():
    """ Main method calculating a nn stack for a set of old Mito/drp stacks """
    folder = 'W:/Watchdog/bacteria/210506_weakSyncro/FOV_1'
    delay = loadiSIMmetadata(folder)
    print(delay)
    # folder = '//lebnas1.epfl.ch/microsc125/Watchdog/bacteria/210409_Caulobacter/FOV_1/Default'
    # loadRationalData(folder)

    # from tensorflow import keras
    # folder = 'W:/Watchdog/bacteria/210317_dualColor/FOV_3/Default'
    # model = keras.models.load_model('W:/Watchdog/Model/model_Dora.h5', compile=False)
    # makePrepImages(folder, model)
    # loadiSIMmetadata(folder)


    # allFiles = glob.glob('//lebnas1.epfl.ch/microsc125/iSIMstorage/Users/Willi/'
    #                      '180420_DRP_mito_Dora/**/*MMStack*lzw.ome.tif', recursive=True)
    # mainPath = 'W:/iSIMstorage/Users/Willi/160622_caulobacter/160622_CB15N_WT/SIM images'
    # mainPath = 'W:/iSIMstorage/Users/Willi/180420_drp_mito_Dora/**'
    # files = glob.glob(mainPath + '/*MMStack*_combine.ome.tif')

    # print('\n'.join(files))
    # files = [Path(file) for file in files]

    # for file in files:
    #     print(file)
    #     outFile = Path('W:/iSIMstorage/Users/Willi/180420_drp_mito_Dora'
    #                    + file.name[0:-8] + '_nn_ffbinary.ome.tif')
    #     # cropOMETiff(file, outFile=outFile, cropFrame=None, cropRect=True)
    #     dataOrderMetadata(file.as_posix())
    #     calculateNNforStack(file.as_posix(), nnPath=outFile)

    # for i in range(7, 10):
    #     file = 'W:/Watchdog/Model/paramSweep' + str(i) + '/prep_data' + str(i) + '.h5'
    #     print(file)
    #     resaveH5(file)


if __name__ == '__main__':
    main()
