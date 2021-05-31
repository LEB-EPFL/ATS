""" Take fixed framerate data and perform a simulation of ATS (adaptive temporal sampling) on
this data to see what performance would have been if ATS was used.

All the parameters are directly after the imports. As a default also the settings from the central
ATS_settings.json file can be used.

Created on Tue Jan  5 11:03:50 2021

@author: Willi Stepp
"""

import glob
import json
import os
import re
import shutil

import numpy as np
import xmltodict
from skimage import io
from tensorflow import keras

from NNfeeder import prepareNNImages
from NNio import dataOrderMetadata, loadTifStack


def main():
    """ Define folders here that should be processed"""
    # stacks = ('C:/Users/stepp/Documents/02_Raw/Caulobacter_Zenodo/'
    #           '160613_ML2159_MTS-mCherry_FtsZ-GFP_80/'
    #           'SIM images/Series5_copy.ome.tiff')
    # stacks = [stacks]

    # Comment if only taking one stack
    allFiles = glob.glob('//lebnas1.epfl.ch/microsc125/iSIMstorage/Users/Willi/'
                         '180420_drp_mito_Dora/**/*MMStack*_combine.ome.tif', recursive=True)

    # allFiles = glob.glob('W:/iSIMstorage/Users/Willi/160622_caulobacter/'
    #                      '160622_CB15N_WT/SIM images/Series*[0-9].ome.tif')

    # Just take the files that have not been analysed so far
    stacks = []
    for file in allFiles:
        # nnFile = file[:-8] + '_nn.ome.tif'
        sample = re.split('sample', file)[1][0]
        cell = re.split('cell_', file)[1][0]
        atsFolder = (os.path.dirname(file) + '/s' + str(sample).zfill(2) + '_c' +
            str(cell).zfill(2) + '_ATS2')

        # atsFolder = file[:-4] + '_ATS_ffmodel'
        # if not os.path.isfile(nnFile):
        if not os.path.isdir(atsFolder):
            stacks.append(file)
    stacks = sorted(stacks)
    print('\n'.join(stacks))
    stacks = ['C:/Users/stepp/Documents/02_Raw/SmartMito/sample1_cell_3.tif']
    atsOnStack(stacks)


def atsOnStack(stacks: list):
    """ Main function of the module that takes a list of files that should be tif stacks. The
    function then performs ATS Simulation on these files and writes outputdata to a folder in
    the same folder the file was in. This data can be displayed using NNGui."""
    #
    #                          USER SETTINGS
    #   see end of file for setting the folders/files to do the simulation on
    #
    modelPath = '//lebnas1.epfl.ch/microsc125/Watchdog/Model/paramSweep5/f32_c07_b08.h5'
    # Should we get the settings from the central settings file?
    extSettings = False
    dataOrder = 0  # 0 for drp/foci first, 1 for mito/structure first
    # data Order will be tried to read from the metadata directly in [Description][@dataOrder]

    if not extSettings:
        # if extSettings is set to False, you can set the settings to be used here
        thresholdLow = 0.8
        thresholdUp = 0.9
        slowRate = 5  # in seconds
        # The fast frame rate is the rate of the original file for now.
        minFastFrames = 4  # minimal number of fast frames after switching to fast. Should be > 2
        #
        #               USER SETTTINGS END
        #

        # Save these settings for later documentation
        settings = {'lowerThreshold': thresholdLow, 'upperThreshold': thresholdUp,
                    'slowRate': slowRate, 'minFastFrames': minFastFrames, 'dataOrder': dataOrder}
    else:
        # if extSettings is True, load the settings from the json file
        with open('./ATS_settings.json') as file:
            settings = json.load(file)
            thresholdLow = settings['lowerThreshold']
            thresholdUp = settings['upperThreshold']
            slowRate = settings['slowRate']
            minFastFrames = settings['minFastFrames']
            settings['dataOrder'] = dataOrder

    model = keras.models.load_model(modelPath, compile=True)
    print('model loaded')

    for stack in stacks:
        print(stack)
        # Make a new folder to put the output in
        # newFolder = re.split(r'.tif', stack)[0] + '_ATS_ffmodel'
        sample = re.split('sample', stack)[1][0]
        cell = re.split('cell_', stack)[1][0]
        newFolder = (os.path.dirname(stack) + '/s' + str(sample).zfill(2) + '_c' +
                     str(cell).zfill(2) + '_ATS2')
        if os.path.exists(newFolder):
            shutil.rmtree(newFolder)
        os.mkdir(newFolder)

        # Save the settings that were used for this run
        settingsFile = os.path.join(newFolder, 'ATSSim_settings.json')
        with open(settingsFile, 'w') as fileHandle:
            jsonString = json.dumps(settings, default=lambda o: o.__dict__,
                                    sort_keys=True, indent=2)
            fileHandle.write(jsonString)

        # Make the file that simulates the NetworkWatchdog txt file output
        txtFile = os.path.join(newFolder, 'output.txt')
        file = open(txtFile, 'w')
        file.close()
        file = open(txtFile, 'a')

        # dataOrder = dataOrderMetadata(stack)
        dataOrder = dataOrderMetadata(stack, write=False)
        drpOrig, mitoOrig, drpTimes, mitoTimes = loadTifStack(stack, order=dataOrder,
                                                              outputElapsed=True)
        print('stack loaded')

        # check if there is timing in the times or just frames and if adjust for timing in ms
        diff = np.diff(drpTimes)
        if np.max(diff) < 3:
            drpTimes = np.multiply(drpTimes, 500)
            mitoTimes = np.multiply(mitoTimes, 500)

        time = drpTimes[0]
        frame = 0
        fastMode = False
        outputFrame = 0
        delay = slowRate
        fastCount = 0
        fastCum = 0

        outputData = []
        print(drpOrig.shape[0])
        while frame < drpOrig.shape[0]-1:
            # make tiles for current frame
            inputData, positions = prepareNNImages(mitoOrig[frame, :, :],
                                                   drpOrig[frame, :, :], model)

            # Run neural network
            outputPredict = model.predict_on_batch(inputData)
            inputSize = round(drpOrig.shape[1]*56/81)

            if model.layers[0].input_shape[0][1] is None:
                # If the network is for full shape images, be sure that shape is multiple of 4
                inputSize = inputSize - inputSize % 4
                npType = np.float
            else:
                npType = np.uint8

            outputDataFull = np.zeros([inputSize, inputSize])
            mitoDataFull = np.zeros([inputSize, inputSize])
            drpDataFull = np.zeros([inputSize, inputSize])

            if model.layers[0].input_shape[0][1] is None:
                outputDataFull = outputPredict[0, :, :, 0]
                mitoDataFull = inputData[0, :, :, 0, 0]
                drpDataFull = inputData[0, :, :, 1, 0]
            else:
                # Stitch the tiles back together
                i = 0
                stitch = positions['stitch']
                stitch1 = None if stitch == 0 else -stitch
                for position in positions['px']:
                    outputDataFull[position[0]+stitch:position[2]-stitch,
                                position[1]+stitch:position[3]-stitch] = \
                        outputPredict[i, stitch:stitch1, stitch:stitch1, 0]
                    mitoDataFull[position[0]+stitch:position[2]-stitch,
                                position[1]+stitch:position[3]-stitch] = \
                        inputData[i, stitch:stitch1, stitch:stitch1, 0]
                    drpDataFull[position[0]+stitch:position[2]-stitch,
                                position[1]+stitch:position[3]-stitch] = \
                        inputData[i, stitch:stitch1, stitch:stitch1, 1]
                    i = i + 1

            # Get the output data from the nn channel
            outputData.append(np.max(outputDataFull))

            # ConstructMetadata
            mitoMeta = {'OME': {'Image': {'Pixels': {'Plane': [{'@DeltaT': mitoTimes[frame]}]}}}}
            mitoMeta['OME']['Image']['Pixels']['Plane'][0]['@DeltaTUnit'] = 'ms'
            mitoMeta['OME']['Image']['Description'] = {'@dataOrder': dataOrder}
            mitoMeta = xmltodict.unparse(mitoMeta).encode(encoding='UTF-8', errors='strict')
            drpMeta = {'OME': {'Image': {'Pixels': {'Plane': [{'@DeltaT': drpTimes[frame]}]}}}}
            drpMeta['OME']['Image']['Pixels']['Plane'][0]['@DeltaTUnit'] = 'ms'
            drpMeta['OME']['Image']['Description'] = {'@dataOrder': dataOrder}
            drpMeta = xmltodict.unparse(drpMeta).encode(encoding='UTF-8', errors='strict')

            mitoPath = (newFolder + '/img_channel000_position000_time' +
                        str((outputFrame*2 + 1)).zfill(9) + '_z000.tif')
            mitoPrepPath = (newFolder + '/img_channel000_position000_time' +
                            str((outputFrame*2 + 1)).zfill(9) + '_z000_prep.tif')
            drpPath = (newFolder + '/img_channel000_position000_time' +
                       str((outputFrame*2)).zfill(9) + '_z000.tif')
            drpPrepPath = (newFolder + '/img_channel000_position000_time' +
                           str((outputFrame*2)).zfill(9) + '_z000_prep.tif')
            nnPath = (newFolder + '/img_channel000_position000_time' +
                      str((outputFrame*2 + 1)).zfill(9) + '_nn.tiff')
            io.imsave(mitoPath, mitoOrig[frame, :, :].astype(np.uint16),
                      check_contrast=False, imagej=False,
                      description=mitoMeta)
                      # {'Info': json.dumps({'ElapsedTime-ms': mitoTimes[frame]})})
            io.imsave(mitoPrepPath, mitoDataFull.astype(npType),
                      check_contrast=False, imagej=False,
                      description=mitoMeta)
            io.imsave(drpPath, drpOrig[frame, :, :].astype(np.uint16),
                      check_contrast=False, imagej=False,
                      description=drpMeta)
            io.imsave(drpPrepPath, drpDataFull.astype(npType),
                      check_contrast=False, imagej=False,
                      description=drpMeta)
            io.imsave(nnPath, outputDataFull.astype(npType), check_contrast=False)

            # Write to output.txt file as networkWatchdog would
            file.write('%d, %.3f\n' % (outputFrame*2 + 1, outputData[-1]))

            # if outputData[-1] > thresholdUp:
            #     delay = 0
            # elif outputData[-1] < thresholdLow and fastMode:
            #     delay = slowRate

            # Decide in which mode to go on, do at least minFastFrames fast frames after high output
            print('                                                                              ', end='\r')
            if outputFrame > 0:
                if fastMode:
                    fastCount = fastCount + 1
                    fastCum = fastCum + 1
                if outputData[-2] > thresholdUp:
                    print('FAST mode', end=' ')
                    fastMode = True
                    fastCount = 0
                elif outputData[-2] < thresholdLow and fastMode and fastCount > minFastFrames - 1:
                    print('SLOW mode', end=' ')
                    fastMode = False
                elif outputData[-2] > thresholdLow and fastMode:
                    print('fast mode', end=' ')
                else:
                    print('slow mode', end=' ')

            if fastMode:
                delay = 0
            else:
                delay = slowRate

            # Write iSIMmetadata File as Matlab would
            if outputFrame > 0:
                iSIMname = ('iSIMmetadata_Timing_' + str(int(time)).zfill(9) + '.txt')
                iSIMMetadataFile = os.path.join(newFolder, iSIMname)
                metaDataFile = open(iSIMMetadataFile, 'w')
                metaDataFile.write('%d\t%.3f\t%d' % (0, delay, 1))
                metaDataFile.close()

            print(str(frame), end=':  ')

            # Decide which frame to take next
            if fastMode:
                frame = frame + 1
                time = drpTimes[frame]
            else:
                # jump slowRate seconds and find which frame would be closest to this
                time = time + slowRate*1000
                oldFrame = frame
                frame = min(range(len(drpTimes)), key=lambda i: abs(drpTimes[i]-time))
                # check if it will jump at least one frame if not, jump one
                if oldFrame == frame:
                    frame = frame + 1
                if frame - oldFrame == 1:
                    print('Jumped one frame. slowRate might not be big enough.')

                time = drpTimes[frame]

            outputFrame = outputFrame + 1
            # print(time)
            print(np.round(outputData[-1], 3), end='')
            print(('  -> ' + str(frame)), end=' ')
            print(('fast frames:' + str(fastCount)), end='\r')

            if outputData[-1] == 0:
                fileHandle = open("ATSSim_logging.txt", "a")
                fileHandle.write('\n outputData was 0 at frame')
                fileHandle.write(str(frame*2))
                fileHandle.write('\n' + stack + '\n' + '\n')
                fileHandle.close()

        file.close()
        print('\ntotal fast frames: ', fastCum, '\n\n')


if __name__ == '__main__':
    main()
