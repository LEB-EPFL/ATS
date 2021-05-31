# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:32:38 2020

@author: stepp

Watchdog application for a tif file on the network that reads the image and
updates it in the associated figure window. Will be modified to make visual
output optional and adding a neural network that is used to mark interesting
events in the image.

Used in an environment where microManager running on another machine saves an
image to a NAS location, that should be observed, read and analysed. The binary
output will go to the same NAS and is expected to be read by a Matlab program
that runs on the same machine as microManager.
"""


import json
import os
import re  # Regular expression module
import sys
import time
from datetime import datetime

import numpy as np
import pyqtgraph as pg
# Qt display imports
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication, QGridLayout, QWidget
from skimage import io
# Tensorflow
from tensorflow import keras
from watchdog.events import PatternMatchingEventHandler
# Watchdog
from watchdog.observers import Observer

from BinOutput import writeBin
# from imageTiles import getTilePositions_v2
# Own modules
from NNfeeder import prepareNNImages
from QtImageViewerMerge import QtImageViewerMerge


class NetworkWatchdog(QWidget):
    """Application used for the Feedback loop in the adaptive temporal sampling approach on the
    iSIM. Uses 'Watchdogs' to monitor a network drive that images are written to from the iSIM using
    the standard microManager save as images utility. Runs the neural network and writes the
    generated output back to the network location for Matlab to read it. Works together with
    FrameNumWrite in order to skip frames for keeping up when imaging fast. NNGui can be used for
    visualizing the data generated.

    Args:
        QWidget ([type]): [description]
    """
    frameChanged = pyqtSignal([], [int])
    refreshGUI = pyqtSignal([], [int])
    reinitGUI = pyqtSignal(int)

    def __init__(self):
        # Set if channels are used in Micromanager
        self.channels = True
        self.bacteria = True

        # Read settings from the json file depending on which computer we are on
        with open('./ATS_settings.json') as file:
            settings = json.load(file)

        self.settings = settings[os.environ['COMPUTERNAME'].lower()]
        self.ATSsettings = settings

        # Setting for the watchdogs
        patterns = ["*.tif"]
        ignorePatterns = ["*.txt", "*.tiff"]
        ignoreDirectories = True
        caseSensitive = True
        myEventHandler = PatternMatchingEventHandler(
            patterns, ignorePatterns, ignoreDirectories, caseSensitive)

        # Loading the NN model
        self.nnImageSize = 128
        self.modelPath = self.settings['modelPath']
        self.model = keras.models.load_model(self.modelPath, compile=True)
        # Prep some structure to hold data

        # Set iSIM specific values
        pixelCalib = 56  # nm per pixel
        self.sig = 121.5/81  # in pixel
        self.resizeParam = pixelCalib/81  # no unit
        # Preprocess the images

        # Figure for the outputs
        QWidget.__init__(self)
        self.viewerMerge = QtImageViewerMerge()
        self.imageDrp = self.viewerMerge.addImage()
        self.imageMito = self.viewerMerge.addImage()
        self.viewerMerge.setLUT(self.imageDrp, 'reds')
        self.viewerMerge.setLUT(self.imageMito, 'grey')
        self.viewerNN = QtImageViewerMerge()
        self.imageNN = self.viewerNN.addImage()
        self.viewerNN.setLUT(self.imageNN, 'inferno')

        self.viewerOutput = pg.PlotWidget()
        self.outputPlot = self.viewerOutput.plot()
        pen = pg.mkPen(color='#FF0000', style=Qt.DashLine)
        self.thrLine1 = pg.InfiniteLine(
            pos=self.ATSsettings['upperThreshold'], angle=0, pen=pen)
        self.thrLine2 = pg.InfiniteLine(
            pos=self.ATSsettings['lowerThreshold'], angle=0, pen=pen)
        self.viewerOutput.addItem(self.thrLine1)
        self.viewerOutput.addItem(self.thrLine2)

        grid = QGridLayout(self)
        grid.addWidget(self.viewerMerge, 1, 0)
        grid.addWidget(self.viewerNN, 1, 1)
        grid.addWidget(self.viewerOutput, 0, 0, 1, 2)
        self.linePen = pg.mkPen(color='#AAAAAA')

        # Assign the event handlers
        myEventHandler.on_created = self.onCreated
        myEventHandler.on_deleted = self.onDeleted
        myEventHandler.on_modified = self.onModified
        myEventHandler.on_moved = self.onMoved
        self.refreshGUI.connect(self.refreshViewBoxes)
        self.reinitGUI.connect(self.reinitializeViewBoxes)
        self.viewerNN.viewBox.setXLink(self.viewerMerge.viewBox)
        self.viewerNN.viewBox.setYLink(self.viewerMerge.viewBox)

        # More settings for the Watchdos
        path = self.settings['imageFolder']
        print('Folder:', path)
        goRecursively = True
        self.myObserver = Observer()
        self.myObserver.schedule(
            myEventHandler, path, recursive=goRecursively)

        # Init the model by running it once
        print('Initialize the model')
        self.model(np.random.randint(
            10, size=[1, self.nnImageSize, self.nnImageSize, 2]))

        self.myObserver.start()

        print('All loaded, running...')
        print('Channel Mode:', self.channels)

        # Init variables
        self.mitoDataFull = None
        self.drpDataFull = None
        self.outputHistogram = None
        self.outputX = None
        self.maxPos = None

        self.frameNumOld = 100
        self.inputSizeOld = 0
        self.folderNameOld = 'noFolder'
        self.outputDataFull = np.zeros([512, 512])
        self.imageViewer = [self.viewerMerge, self.viewerNN]
        for viewer in self.imageViewer:
            viewer.lines = []

    def onCreated(self, event):
        """Do action when file is created in watchlocation"""

    def onDeleted(self, event):
        """Do action when file is deleted in watchlocation"""

    def onModified(self, event):
        """Run the Feedback loop when a new image is written by microManager to the network
        location to be watched. Writes the calculated output value to a binary file that
        Matlab on the iSIM can read to decide on the next framerate to use.

        Args:
            event ([type]): [description]
        """
        size = os.path.getsize(event.src_path)

        # Extract the frame number from the filename
        splitStr = re.split(r'img_channel\d+_position\d+_time',
                            os.path.basename(event.src_path))
        splitStr = re.split(r'_z\d+', splitStr[1])
        frameNum = int(splitStr[0])

        # Extract channel number if mode is channels
        if self.channels:
            splitStr = re.split(r'img_channel',
                                os.path.basename(event.src_path))
            splitStr = re.split(r'_position\d+_time\d+_z\d+', splitStr[1])
            channelNum = int(splitStr[0])

        # Only go on if frame number is odd and this frame was not analysed yet
        if self.channels:
            if channelNum == 0 or frameNum == self.frameNumOld:
                return
        else:
            if not frameNum % 2 or frameNum == self.frameNumOld:
                return

        # check framNumwrite.py output from the binary
        if not self.channels:
            file = open(os.path.join(self.settings['frameNumBinary']), mode='rb')
            content = file.read()
            file.close()

            # Skip file if a newer one is already in the folder
            if not frameNum >= len(content)-1:
                print(int((frameNum-1)/2), ' passed because newer file is there')
                return

        # Construct paths
        # DO THIS IN A NICER WAY! not required to have that exact format
        if self.channels:
            time.sleep(0.1)
            mitoFile = 'img_channel000_position000_time' \
                       + str((frameNum)).zfill(9) + '_z000.tif'
        else:
            mitoFile = 'img_channel000_position000_time' \
                + str((frameNum-1)).zfill(9) + '_z000.tif'
        nnFile = 'img_channel000_position000_time' \
            + str((frameNum)).zfill(9) + '_nn.tiff'
        mitoPath = os.path.join(os.path.dirname(event.src_path), mitoFile)
        nnPath = os.path.join(os.path.dirname(event.src_path), nnFile)
        txtFile = os.path.join(os.path.dirname(event.src_path), 'output.txt')

        # Mito is already written, so check size
        mitoSize = os.path.getsize(mitoPath)

        if not size > mitoSize*0.95:
            return

        if self.channels:
            print('frame', frameNum)
        else:
            print('frame', int((frameNum-1)/2))
        # Read the mito image first, as it should already be written
        # mitoFull = io.imread(mitoPath)
        # drpFull = io.imread(event.src_path)
        # switched for Drp > green mito > red
        print(event.src_path)
        mitoFull = io.imread(event.src_path)
        drpFull = io.imread(mitoPath)
        print(mitoFull.shape)

        # If this is the first frame, reinitialize the plot
        inputSize = mitoFull.shape[0]
        folderName = os.path.dirname(event.src_path)
        if not folderName == self.folderNameOld and not inputSize == self.inputSizeOld:
            inputSize = round(drpFull.shape[0]*self.resizeParam) if not \
                inputSize == 128 else 128
            self.outputDataFull = np.zeros([inputSize, inputSize])
            self.mitoDataFull = np.zeros([inputSize, inputSize])
            self.drpDataFull = np.zeros([inputSize, inputSize])
            # Redraw the lines
            self.reinitGUI.emit(inputSize)
            self.viewerNN.cross.show()
            print('Reinitialized plot')
        if not folderName == self.folderNameOld:
            # Make the txt file to write the output data to
            print(folderName)
            print('New txt file written')
            open(txtFile, 'w+')
            self.outputHistogram = []
            self.outputX = []
            # Save the settings used for this run (TODO)

        # Preprocess the data and make tiles if necessary
        inputData, positions = prepareNNImages(
            mitoFull, drpFull, self.model, self.bacteria)
        # print(inputData.shape)
        # Calculate the prediction on the full batch of images
        outputPredict = self.model.predict_on_batch(inputData)

        # Stitch the tiles back together (~2ms 512x512)
        i = 0
        stitch = positions['stitch']
        stitch1 = None if stitch == 0 else -stitch
        for position in positions['px']:
            self.outputDataFull[position[0]+stitch:position[2]-stitch,
                                position[1]+stitch:position[3]-stitch] = \
                outputPredict[i, stitch:stitch1, stitch:stitch1, 0]
            self.mitoDataFull[position[0]+stitch:position[2]-stitch,
                              position[1]+stitch:position[3]-stitch] = \
                inputData[i, stitch:stitch1, stitch:stitch1, 0]
            self.drpDataFull[position[0]+stitch:position[2]-stitch,
                             position[1]+stitch:position[3]-stitch] = \
                inputData[i, stitch:stitch1, stitch:stitch1, 1]
            i = i + 1

        # OUTPUT Calculation
        approach = 3
        if approach == 1:
            mask = self.outputDataFull > 10
            outputDataThresh = np.zeros_like(mask).astype(int)
            outputDataThresh[mask] = self.outputDataFull[mask]
            output = int(round(np.sum(outputDataThresh)))
        elif approach == 2:
            numPixels = 4
            output = 0
            for i in range(0, numPixels):
                output = output + np.max(self.outputDataFull)
                maxX = np.argmax(self.outputDataFull, axis=0)
                maxY = np.argmax(self.outputDataFull, axis=1)
                self.outputDataFull[maxX, maxY] = 0
            outputDataThresh = self.outputDataFull
            output = round(output)
        elif approach == 3:
            output = int(round(np.max(self.outputDataFull)))
            outputDataThresh = self.outputDataFull
        # Define where the position was found

        self.maxPos = list(zip(*np.where(self.outputDataFull == np.max(self.outputDataFull))))

        nowStr = datetime.now()
        hours = nowStr.strftime("%H")
        minutes = nowStr.strftime("%M")
        seconds = nowStr.strftime("%S")
        millis = nowStr.strftime("%f")
        timeX = (int(hours)*60*60*1000 + int(minutes)*60*1000 +
                 int(seconds)*1000 + int(millis[:3]))/60/60/1000

        lengthCache = 100
        if len(self.outputHistogram) > lengthCache:
            self.outputX = self.outputX[1:]
            self.outputHistogram = self.outputHistogram[1:]
        self.outputHistogram.append(output)
        self.outputX.append(timeX)  # (frameNum-1)/2)

        # Write output to binary for Matlab to read
        writeBin(output, printTime=0, path=self.settings['matlabPath'], filename='')
        # Write output to txt file for later
        file = open(txtFile, 'a')
        file.write('%d, %d\n' % (frameNum, output))
        file.close()
        # Save the nn image
        io.imsave(nnPath, self.outputDataFull.astype(np.uint8), check_contrast=False)

        # Prepare images for plot and emit plotting event
        # self.mitoDisp = gray2qimage(self.mitoDataFull, normalize=True)
        # self.mitoDisp.setColorTable(self.lutHot)
        # self.drpDisp = gray2qimage(self.drpDataFull, normalize=True)
        # self.nnDisp = gray2qimage(self.outputDataFull, normalize=(0, 50))
        # np.mean(self.outputHistogram)*2 instead of 50?!
        # self.nnDisp.setColorTable(getQtcolormap('inferno'))
        self.refreshGUI.emit()

        # Prepare for next cycle
        self.frameNumOld = frameNum
        self.inputSizeOld = inputSize
        self.folderNameOld = folderName
        print('output generated   ', int(output), '\n')

    def refreshViewBoxes(self):
        """ Update the merged vies and the output plot when a calculation has finished. """
        self.imageMito.setImage(self.mitoDataFull)
        self.imageDrp.setImage(self.drpDataFull)
        self.imageNN.setImage(self.outputDataFull)
        self.outputPlot.setData(self.outputX, self.outputHistogram)
        self.viewerMerge.cross.setPosition(self.maxPos)
        self.viewerNN.cross.setPosition(self.maxPos)
        self.viewerOutput.enableAutoRange()

    def reinitializeViewBoxes(self, inputSize):
        """ reinitialize the GUI if data with a new size is processed. This also used to write the
        grid lines that correspond to the tiles used to feed the neural network. """
        # This was used to draw a grid on the ViewBox that shows the tiling. Will have to redo for
        # the new plotting environment
        # positions = getTilePositions_v2(
        #         np.ones((inputSize, inputSize)), self.nnImageSize)

        # for viewer in self.imageViewer:
        #     for line in viewer.lines:
        #         viewer.scene.removeItem(line)
        #     viewer.lines = []

        #     for line in viewer.lines:
        #         line.setPen(self.linePen)
        #         line.setZValue(100)
        self.viewerMerge.viewBox.setRange(xRange=(0, inputSize), yRange=(0, inputSize))

    def onMoved(self, event):
        """ start when a file was used in the watchlocation. Could be removed(?) """

    def closeEvent(self, _):
        """ Terminate the watchdogs and clean up when the windows of the GUI is closed """
        self.myObserver.stop()
        self.myObserver.join()
        print('Watchdogs stopped')


def main():
    """Launching the main GUI for SATS mode in smartiSIM"""
    app = QApplication(sys.argv)
    watchdog = NetworkWatchdog()

    watchdog.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
