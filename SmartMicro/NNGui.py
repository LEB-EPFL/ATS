# -*- coding: utf-8 -*-
"""
This is a GUI that runs a neural network on the input data and displays
original data and the result in a scrollable form.
v1 was made using matplotlib. Unfortunately that is very slow, so here
I try to use pyqtgraph for faster performance
Sources:
https://stackoverflow.com/questions/41526832/pyqt5-qthread-signal-not-working-gui-freeze

Created on Mon Oct  5 12:18:48 2020

@author: stepp
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pyqtgraph as pg
import tifffile
from PyQt5.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (QApplication, QFileDialog, QGridLayout, QGroupBox,
                             QLabel, QPlainTextEdit, QProgressBar, QPushButton,
                             QSlider, QWidget)
from skimage import exposure, transform
from tensorflow import keras

from NNfeeder import prepareNNImages
from NNio import (dataOrderMetadata, loadTifFolder, loadTifStack,
                  loadTifStackElapsed)
from QtImageViewerMerge import QtImageViewerMerge
from SatsGUI import SatsGUI

# Adjust for different screen sizes
QApplication.setAttribute(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)


class NNGui(QWidget):
    """Used to visualize and examine data taken using an adaptive temporal sampling approach.
    Used for Mito/Drp and Bacteria/FtsZ and a neural network tought to detect division events
    in mitochondria (see Mito2Drp1.py). Uses QtImageViewerMerge to visualize the channels and
    SATS_GUI to visualize the temporal sampling of the data.
    Loads stacks of interleaved images of the two channels or folders that were populated using
    the ATS/smartiSIM approach in combination with network_Watchdog.py
    Originally based on TifStackViewer

    Args:
        QWidget ([type]): [description]
    """

    frameChanged = pyqtSignal([], [int])

    def __init__(self, app):
        QWidget.__init__(self)

        # Handle to the image stack tiffcapture object.
        self._tiffCaptureHandle = None
        self.currentFrameIndex = None
        self.testfile = ('C:/Users/stepp/Documents/02_Raw/SmartMito/'
                         'sample1_cell_3_MMStack_Pos0_2_crop_lzw.ome_ATS/'
                         'img_channel000_position000_time000000000_z000.tif')

        # Image frame viewer.
        self.viewerOrig = QtImageViewerMerge()
        self.imageItemDrpOrig = self.viewerOrig.addImage()
        self.imageItemMitoOrig = self.viewerOrig.addImage()
        self.viewerOrig.setLUT(self.imageItemDrpOrig, 'reds')
        self.viewerOrig.setLUT(self.imageItemMitoOrig, 'grey')

        self.viewerProc = QtImageViewerMerge()
        self.imageItemDrpProc = self.viewerProc.addImage()
        self.imageItemMitoProc = self.viewerProc.addImage()
        self.viewerProc.setLUT(self.imageItemDrpProc, 'reds')
        self.viewerProc.setLUT(self.imageItemMitoProc, 'grey')

        self.viewerNN = QtImageViewerMerge()
        self.imageItemNN = self.viewerNN.addImage()
        self.viewerNN.setLUT(self.imageItemNN, 'inferno')
        self.loadBox = QGroupBox()

        # Connect the viewers
        self.viewerOrig.viewBox.setXLink(self.viewerProc.viewBox)
        self.viewerOrig.viewBox.setYLink(self.viewerProc.viewBox)
        self.viewerNN.viewBox.setXLink(self.viewerProc.viewBox)
        self.viewerNN.viewBox.setYLink(self.viewerProc.viewBox)

        # Slider and arrow buttons for frame traversal.
        self.sliderBox = QGroupBox()
        self.frameSlider = QSlider(Qt.Horizontal)
        self.frameSlider.setMinimumHeight(20)
        self.frameSlider.setValue(0)
        self.frameSlider.setDisabled(True)
        self.frameSlider.wheelEvent = self.sliderWheelEvent
        self.prevFrameButton = QPushButton("<")
        self.nextFrameButton = QPushButton(">")

        # loadBox content: Buttons for load model and data
        self.modelButton = QPushButton("load model")
        self.dataButton = QPushButton("load data")
        self.dataButton.setDisabled(True)
        self.orderButton = QPushButton("order: Drp first")
        self.hideOrig = QPushButton('Orig')
        self.hideOrig.setCheckable(True)
        self.setVirtual = QPushButton('virtual Stack')
        self.setVirtual.setCheckable(True)
        self.hideNN = QPushButton('NN')
        self.hideNN.setCheckable(True)
        self.currentFrameLabel = QLabel('Frame')
        self.loadingStatusLabel = QLabel('')
        # progress bar for loading data
        self.progress = QProgressBar(self)
        self.log = QPlainTextEdit(self)

        self.outputPlot = SatsGUI()

        pen = pg.mkPen(color='#AAAAAA', style=Qt.DashLine)
        self.frameLine = pg.InfiniteLine(pos=0.5, angle=90, pen=pen)
        self.outputPlot.plotItem1.addItem(self.frameLine)

        # Connect functions to the interactive elements
        self.modelButton.clicked.connect(self.loadModel)
        self.dataButton.clicked.connect(self.loadDataThread)
        self.orderButton.clicked.connect(self.orderChange)
        self.hideOrig.clicked.connect(self.hideViewerOrig)
        self.setVirtual.clicked.connect(self.setVirtualCallback)
        self.hideNN.clicked.connect(self.hideViewerNN)
        self.prevFrameButton.clicked.connect(self.prevFrame)
        self.nextFrameButton.clicked.connect(self.nextFrame)

        # Layout.
        grid = QGridLayout(self)
        grid.addWidget(self.viewerOrig, 0, 0)
        grid.addWidget(self.viewerProc, 0, 1)
        grid.addWidget(self.viewerNN, 0, 2)
        grid.addWidget(self.outputPlot, 1, 0, 1, 2)
        grid.addWidget(self.loadBox, 1, 2)
        grid.addWidget(self.sliderBox, 2, 0, 1, 3)

        gridprogress = QGridLayout(self.sliderBox)
        gridprogress.addWidget(self.prevFrameButton, 0, 0)
        gridprogress.addWidget(self.frameSlider, 0, 1)
        gridprogress.addWidget(self.nextFrameButton, 0, 3)
        gridprogress.setContentsMargins(0, 0, 0, 0)

        gridBox = QGridLayout(self.loadBox)
        gridBox.addWidget(self.modelButton, 0, 0)
        gridBox.addWidget(self.dataButton, 0, 1)
        gridBox.addWidget(self.orderButton, 0, 2)
        gridBox.addWidget(self.hideOrig, 1, 0)
        gridBox.addWidget(self.setVirtual, 1, 1)
        gridBox.addWidget(self.hideNN, 1, 2)
        gridBox.addWidget(self.loadingStatusLabel, 2, 0, 1, 3)
        gridBox.addWidget(self.progress, 3, 0, 1, 3)
        gridBox.addWidget(self.currentFrameLabel, 4, 0)
        gridBox.addWidget(self.log, 5, 0, 1, 3)

        self.threads = []

        self.timer = QTimer()
        self.timer.timeout.connect(self.onTimer)
        self.timer.setInterval(20)

        self.frameSlider.sliderPressed.connect(self.startTimer)
        self.frameSlider.sliderReleased.connect(self.stopTimer)
        self.frameSlider.valueChanged.connect(self.onSlider)
        # init variables
        self.fileList = [None]*3
        self.mode = None
        self.model = None
        self.imageDrpOrig = None
        self.imageMitoOrig = None
        self.nnOutput = None
        self.mitoDataFull = None
        self.drpDataFull = None
        self.maxPos = None
        self.maxPosRational = None
        self.nnRecalculated = None
        self.settings = None
        self.folder = 'C:/Users/stepp/Documents/02_Raw/SmartMito'
        self.virtualFolder = None
        self.virtualStack = False

        self.app = app
        self.order = 1
        self.linePen = pg.mkPen(color='#AAAAAA')

    def setProgressValue(self, progress):
        """ Set the progress value from the loading thread """
        self.progress.setValue(progress)

    def setLabelString(self, message):
        """ Set the Satus Label when called from loading thread """
        self.loadingStatusLabel.setText(message)

    def setLogString(self, message):
        """ Append to the log signaled from loading thread"""
        self.log.appendPlainText(message)

    def receiveData(self, data):
        """ Receive the data from the loading worker """
        self.frameSlider.setDisabled(True)
        self.setLogString('copy the loaded data to GUI')
        self.mode = data.mode
        self.imageMitoOrig = data.imageMitoOrig
        self.imageDrpOrig = data.imageDrpOrig
        self.mitoDataFull = data.mitoDataFull
        self.drpDataFull = data.drpDataFull
        self.folder = data.startFolder
        self.virtualFolder = data.folder
        self.nnOutput = data.nnOutput
        self.maxPos = data.maxPos
        self.maxPosRational = data.maxPosRational
        self.frameSlider.setMaximum(data.frameNum - 1)
        # self.refreshGradients()
        self.loadingStatusLabel.setText('Done')
        self.viewerProc.viewBox.setRange(xRange=(0, data.postSize), yRange=(0, data.postSize))

        # Make the SATS_GUI plot for nn_output vs time
        self.outputPlot.resetPlot()
        if self.mode == 'stack':
            self.outputPlot.nnline.setData(data.elapsed[0::2], data.outputData)
            self.outputPlot.scatter.setData(data.elapsed[0::2], data.outputData)
            self.outputPlot.elapsed = data.elapsed[0::2]
            self.outputPlot.delay = [data.elapsed[2] - data.elapsed[0]]*len(data.elapsed[0::2])
            nnPrep = np.stack((list(np.arange(0, len(data.outputData))), data.outputData), 0)
            self.outputPlot.nnData = nnPrep.transpose()
        else:
            self.loadingStatusLabel.setText('Getting the timing data')
            self.outputPlot.loadData(data.folder, self.progress, self.app)
            for i in range(-1, 7):
                self.outputPlot.inc = i
                self.outputPlot.updatePlot()

        # Adjust the lines to fit what was used
        if data.settings is not None:
            with open(data.settings) as file:
                self.settings = json.load(file)
            self.outputPlot.thrLine1.setPos(self.settings['lowerThreshold'])
            self.outputPlot.thrLine2.setPos(self.settings['upperThreshold'])

        self.onTimer(0)
        if not self.virtualStack:
            self.viewerOrig.resetRanges()
        self.viewerProc.resetRanges()
        self.viewerNN.resetRanges()
        self.onTimer()
        # set up the progress bar
        self.frameSlider.setDisabled(False)

    def updateProgress(self, rangeMax):
        """ Set the range of the progress bar """
        self.progress.setMaximum(rangeMax)

    def loadDataThread(self):
        """ Create and start the thread to load data """
        worker = LoadingThread(self.folder, self.order, self.model, self.virtualStack)
        thread = QThread()
        thread.setObjectName('Data Loader')
        self.threads.append((thread, worker))
        worker.moveToThread(thread)
        worker.change_progress.connect(self.setProgressValue)
        worker.setLabel.connect(self.setLabelString)
        worker.setLog.connect(self.setLogString)
        worker.loadingDone.connect(self.receiveData)
        worker.updateProgressRange.connect(self.updateProgress)
        thread.started.connect(worker.work)
        thread.start()

    def loadModel(self):
        """ Load a .h5 model generated using Keras """
        self.loadingStatusLabel.setText('Loading Model')
        fname = QFileDialog.getOpenFileName(
            self, 'Open file', '//lebnas1.epfl.ch/microsc125/Watchdog/Model',
            "Keras models (*.h5)")
        self.model = keras.models.load_model(fname[0], compile=True)
        self.loadingStatusLabel.setText('Done')
        self.log.appendPlainText(fname[0])
        self.dataButton.setDisabled(False)

    def hideViewerOrig(self):
        """ Hide the Orig view to get better performance for big datasets """

        if self.viewerOrig.isHidden():
            self.viewerOrig.show()
        else:
            self.viewerOrig.hide()

    def hideViewerNN(self):
        """ Hide the NN view to get better performance for big datasets """

        if self.viewerNN.isHidden():
            self.viewerNN.show()
        else:
            self.viewerNN.hide()

    def setVirtualCallback(self):
        """ React to a press of the virtual Stack button """
        if self.virtualStack:
            self.virtualStack = False
            self.viewerOrig.show()
            self.hideOrig.setChecked(False)
        else:
            self.virtualStack = True
            self.viewerOrig.hide()
            self.hideOrig.setChecked(True)

    def orderChange(self):
        """ React to a press of the order button to read interleaved data into the right order """
        if self.order == 0:
            self.order = 1
            orderStr = 'order: Drp first'
        else:
            self.order = 0
            orderStr = 'order: Mito first'
        self.setLogString('Set ' + orderStr)
        self.orderButton.setText(orderStr)

    def onSlider(self):
        """ Call onTimer if the timer is not active but the value of the slider changed anyway.
        This is especially the case, if using the mouse wheel to scroll the slider."""
        if not self.timer.isActive():
            self.onTimer()

    def onTimer(self, i=None):
        """ Reset the data in the GUI on the timer when button or slider is pressed """
        if i is None:
            i = self.frameSlider.value()

        if self.virtualStack is True:
            self.getFileNames(i)
            images = tifffile.imread(self.fileList)
            self.viewerProc.setImage(images[0], 1)
            self.viewerProc.setImage(images[1], 0)
            self.viewerNN.setImage(images[2], 0)
        else:
            self.viewerOrig.setImage(self.imageMitoOrig[i], 1)
            self.viewerOrig.setImage(self.imageDrpOrig[i], 0)
            self.viewerProc.setImage(self.mitoDataFull[i], 1)
            self.viewerProc.setImage(self.drpDataFull[i], 0)
            self.viewerNN.setImage(self.nnOutput[i], 0)
            self.currentFrameLabel.setText(str(i))

        # if self.mode == 'stack':
        #     self.frameLine.setValue(i)
        # else:
        self.frameLine.setValue(self.outputPlot.elapsed[i])
        self.viewerOrig.cross.setPosition([self.maxPos[i][0]])
        self.viewerProc.cross.setPosition([self.maxPos[i][0]])
        self.viewerNN.cross.setPosition([self.maxPos[i][0]])
        self.viewerProc.crossRational.setPosition([self.maxPosRational[i][0]])

    def getFileNames(self, frame):
        """ Get the filenames for display of a specific frame in virtual stack mode """
        baseName = '/img_channel000_position000_time'
        self.fileList[self.order] = (self.virtualFolder + baseName +
                                     str((frame*2 + 1)).zfill(9) + '_z000_prep.tif')
        self.fileList[np.abs(self.order - 1)] = (self.virtualFolder + baseName +
                                                 str((frame*2)).zfill(9) + '_z000_prep.tif')
        self.fileList[2] = self.virtualFolder + baseName + str((frame*2 + 1)).zfill(9) + '_nn.tiff'

    def startTimer(self):
        """ start Timer when slider is pressed """
        self.timer.start()

    def stopTimer(self):
        """ stop timer when slider is released"""
        self.timer.stop()

    # def refreshGradients(self):
    #     """ refresh the images when a LUT was changed in the popup window """
    #     self.viewerOrig.setImage(self.imageMitoOrig[0], 1)
    #     self.viewerOrig.setImage(self.imageDrpOrig[0], 0)
    #     self.viewerProc.setImage(self.mitoDataFull[0], 1)
    #     self.viewerProc.setImage(self.drpDataFull[0], 0)
    #     self.viewerNN.setImage(self.nnOutput[0], 0)

    def nextFrame(self):
        """ display next frame in all viewBoxes when '>' button is pressed """
        i = self.frameSlider.value()
        self.frameSlider.setValue(i + 1)
        self.onTimer()

    def prevFrame(self):
        """ display previous frame in all viewBoxes when '<' button is pressed """
        i = self.frameSlider.value()
        self.frameSlider.setValue(i - 1)
        self.onTimer()

    def sliderWheelEvent(self, event):
        self.frameSlider.setValue(self.frameSlider.value() + np.sign(event.angleDelta().y()))

    def closeEvent(self, _):
        """ Terminate the threads that are running"""
        for thread in self.threads:
            thread[0].quit()


class LoadingThread(QObject):
    """ Extra Thread for loading data """
    change_progress = pyqtSignal(int)
    setLabel = pyqtSignal(str)
    loadingDone = pyqtSignal(QObject)
    setLog = pyqtSignal(str)
    updateProgressRange = pyqtSignal(int)

    def __init__(self, startFolder, dataOrder, kerasModel, virtualStack):
        super().__init__()
        self.startFolder = startFolder
        self.dataOrder = dataOrder
        self.model = kerasModel
        self.virtualStack = virtualStack
        self.mode = None
        self.imageDrpOrig = None
        self.imageMitoOrig = None
        self.nnOutput = None
        self.outputData = None
        self.mitoDataFull = None
        self.drpDataFull = None
        self.nnRecalculated = None
        self.elapsed = None
        self.postSize = None
        self.frameNum = None
        self.maxPos = []
        self.maxPosRational = []
        self.settings = None
        self.folder = None
        self.nnImageSize = 128
        self.pixelCalib = 56  # nm per pixel
        self.resizeParam = self.pixelCalib/81  # no unit

    @pyqtSlot()
    def work(self):
        """ Function used to start the loading in the thread load """
        self.setLog.emit('\n\nLoading thread started')
        self._getFolder()
        newData = self._loadData()
        self.loadingDone.emit(newData)

    def _getFolder(self):
        """Getting GUI folder input, determine data type and load accordingly
        order 0 is mito first"""
        fname = QFileDialog.getOpenFileName(QWidget(), 'Open file', self.startFolder)
        self.setLabel.emit('Loading images from files into arrays')
        if self.virtualStack:
            self.setLog.emit("Virtual stack mode")
            self.folder = os.path.dirname(fname[0])
            self.setLog.emit(self.folder)
            self.startFolder = os.path.dirname(os.path.dirname(fname[0]))
            self.mode = 'virtual'
            if os.path.exists(self.folder + '/ATSSim_settings.json'):
                self.settings = self.folder + '/ATSSim_settings.json'
        elif not re.match(r'img_channel\d+_position\d+_time', os.path.basename(fname[0])) is None:
            # Check for microManager file like name and if so load as files in folder
            self.setLog.emit("Folder mode")
            self.mode = 'folder'
            self.imageDrpOrig, self.imageMitoOrig, self.nnOutput = loadTifFolder(
                os.path.dirname(fname[0]), self.resizeParam, self.dataOrder)
            # Save this to go back to when the user wants to load another file
            self.startFolder = os.path.dirname(os.path.dirname(fname[0]))
            # Save this to use in SATS_gui
            self.folder = os.path.dirname(fname[0])
            self.setLog.emit(os.path.dirname(fname[0]))
            # Check if this folder was written by ATSSim
            if os.path.exists(self.folder + '/ATSSim_settings.json'):
                self.settings = self.folder + '/ATSSim_settings.json'
        else:
            # If not singular files in folder, load as interleaved stack
            self.mode = 'stack'
            self.setLog.emit("Stack mode")
            detectDataOrder = dataOrderMetadata(fname[0], write=False)
            if detectDataOrder is not None:
                self.dataOrder = int(detectDataOrder)
            self.imageDrpOrig, self.imageMitoOrig = loadTifStack(fname[0], order=self.dataOrder)
            # Save this to go back to when the user wants to load another file
            self.startFolder = os.path.dirname(fname[0])
            self.folder = self.startFolder
            self.setLog.emit(fname[0])
            self.elapsed = loadTifStackElapsed(fname[0])
        # self.updateProgressRange.emit(self.imageDrpOrig.shape[0])
        self.setLog.emit('Data order: ' + str(self.dataOrder))

    def _loadData(self):
        """load tif stack or ATS folder into the GUI. Reports progress using a textbox and the
        progress bar.
        """
        self.outputData = []

        # Make shortcut if using virtual stack
        if self.mode == 'virtual':
            allFiles = sorted(glob.glob(self.folder + '/img_channel*_nn.tiff'))
            splitStr = re.split(r'img_channel\d+_position\d+_time',
                                allFiles[-1])
            splitStr = re.split(r'_nn+', splitStr[1])
            self.frameNum = int((int(splitStr[0])-1)/2)
            for frame in range(0, self.frameNum):
                nnOutput = tifffile.imread(allFiles[frame])
                self.outputData.append(np.max(nnOutput))
                self.maxPos.append(list(zip(
                    *np.where(nnOutput == self.outputData[-1]))))
                self.postSize = nnOutput.shape[1]
            return self

        self.change_progress.emit(0)
        self.setLog.emit('input shape {}'.format(self.imageMitoOrig.shape))
        # Initialize values and data for neural network
        self.frameNum = self.imageMitoOrig.shape[0]
        self.postSize = round(self.imageMitoOrig.shape[1]*self.resizeParam)
        if self.model.layers[0].input_shape[0][1] is None:
            # If the network is for full shape images, be sure that shape is multiple of 4
            self.postSize = self.postSize - self.postSize % 4
        # if self.mode == 'stack':
        self.nnOutput = np.zeros((self.frameNum, self.postSize, self.postSize))

        self.mitoDataFull = np.zeros_like(self.nnOutput)
        self.drpDataFull = np.zeros_like(self.nnOutput)
        self.nnRecalculated = np.zeros(self.frameNum)

        # Process data for all frames that where found
        self.setLabel.emit('Processing frames and running the network')
        for frame in range(0, self.imageMitoOrig.shape[0]):
            # Make preprocessed tiles that can be fed to the neural network
            inputData, positions = prepareNNImages(
                self.imageMitoOrig[frame, :, :],
                self.imageDrpOrig[frame, :, :], self.model)

            # Do the NN calculation if there is not already a file there
            outputPredict = None
            if self.mode == 'folder' and np.max(self.nnOutput[frame]) > 0:
                nnDataPres = 1
            else:
                nnDataPres = 0
                outputPredict = self.model.predict_on_batch(inputData)
                self.nnRecalculated[frame] = 1

            if self.model.layers[0].input_shape[0][1] is None:
                # Just copy the full frame if a full frame network was used
                if nnDataPres == 0:
                    self.nnOutput[frame] = outputPredict[0, :, :, 0]
                    self.mitoDataFull[frame] = inputData[0, :, :, 0, 0]
                    self.drpDataFull[frame] = inputData[0, :, :, 1, 0]
            else:
                # Stitch the tiles made back together if model needs it
                i = 0
                st0 = positions['stitch']
                st1 = None if st0 == 0 else -st0
                for pos in positions['px']:
                    if nnDataPres == 0:
                        self.nnOutput[frame, pos[0]+st0:pos[2]-st0, pos[1]+st0:pos[3]-st0] =\
                                    outputPredict[i, st0:st1, st0:st1, 0]
                    self.mitoDataFull[frame, pos[0]+st0:pos[2]-st0, pos[1]+st0:pos[3]-st0] = \
                        inputData[i, st0:st1, st0:st1, 0]
                    self.drpDataFull[frame, pos[0]+st0:pos[2]-st0, pos[1]+st0:pos[3]-st0] = \
                        inputData[i, st0:st1, st0:st1, 1]
                    i += 1

            # Get the output data from the nn channel and its position
            self.outputData.append(np.max(self.nnOutput[frame, :, :]))
            self.maxPos.append(list(zip(*np.where(self.nnOutput[frame] == self.outputData[-1]))))

            self.change_progress.emit(frame)
            # QApplication.processEvents()

        # Check if there is an output.txt in this folder, if not write it.
        txtFile = os.path.join(self.folder, 'output.txt')
        if not os.path.isfile(txtFile) and self.mode == 'folder':
            file = open(txtFile, 'w+')
            print(self.outputData)
            if np.max(self.outputData) > 1.1:
                for frameNum, output in enumerate(self.outputData):
                    file.write('%d, %d\n' % (frameNum, output))
            else:
                for frameNum, output in enumerate(self.outputData):
                    file.write('%d, %f\n' % (frameNum, output))
            file.close()

        self.setLabel.emit('Resize the original frames to fit the output')
        self.setLog.emit('Size after network: {}x{}'.format(self.postSize, self.postSize))
        imageDrpOrigScaled = np.zeros((self.imageDrpOrig.shape[0], self.postSize, self.postSize))
        imageMitoOrigScaled = np.zeros((self.imageDrpOrig.shape[0], self.postSize, self.postSize))
        for i in range(0, self.imageMitoOrig.shape[0]):
            imageDrpOrigScaled[i] = transform.rescale(self.imageDrpOrig[i],
                                                      self.postSize/self.imageDrpOrig[i].shape[1],
                                                      anti_aliasing=True, preserve_range=True)
            imageMitoOrigScaled[i] = transform.rescale(self.imageMitoOrig[i],
                                                       self.postSize/self.imageMitoOrig[i].shape[1],
                                                       anti_aliasing=True, preserve_range=True)
            self.maxPosRational.append(
                list(zip(*np.where(imageDrpOrigScaled[i] == np.max(imageDrpOrigScaled[i])))))
            self.change_progress.emit(i)
            # QApplication.processEvents()

        # Rescale the exposures
        self.setLabel.emit('Rescale the original frames to 8 bit')
        imageDrpOrigScaled = exposure.rescale_intensity(
            np.array(imageDrpOrigScaled), (np.min(np.array(imageDrpOrigScaled)),
                                           np.max(np.array(imageDrpOrigScaled))),
            out_range=(0, 255))
        imageMitoOrigScaled = exposure.rescale_intensity(
            np.array(imageMitoOrigScaled), (np.min(np.array(imageMitoOrigScaled)),
                                            np.max(np.array(imageMitoOrigScaled))),
            out_range=(0, 255))
        self.imageDrpOrig = np.array(imageDrpOrigScaled).astype(np.uint8)
        self.imageMitoOrig = np.array(imageMitoOrigScaled).astype(np.uint8)

        return self


def main():
    """ main method to run and display the NNGui """
    app = QApplication(sys.argv)
    # setAttribute(QtCore.Qt.AA_Use96Dpi)
    stackViewer = NNGui(app)

    stackViewer.showMaximized()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
