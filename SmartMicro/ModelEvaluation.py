import os

import h5py  # HDF5 data file management library
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtWidgets
from skimage import io
from tensorflow import keras
from tqdm import tqdm

from NNfeeder import prepareNNImages

plt.switch_backend('Agg')

DataPath = '//lebnas1.epfl.ch/microsc125/Watchdog/Model/'
Colors = {
    'ats':  (210/255, 42/255, 38/255),
    'slow': (2/255, 53/255, 62/255),
    'fast': (2/255, 147/255, 164/255)
}


def main(collection='paramSweep'):
    """ Main function of the module that prepares all the data for later plotting """
    inputDataPath = DataPath + collection + '/prep_data' + collection[-1] + '.h5'
    hf = h5py.File(inputDataPath, 'r')
    input1 = hf.get('Mito')  # Mito
    input1 = np.array(input1).astype(np.float)
    input2 = hf.get('Drp1')  # Mito
    input2 = np.array(input2).astype(np.float)
    inputDataFull = np.stack((input1, input2), axis=3)
    print('Converting to tensor')

    outputData = hf.get('Proc')
    outputDataFull = np.array(outputData).astype(bool)
    hf.close()

    try:
        iSIMdataPath = 'C:/Users/stepp/Documents/02_Raw/SmartMito/180420_130.tif'
        iSIMdata = io.imread(iSIMdataPath)
    except FileNotFoundError:
        iSIMdataPath = '//lebnas1.epfl.ch/microsc125/Watchdog/GUI/180420_130.tif'
        iSIMdata = io.imread(iSIMdataPath)
    iSIMdataPath2 = '//lebnas1.epfl.ch/microsc125/Watchdog/GUI/180420_111.tif'
    iSIMdata2 = io.imread(iSIMdataPath2)

    # Make the ouput data a little bigger, to not penalize to much on the correct location
    # for frame in range(outputData.shape[0]):
    #     outputData[frame] = morphology.binary_dilation(outputData[frame])
    #     outputData[frame] = morphology.binary_dilation(outputData[frame])

    print(inputDataFull.shape)
    print(outputDataFull.shape)

    pand = pd.DataFrame(columns=['model', 'filters', 'convs', 'batchSize', 'totalTruth',
                                 'totalPredict', 'predictVariance', 'maskPredictThresh',
                                 'maskPredict',
                                 'truePositive', 'truePositiveThresh', 'falsePositive',
                                 'falsePositiveThresh', 'iSIMoutput', 'iSIMoutput2'])

    # pand.loc[pand.shape[0]] = [modelName, f, c, b, totalTruth, totalPredict,
    #                             maskPredictThresh,
    #                             maskPredict, truePositive, truePositiveThresh,
    #                             falsePositive, falsePositiveThresh]

    filters = [8, 16, 32]
    convs = [3, 5, 7, 9]
    batches = [8, 16, 32]
    for f in filters:
        for c in convs:
            for b in batches:
                modelName = ('f' + str(f).zfill(2) +
                             '_c' + str(c).zfill(2) +
                             '_b' + str(b).zfill(2))

                modelPath = os.path.join(DataPath, collection, (modelName + '.h5'))
                testSetPath = os.path.join(DataPath, collection, (modelName + '_labels.pkl'))

                model = keras.models.load_model(modelPath, compile=True)

                # Get the data that was not used for training
                labels = pd.read_pickle(testSetPath)
                inputData = inputDataFull[labels]
                inputData = tf.convert_to_tensor(inputData)
                outputData = outputDataFull[labels]
                # for part in range(0, inputTest.shape[0], 100):
                predictTest = np.zeros([len(labels), 128, 128, 1])
                step = 100
                for frame in tqdm(np.arange(0, int(inputData.shape[0]/step))):
                    predictTest[frame*step:frame*step+step] = model.predict_on_batch(
                        inputData[frame*step:frame*step+step])

                # Predict on the actual iSIM data
                iSIMoutput = []
                for frame in tqdm(range(0, iSIMdata.shape[0], 2)):
                    iSIMdataprep = prepareNNImages(iSIMdata[frame], iSIMdata[frame+1], None)
                    iSIMoutput.append(np.max(model.predict(iSIMdataprep[0])))

                iSIMoutput2 = []
                for frame in tqdm(range(0, iSIMdata2.shape[0], 2)):
                    iSIMdataprep = prepareNNImages(iSIMdata2[frame], iSIMdata2[frame+1], None)
                    iSIMoutput2.append(np.max(model.predict(iSIMdataprep[0])))

                # Mask the prediction with the ground truth
                # Take the maximum, or what we take as input for ATS
                maxOutput = list()
                maxPredict = list()
                for frame in range(0, inputData.shape[0]):
                    maxPredict.append(np.max(predictTest[frame]))
                    maxOutput.append(np.max(outputData[frame]))
                maxPredict = np.array(maxPredict)

                predictVariance = np.var(maxPredict)

                maxOutput = np.array(maxOutput)
                predictTestTruePos = maxPredict[maxOutput]  # predictTest[outputData]

                totalPredict = np.sum(maxPredict)
                totalTruth = np.sum(maxOutput)
                maskPredict = np.sum(predictTestTruePos)
                maskPredictThresh = np.sum(predictTestTruePos > 0.85)
                truePositive = maskPredict/totalTruth
                truePositiveThresh = maskPredictThresh/totalTruth
                falsePositive = (totalPredict - maskPredict)/totalTruth
                falsePositiveThresh = (np.sum(maxPredict > 0.85) - maskPredictThresh)/totalTruth

                pand.loc[pand.shape[0]] = [modelName, f, c, b, totalTruth, totalPredict,
                                           predictVariance, maskPredictThresh,
                                           maskPredict, truePositive, truePositiveThresh,
                                           falsePositive, falsePositiveThresh, iSIMoutput,
                                           iSIMoutput2]
                print(pand)
    pand.to_pickle(os.path.join(DataPath, collection, 'evaluation.pkl'))

    return


class TableWindow(QtWidgets.QMainWindow):
    """ A window that we can plot into that gives us rows """
    def __init__(self):
        self.qapp = QtWidgets.QApplication([])
        QtWidgets.QMainWindow.__init__(self)
        self.widget = QtWidgets.QWidget()
        # self.widget.setWidgetResizable(True)
        self.setCentralWidget(self.widget)
        self.widget.setLayout(QtWidgets.QVBoxLayout())
        self.widget.layout().setContentsMargins(0, 0, 0, 0)
        self.widget.layout().setSpacing(0)

        self.table = QtWidgets.QScrollArea()
        self.tableWidget = QtWidgets.QWidget()
        self.tableWidget.setLayout(QtWidgets.QVBoxLayout())
        self.tableWidget.layout().setContentsMargins(0, 0, 0, 0)
        self.tableWidget.layout().setSpacing(0)
        self.table.setWidget(self.tableWidget)
        self.table.setWidgetResizable(True)
        self.table.setFixedWidth(1500)

        self.headerLine = QtWidgets.QWidget()
        self.headerLine.setLayout(QtWidgets.QVBoxLayout())
        self.headerLine.setFixedHeight(50)

        self.widget.layout().addWidget(self.headerLine)
        self.widget.layout().addWidget(self.table)

        self.lines = 0

    def addHeader(self, fig):

        canvas = FigureCanvas(fig)
        self.headerLine.layout().addWidget(canvas)

    def addLine(self, fig):
        canvas = FigureCanvas(fig)
        canvas.draw()
        self.tableWidget.layout().addWidget(canvas)
        self.tableWidget.layout().addStretch()
        self.lines = self.lines + 1

    def done(self):
        self.setGeometry(300, 100, 1500, 1000)
        self.tableWidget.setFixedHeight(70*self.lines)
        self.show()
        exit(self.qapp.exec_())


def visualizeDecision(collections):

    # Make line for one model
    for collection in collections:
        evalFilePath = os.path.join(DataPath, collection, 'evaluation.pkl')
        newPand = pd.read_pickle(evalFilePath)
        newPand['collection'] = collection
        if collection == collections[0]:
            pand = newPand
        else:
            pand = pd.concat([pand, newPand])
    print(pand)
    pand = pand.sort_values('truePositiveThresh', ascending=False)
    # Make a window that allows for esay adding of plot lines
    win = TableWindow()
    fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(16, 3))
    fig.subplots_adjust(bottom=-1, top=0)
    axes[0].set_title('Positives Thresholded')
    axes[1].set_title('Positives')
    axes[2].set_title('High Activity')
    axes[3].set_title('Low Activity')
    for ax in axes:
        ax.set_yticks([])
    win.addHeader(fig)

    for modelData in pand.iterrows():
        modelData = modelData[1]
        modelName = modelData['model']
        collection = modelData['collection']

        fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(16, 3))

        axes[0].barh([0, 1],
                     [modelData['truePositiveThresh'],
                      modelData['falsePositiveThresh']],
                     1,
                     color=[Colors['slow'], Colors['ats']],
                     zorder=10)
        axes[0].set_ylabel(modelName, rotation=0, labelpad=80, va='center')
        plt.text(0.92, 0.5, collection, transform=plt.gcf().transFigure, va='center')

        axes[1].barh([0, 1],
                     [modelData['truePositive'], modelData['falsePositive']],
                     1,
                     color=[Colors['slow'], Colors['ats']],
                     alpha=0.5,
                     zorder=10)

        for i in [0, 1]:
            axes[i].set_xlim((0, 1.01))
            axes[i].set_xticks([0.05, 0.1, 0.15, 0.7, 0.8, 0.9, 1])
            axes[i].set_xticklabels(['', '', '', '', '', '', ''])
            axes[i].set_yticks([])
            axes[i].set_frame_on(False)
            axes[i].grid(axis='x', zorder=1)
            axes[i].xaxis.set_tick_params(length=0)

        axes[2].plot(modelData['iSIMoutput'],
                     color=Colors['slow'])

        axes[3].plot(modelData['iSIMoutput2'],
                     color=Colors['slow'])

        for i in [2, 3]:
            axes[i].set_frame_on(False)
            axes[i].set_ylim((0, 1))
            axes[i].set_yticks([])
            axes[i].set_xticks([])
            greyVal = 0.75
            axes[i].axhline(1, color=(greyVal, greyVal, greyVal))
            axes[i].axhline(0, color=(greyVal, greyVal, greyVal))
            axes[i].grid(axis='y', zorder=1)
            axes[i].xaxis.set_major_locator(plt.NullLocator())
            axes[i].xaxis.set_major_formatter(plt.NullFormatter())

        win.addLine(fig)

    win.done()


def visualize(collection='paramSweep'):
    """Function to visualize the overall outcome of a paramSweep of models for different parameters

    Args:
        collection (str, optional): Which collection to use for analysis. Defaults to 'paramSweep'.
    """
    dataPath = 'W:/Watchdog/Model/'
    filePath = os.path.join(dataPath, collection, 'evaluation.csv')
    pand = pd.read_csv(filePath)
    print(pand)

    axesList = []

    fig1 = plt.figure(figsize=(20, 10))
    ax = fig1.add_subplot(331)
    axesList.append(ax)
    pand.sort_values('filters').plot(x='filters', y='truePositiveThresh', kind='bar', ax=ax,
                                     zorder=10)
    ax.set_title('Sorted by filters')
    plt.ylabel('True Positive Thresholded')
    ax = fig1.add_subplot(332)
    axesList.append(ax)
    pand.sort_values('convs').plot(x='convs', y='truePositiveThresh', kind='bar', ax=ax,
                                   zorder=10)
    ax.set_title('Sorted by firstConv')
    ax = fig1.add_subplot(333)
    axesList.append(ax)
    pand.sort_values('batchSize').plot(x='batchSize', y='truePositiveThresh', kind='bar', ax=ax,
                                       zorder=10)
    ax.set_title('Sorted by batchSize')

    ax = fig1.add_subplot(334)
    axesList.append(ax)
    pand.sort_values('filters').plot(x='filters', y='totalPredict', kind='bar', ax=ax, zorder=10)
    plt.ylabel('Total Prediction')
    ax = fig1.add_subplot(335)
    axesList.append(ax)
    pand.sort_values('convs').plot(x='convs', y='totalPredict', kind='bar', ax=ax, zorder=10)
    ax = fig1.add_subplot(336)
    axesList.append(ax)
    pand.sort_values('batchSize').plot(x='batchSize', y='totalPredict', kind='bar', ax=ax,
                                       zorder=10)

    ax = fig1.add_subplot(337)
    axesList.append(ax)
    pand.sort_values('filters').plot(x='filters', y='falsePositiveThresh', kind='bar', ax=ax,
                                     zorder=10)
    plt.ylabel('False Positive Thresholded')
    ax = fig1.add_subplot(338)
    axesList.append(ax)
    pand.sort_values('convs').plot(x='convs', y='falsePositiveThresh', kind='bar', ax=ax,
                                   zorder=10)
    ax = fig1.add_subplot(339)
    axesList.append(ax)
    pand.sort_values('batchSize').plot(x='batchSize', y='falsePositiveThresh', kind='bar', ax=ax,
                                       zorder=10)

    for ax in axesList:
        ax.get_legend().remove()
        ax.grid(axis='y', zorder=1)
    plt.show()


if __name__ == '__main__':
    coll = 'paramSweep8'
    visualizeDecision(['paramSweep5', 'paramSweep6', 'paramSweep7','paramSweep8', 'paramSweep9'])
    # main(coll)
    # visualize(coll)
