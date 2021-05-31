
import sys
import os
import re
import tifffile
import json
import glob
import xmltodict

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import SmartMicro.NNio
from toolbox.plotting.style_mpl import set_mpl_style as set_style
from toolbox.plotting.style_mpl import set_mpl_font as set_font

COLORS = {
        'ats':  (210/255, 42/255, 38/255),
        'slow': (2/255, 53/255, 62/255),
        'fast': (2/255, 147/255, 164/255)
    }
style = "publication"
set_style(style)
set_font(size=12)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = [colors[0], colors[3], colors[2], colors[5], colors[4]]

output_folder = "//lebsrv2.epfl.ch/LEB_Perso/Willi-Stepp/ATS_Figures/"

PLOT_RATIONAL = False


def main():
    # set plot style
    matplotlib.rcParams.update({'lines.linewidth': 4})


    # ATSquant(colors, skipTime=5)
    file = 'c:/Users/stepp/Documents/05_Software/Analysis/fullframe_model/model4_noATS.pkl'
    file = 'c:/Users/stepp/Documents/02_Raw/SmartMito/s01_c03_ATS2/plot.pkl'
    # file = 'C:/Users/stepp/OneDrive - epfl.ch/210316_GroupMeeting/binary_noATSfinal.pkl'
    # file = 'c:/Users/stepp/Documents/05_Software/Analysis/Caulobacter/realATS.pkl'
    # file = 'c:/Users/stepp/Documents/05_Software/Analysis/Caulobacter_ats/rational_dec/210416_02_rational.pkl'
    # file = 'W:/Watchdog/bacteria/210506_weakSyncro/FOV_1/ATS_plot.pkl'
    file = 'c:/Users/stepp/Documents/05_Software/Analysis/210316_GroupMeeting/ATS_0.pkl'

    replotATStimes(file, COLORS)


def replotATStimes(file, colors):
    with open(file, 'rb') as fileHandle:
        data = pickle.load(fileHandle)

    if style == "publication":
        cm = 1/2.54
        fig, ax1 = plt.subplots(figsize=(16*cm, 6*cm))
    else:
        fig, ax1 = plt.subplots(figsize=(18,7.5))
    diff = np.diff(data['times'])
    noTimes = False
    if diff[2] == 2:
        print('No times for frames')
        ax1.set_xlabel('Frames')
        noTimes = True
    else:
        if diff[0] > 10:
            data['times'] = [time/1000 for time in data['times']]
        ax1.set_xlabel('Times [s]')

    if len(data['delay']) == 5:
        # The delay data was probably not saved from NNGui
        data['delay'] = np.round(np.diff(data['times'])*100)
        data['delay'] = np.insert(data['delay'],0 ,data['delay'][0])[:-1]

    delayY, delayX = makeStepSeries(data['delay'][np.abs(len(data['times'])-len(data['delay'])):],
                                    data['times'])
    delayY = [1/delay for delay in delayY]
    # ignore first fast frame for plot

    delayY[0] = delayY[2]
    delayY[1] = delayY[2]


    diff = np.diff(delayY)
    print(diff)
    ats = bool(np.sum(np.abs(diff))*1000)
    if ats:
        ax2 = ax1.twinx()
        ax2.plot(delayX, delayY, color=colors['ats'], linewidth=1)
        ax2.set_ylabel('Frame Rate [Hz]', color = colors['ats'])
        rect = matplotlib.patches.Rectangle((delayX[0], 80), delayX[-1] - delayX[0], 20,
                                            linewidth=0, edgecolor='none',
                                            facecolor=(0.5, 0.5, 0.5, 0.5))
        ax1.add_patch(rect)
    else:
        if not noTimes:
            txt = plt.text(0.05, 0.9, 'Frame Rate: {} Hz'.format(1/delayY[0]), fontsize=12,
                           color=colors['ats'], transform=ax1.transAxes)

    # ax1.plot(data['times'][(data['times'].shape[0] - data['nnOutput'].shape[0]):],
    #          data['nnOutput'][:,1], color=colors['slow'], linewidth=1,
    #                     linestyle='dotted', marker='.', markersize=7)

    # do not plot the last values of nnOutput
    crop = 1
    ax1.plot(data['times'][(data['times'].shape[0] - data['nnOutput'].shape[0]):-crop],
             data['nnOutput'][:-crop,1], color=colors['slow'], linewidth=1,
                        linestyle='dotted', marker='.', markersize=7)

    if PLOT_RATIONAL:
        try:
            ax1.plot(data['times'][(data['times'].shape[0] - data['rational'].shape[0]):],
                    data['rational'], color=colors['fast'], linewidth=1,
                                linestyle='dotted', marker='.', markersize=7)
            h_legend = ax1.legend(('nn', 'rational'),fontsize=10, bbox_to_anchor=(0, 1.15))
        except KeyError:
            print('No rational data')


    ax1.grid(False)
    ax2.grid(False)
    ax1.set_ylabel('Neural Network [AU]', color=colors['slow'])
    plt.tight_layout()
    plt.savefig(output_folder + "ATS_plot.pdf", facecolor=None, edgecolor=None,
                transparent=True)
    plt.show()



def ATSquant(colors, skipTime=5):

    directory = 'W:/iSIMstorage/Users/Willi/160622_caulobacter/160622_CB15N_WT/'
    directory = 'W:/iSIMstorage/Users/Willi/180420_drp_mito_Dora/'
    # dir = ('//lebnas1/microsc125/iSIMstorage/Users/Dora/20201205_mitoSmart/'
    #        'cell_Int0s_30pc_488_50pc_561_2/cropped')
    saveFolder = 'C:/Users/stepp/Documents/05_Software/Analysis/ffbinary_data/'
    thr = .9  # threshold for the thresholded info plots; 90 for caulobacter

    # calculateAll(directory, saveFolder, skipTime)

    atsTimes = loadObj('atsTimes', saveFolder)
    atsInfos = loadObj('atsInfos', saveFolder)
    folders = loadObj('folders', saveFolder)

    fastTimes = loadObj('fastTimes', saveFolder)
    fastInfos = loadObj('fastInfos', saveFolder)

    slowTimes = loadObj('slowTimes', saveFolder)
    slowInfos = loadObj('slowInfos', saveFolder)

    fig = plt.figure(figsize=(15,13))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    ats = None  # placeholders for later plot
    fast = None
    slow = None
    slowColor = colors['slow']  # '#0294A5'
    fastColor = colors['fast']  # '#03353E'
    atsColor =  colors['ats'] # '#C1403D'

    slowInfosThrs = {}
    fastInfosThrs = {}
    atsInfosThrs = {}
    # folders.pop(2)
    print('\n'.join(folders))

    # find the right cropPos
    cropPos = 1000
    for folder in folders:
        file = getFileName(folder, saveFolder)
        cropPos = np.min([len(atsInfos[folder]), cropPos, len(slowInfos[file])])
    print('cropPos adjusted: ', cropPos)



    # for folder in [folders[2]]:
    i = 0
    for folder in folders:
        # folder = folders[0]
        file = getFileName(folder, saveFolder)
        print(file)


        # do a thresholded info approach for all info lists
        atsInfosThr = makeThrList(atsInfos[folder], thr)
        slowInfosThr = makeThrList(slowInfos[file], thr)
        fastInfosThr = makeThrList(fastInfos[file], thr)

        #crop all

        # fastInfosMat = np.array(np.subtract(fastInfos[file][0*cropPos:0*cropPos+cropPos], fastInfos[file][0*cropPos]))
        # for i in range(1, int(np.floor(len(fastInfos[file])/cropPos))):
        #     fastInfosMat = np.vstack((fastInfosMat, np.subtract(fastInfos[file][i*cropPos:i*cropPos+cropPos], fastInfos[file][i*cropPos])))
        # fastInfos[file] = fastInfosMat.mean(0)


        # # do the same for the thresholded fast
        # fastInfosMat = np.array(np.subtract(fastInfosThr[0*cropPos:0*cropPos+cropPos], fastInfosThr[0*cropPos]))
        # for i in range(1, int(np.floor(len(fastInfosThr)/cropPos))):
        #     fastInfosMat = np.vstack((fastInfosMat, np.subtract(fastInfosThr[i*cropPos:i*cropPos+cropPos], fastInfosThr[i*cropPos])))
        # fastInfosThrs[file] = fastInfosMat.mean(0)


        # normalize all times to 150 cycle time
        # timeFactor = (fastTimes[file][1]-fastTimes[file][0])/150

        slowInfos[file] = slowInfos[file][:cropPos]
        slowInfos[file] = slowInfos[file] - np.min(slowInfos[file]) + 1
        slowTimes[file] = np.true_divide(slowTimes[file][:cropPos],1000)
        slowInfosThrs[file] = slowInfosThr[:cropPos]
        slowInfosThrs[file] = slowInfosThrs[file] - np.min(slowInfosThrs[file]) + 1

        atsInfos[folder] = atsInfos[folder][:cropPos]
        atsInfos[folder] = atsInfos[folder] - np.min(atsInfos[folder]) + 1
        atsTimes[folder] = np.true_divide(atsTimes[folder][:cropPos],1000)
        atsInfosThrs[folder] = atsInfosThr[:cropPos]
        atsInfosThrs[folder] = atsInfosThrs[folder] - np.min(atsInfosThrs[folder]) + 1

        # First frames many times are good, so take fast part from somewhere in the middle
        fastOffset = int(len(fastTimes[file])/2 - cropPos/2)
        fastTimes[file] = np.true_divide(fastTimes[file][:cropPos],1000)

        fastInfosThrs[file] = fastInfosThr[fastOffset:cropPos + fastOffset]
        fastInfosThrs[file] = fastInfosThrs[file] - np.min(fastInfosThrs[file]) + 1

        fastInfos[file] = fastInfos[file][fastOffset:cropPos + fastOffset]
        fastInfos[file] = fastInfos[file] - np.min(fastInfos[file]) + 1

        i = i + 1
        # skip some folders for plotting, the data will still be in the mean plots
        # if not i == 5:
        # print(folder)
        #     continue

        plt.sca(ax1)
        ats, = plt.plot(atsTimes[folder], atsInfos[folder], color=atsColor)
        fast, = plt.plot(fastTimes[file], fastInfos[file], color=fastColor)
        slow, = plt.plot(slowTimes[file], slowInfos[file], color=slowColor)

        plt.sca(ax2)
        ats, = plt.plot(np.linspace(
            0, 100, len(atsInfos[folder])), atsInfos[folder], color=atsColor)
        fast, = plt.plot(np.linspace(
            0, 100, len(fastInfos[file])), fastInfos[file], color=fastColor)
        slow, = plt.plot(np.linspace(
            0, 100, len(slowInfos[file])), slowInfos[file], color=slowColor)

        plt.sca(ax3)
        ats, = plt.plot(atsTimes[folder], np.linspace(
            0, 100, len(atsTimes[folder])), color=atsColor)
        fast, = plt.plot(fastTimes[file], np.linspace(
            0, 100, len(fastTimes[file])), color=fastColor)
        slow, = plt.plot(slowTimes[file], np.linspace(
            0, 100, len(slowTimes[file])), color=slowColor)

        plt.sca(ax4)
        ats, = plt.plot(np.linspace(
            0, 100, len(atsInfosThrs[folder])), atsInfosThrs[folder], color=atsColor)
        fast, = plt.plot(np.linspace(
            0, 100, len(fastInfosThrs[file])), fastInfosThrs[file], color=fastColor)
        slow, = plt.plot(np.linspace(
            0, 100, len(slowInfosThrs[file])), slowInfosThrs[file], color=slowColor)



    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Cumulative Information [AU]')
    ax1.legend((ats, fast, slow),('ats', 'fast', 'slow'))
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax2.set_xlabel('Photon Budget Spent [%]')
    ax2.set_ylabel('Cumulative Information [AU]')
    ax2.legend((ats, fast, slow),('ats', 'fast', 'slow'))
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Photon Budget [%]')
    ax3.legend((ats, fast, slow),('ats', 'fast', 'slow'))
    ax4.set_xlabel('Photon Budget Spent [%]')
    ax4.set_ylabel('Cumulative Information [AU]')
    ax4.legend((ats, fast, slow),('ats', 'fast', 'slow'))
    ax4.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.savefig(saveFolder + 'fullFig.png')


    # Calculate means for the metaplots
    mode = 'perSample'
    # per Sample means that ATS slow and fast are compared per sample and a ratio to the mean
    # is extracted and then the mean of this ratio is calculated for each Mode
    # per Mode means that the raw information values are calculated for each mode and that metric
    # is comapred. This is the older mode.
    if mode == 'perMode':
        file = folders[0][:-8] + '_nn.ome.tif'
        file = (os.path.dirname(folders[0]) + '/sample1_cell_' + folders[0][-6] +
                '_MMStack_Pos0_combine_nn_ffbinary.ome.tif')
        fastInfosMat = fastInfos[file]
        slowInfosMat = slowInfos[file]
        atsInfosMat = atsInfos[folders[0]]
        for i in range(1, len(folders)):
            file = folders[i][:-8] + '_nn.ome.tif'
            file = (os.path.dirname(folders[i]) + '/sample1_cell_' + folders[i][-6] +
                '_MMStack_Pos0_combine_nn_ffbinary.ome.tif')
            fastInfosMat = np.vstack((fastInfosMat, fastInfos[file]))
            slowInfosMat = np.vstack((slowInfosMat, slowInfos[file]))
            atsInfosMat = np.vstack((atsInfosMat, atsInfos[folders[i]]))
        fastInfosMean = fastInfosMat.mean(0)
        slowInfosMean = slowInfosMat.mean(0)
        atsInfosMean = atsInfosMat.mean(0)

        file = folders[0][:-8] + '_nn.ome.tif'
        file = (os.path.dirname(folders[0]) + '/sample1_cell_' + folders[0][-6] +
                '_MMStack_Pos0_combine_nn_ffbinary.ome.tif')
        fastInfosThrsMat = fastInfosThrs[file]
        slowInfosThrsMat = slowInfosThrs[file]
        atsInfosThrsMat = atsInfosThrs[folders[0]]
        for i in range(1, len(folders)):
            file = folders[i][:-8] + '_nn.ome.tif'
            file = (os.path.dirname(folders[i]) + '/sample1_cell_' + folders[i][-6] +
                '_MMStack_Pos0_combine_nn_ffbinary.ome.tif')
            fastInfosThrsMat = np.vstack((fastInfosThrsMat, fastInfosThrs[file]))
            slowInfosThrsMat = np.vstack((slowInfosThrsMat, slowInfosThrs[file]))
            atsInfosThrsMat = np.vstack((atsInfosThrsMat, atsInfosThrs[folders[i]]))
        fastInfosThrsMean = fastInfosThrsMat.mean(0)
        slowInfosThrsMean = slowInfosThrsMat.mean(0)
        atsInfosThrsMean = atsInfosThrsMat.mean(0)

    elif mode == 'perSample':
        file = getFileName(folders[0], saveFolder)
        meanInfo = np.vstack((fastInfos[file], slowInfos[file], atsInfos[folders[0]])).mean(0)
        fastInfosMat = np.divide(fastInfos[file], meanInfo)
        slowInfosMat = np.divide(slowInfos[file], meanInfo)
        atsInfosMat = np.divide(atsInfos[folders[0]], meanInfo)
        for i in range(1, len(folders)):
            file = getFileName(folders[i], saveFolder)
            meanInfo = np.vstack((fastInfos[file], slowInfos[file], atsInfos[folders[i]])).mean(0)
            fastInfosMat = np.vstack((fastInfosMat, np.divide(fastInfos[file], meanInfo)))
            slowInfosMat = np.vstack((slowInfosMat, np.divide(slowInfos[file], meanInfo)))
            atsInfosMat = np.vstack((atsInfosMat, np.divide(atsInfos[folders[i]], meanInfo)))
        fastInfosMean = fastInfosMat.mean(0)
        slowInfosMean = slowInfosMat.mean(0)
        atsInfosMean = atsInfosMat.mean(0)

        file = getFileName(folders[0], saveFolder)
        meanInfo = np.vstack((fastInfosThrs[file], slowInfosThrs[file], atsInfosThrs[folders[0]])).mean(0)
        fastInfosThrsMat = np.divide(fastInfosThrs[file], meanInfo)
        slowInfosThrsMat = np.divide(slowInfosThrs[file], meanInfo)
        atsInfosThrsMat = np.divide(atsInfosThrs[folders[0]], meanInfo)
        for i in range(1, len(folders)):
            file = getFileName(folders[i], saveFolder)
            meanInfo = np.vstack((
                fastInfosThrs[file], slowInfosThrs[file], atsInfosThrs[folders[i]])).mean(0)
            fastInfosThrsMat = np.vstack(
                (fastInfosThrsMat, np.divide(fastInfosThrs[file], meanInfo)))
            slowInfosThrsMat = np.vstack(
                (slowInfosThrsMat, np.divide(slowInfosThrs[file], meanInfo)))
            atsInfosThrsMat = np.vstack(
                (atsInfosThrsMat, np.divide(atsInfosThrs[folders[i]], meanInfo)))
        fastInfosThrsMean = fastInfosThrsMat.mean(0)
        slowInfosThrsMean = slowInfosThrsMat.mean(0)
        atsInfosThrsMean = atsInfosThrsMat.mean(0)

    fig = plt.figure(figsize=(10,13))
    ax1 = fig.add_subplot(211)
    ats, = plt.plot(np.linspace(0, 100, len(atsInfosMean)), atsInfosMean, color=atsColor)
    fast, = plt.plot(np.linspace(0, 100, len(fastInfosMean)), fastInfosMean, color=fastColor)
    slow, = plt.plot(np.linspace(0, 100, len(slowInfosMean)), slowInfosMean, color=slowColor)
    ax1.set_xlabel('Photon Budget Spent [%]')
    ax1.set_ylabel('Cumulative Information [AU]')
    ax1.legend((ats, fast, slow),('ats', 'fast', 'slow'))
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    ax2 = fig.add_subplot(212)
    ats, = plt.plot(np.linspace(0, 100, len(atsInfosThrsMean)), atsInfosThrsMean, color=atsColor)
    fast, = plt.plot(np.linspace(0, 100, len(fastInfosThrsMean)), fastInfosThrsMean, color=fastColor)
    slow, = plt.plot(np.linspace(0, 100, len(slowInfosThrsMean)), slowInfosThrsMean, color=slowColor)
    ax2.set_xlabel('Photon Budget Spent [%]')
    ax2.set_ylabel('Cumulative Information [AU]')
    ax2.legend((ats, fast, slow),('ats', 'fast', 'slow'))
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    plt.savefig(saveFolder + 'meanFig.png')
    plt.show()


def calculateAll(directory, saveFolder, skipTime):
    atsTimes, atsInfos, atsBleachs, folders = getATSData(directory)
    # save this information
    saveObj(atsTimes, 'atsTimes', saveFolder)
    saveObj(atsInfos, 'atsInfos', saveFolder)
    saveObj(atsBleachs, 'atsBleachs', saveFolder)
    saveObj(folders, 'folders', saveFolder)

    fastInfos, fastTimes = getOriginalData(folders)
    saveObj(fastTimes, 'fastTimes', saveFolder)
    saveObj(fastInfos, 'fastInfos', saveFolder)

    slowInfos, slowTimes = getOriginalData(folders, skipTime, fastTimes) # second parameter skip seconds
    saveObj(slowTimes, 'slowTimes', saveFolder)
    saveObj(slowInfos, 'slowInfos', saveFolder)

def makeThrList(infoList, thr):
    infoListThr = [0]
    for i in range(1, len(infoList)):
        change = infoList[i] - infoList[i-1]
        if infoList[i] - infoList[i-1] > thr:
            infoListThr.append(infoListThr[-1] + change)
        else:
            infoListThr.append(infoListThr[-1])
    return infoListThr

def getOriginalData(folders, skipTime=0, fastTimes=None):
    infos = {}
    elapsed = {}

    for folder in tqdm(folders):
        file = getFileName(folder)
        if fastTimes is not None:
            skipFrames = int(np.round(skipTime*1000/(fastTimes[file][1]-fastTimes[file][0])))-1
            print('skip frames: ', str(skipFrames))
        else:
            skipFrames = 0

        info = [0]

        nnStack = tifffile.imread(file)
        for frame in range(nnStack.shape[0]):
            if (frame % (skipFrames+1)) != 0:
                continue
            info.append(info[-1] + np.max(nnStack[frame]))
        infos[file] = info[1::]
        elapsed[file] = SmartMicro.NNio.loadTifStackElapsed(file, nnStack.shape[0], skipFrames=skipFrames)

        print(len(elapsed[file]))
    return infos, elapsed


def getATSData(directory: str):
    # get subfolders
    folders = sorted(glob.glob(directory + '/**/*_ATS2'))
    print(folders)
    mitoTimes = {}
    mitoBleachs = {}
    infos = {}

    # Go through all folders in this directory and look for image files
    print('\n'.join(folders))
    for folder in tqdm(folders):

        files = sorted(glob.glob(folder + '/img_*.tif*'))
        mitoFiles = []
        drpFiles = []
        nnFiles = []
        mitoTime = []
        mitoBleach = []
        info = [0]
        # Go through every file (numerical order) and categorize to the different image types
        print(files[0])
        dataOrder = SmartMicro.NNio.dataOrderMetadata(files[0], write=False)
        for file in files:
            splitStr = re.split(r'img_channel\d+_position\d+_time', file)
            if len(splitStr) > 1:
                splitStr = re.split(r'_', splitStr[1])
                frameNum = int(splitStr[0])
                if splitStr[-1][0:4] == 'prep':
                    pass
                elif splitStr[1][0:2] == 'nn':
                    nnFiles.append(os.path.join(folder, file))
                elif frameNum % 2 and dataOrder:
                    mitoFiles.append(os.path.join(folder, file))
                elif frameNum % 2 and not dataOrder:
                    drpFiles.append(os.path.join(folder, file))
                elif dataOrder:
                    drpFiles.append(os.path.join(folder, file))
                elif not dataOrder:
                    mitoFiles.append(os.path.join(folder, file))
        # For every file in the mito channel get the respective data
        i = 0
        for file in drpFiles:
            with tifffile.TiffFile(file) as tif:
                try:
                    mitoTime.append(json.loads(tif.imagej_metadata['Info'])['ElapsedTime-ms'])
                except TypeError:
                    mdInfoDict = xmltodict.parse(tif.ome_metadata)
                    mitoTime.append(float(mdInfoDict['OME']['Image']['Pixels']['Plane']['@DeltaT']))
            # img = tifffile.imread(file)
            # mitoBleach.append(np.mean(img))
            nn = tifffile.imread(nnFiles[i])
            info.append(info[-1] + np.max(nn))
            i = i + 1
        mitoTimes[folder] = mitoTime
        mitoBleachs[folder] = mitoBleach
        infos[folder] = info[1::]

    return mitoTimes, infos, mitoBleachs, folders

def makeStepSeries(dataY: list, dataX: list = None) -> tuple:
    if dataX is None:
        dataX = list(range(len(dataY)))
    newDataY = []
    newDataX = []
    for i in range(len(dataY)-1):
        newDataY.append(dataY[i])
        newDataX.append(dataX[i])
        if dataY[i+1] != dataY[i]:
            newDataY.append(dataY[i])
            newDataX.append(dataX[i + 1])
    return newDataY, newDataX

def getFileName(folder: str, saveFolder: str):
    if os.path.basename(saveFolder[:-1]) == 'ffbinary_data':
        file = (os.path.dirname(folder) + '/sample1_cell_' + folder[-6] +
             '_MMStack_Pos0_combine_nn_ffbinary.ome.tif')
    else:
        file = folder[0:-8] + '_nn.ome.tif'

    return file

def saveObj(obj, name, saveFolder=os.getcwd()):
    fileDirectory = saveFolder + name + '.pkl'
    try:
        with open(fileDirectory, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    except FileNotFoundError:
        print(os.path.dirname(saveFolder + name))
        os.mkdir(os.path.dirname(saveFolder + name))
        with open(fileDirectory, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def loadObj(name, saveFolder=os.getcwd() ):
    with open(saveFolder + name + '.pkl', 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    main()
