# -*- coding: utf-8 -*-
"""
Created on 201102

@author: stepp

Application to read a h5 file and save a tif from it at 2Hz to test the
watchdog on lebpc20
"""

import glob
import json
import os
import time

import h5py
import numpy as np
from skimage import io


def main():
    """Method to simulate microManager saving for testing network_watchdog"""

    dataPath = 'C:/Users/stepp/Documents/02_Raw/SmartMito/'  # nb: begin with /
    print('dataPath : ', dataPath)
    print()
    inputDataFilename1 = dataPath + 'Mito.h5'  # Mito
    inputDataFilename2 = dataPath + 'Drp1.h5'  # Drp1

    # inputDataFilename1 = dataPath + 'cell_8_Pos1_mito_741.tif'  # Mito
    # inputDataFilename2 = dataPath + 'cell_8_Pos1_drp1_741.tif'  # Drp1

    # inputDataFilename1 = dataPath + 'cell_8_mito_1024.tif'  # Mito
    # inputDataFilename2 = dataPath + 'cell_8_drp1_1024.tif'  # Drp1

    inputDataFilename1 = dataPath + 'cell_8_mito.tif'  # Mito
    inputDataFilename2 = dataPath + 'cell_8_drp1.tif'  # Drp1

    inputDataFilename1 = dataPath + 's3_c6_mito.tif'  # Mito
    inputDataFilename2 = dataPath + 's3_c6_drp1.tif'  # Drp1

    inputDataFilename1 = dataPath + 's6_c12_p0_mito.tif'  # Mito
    inputDataFilename2 = dataPath + 's6_c12_p0_drp1.tif'  # Drp1

    # inputData = dataPath + '180420_111.tif'

    inputData = dataPath + '180420_130.tif'

    mito = []
    drp = []

    if 'inputData' in locals():
        mitoAll = io.imread(inputData)
        mito = mitoAll[1::2]
        drp = mitoAll[0::2]
        print('single file interleaved')
    else:
        if inputDataFilename1[-3:] == '.h5':
            hFile = h5py.File(inputDataFilename1, 'r')
            mito = hFile.get('Mito')  # Mito
        else:
            mito = io.imread(inputDataFilename1)

        if inputDataFilename2[-3:] == '.h5':
            hFile = h5py.File(inputDataFilename2, 'r')
            drp = hFile.get('Drp1')
        else:
            drp = io.imread(inputDataFilename2)

    print(drp.dtype)
    print('Input : ', drp.shape)
    print('Input : ', np.shape(drp)[0])

    with open('./ATS_settings.json') as file:
        settings = json.load(file)[os.environ['COMPUTERNAME'].lower()]

    nasPath = os.path.join(settings['imageFolder'], 'python_saver/')

    # Make new folder there as microManager would
    dirs = glob.glob(nasPath + '*/')
    if len(dirs) > 0:
        num = int(dirs[-1][-4:-1])
    else:
        num = 0
    newFolder = os.path.join(nasPath, 'test_' + str(num+1).zfill(3))
    os.mkdir(newFolder)

    i = 0
    for item in range(0, 100):
        # for item in range(1, 2002, 1000):  # [1, 208]:  # range(150, 210):
        time1 = time.perf_counter()
        print(item)
        mitoPath = newFolder + '/img_channel000_position000_time' + \
            str((i*2)).zfill(9) + '_z000.tif'
        drpPath = newFolder + '/img_channel000_position000_time' + \
            str((i*2 + 1)).zfill(9) + '_z000.tif'

        # First write mito then drp as watchdog waits for drp image
        io.imsave(mitoPath, mito[item, :, :].astype(np.uint16),
                  check_contrast=False)
        time.sleep(0.05)
        io.imsave(drpPath, drp[item, :, :].astype(np.uint16),
                  check_contrast=False)
        time2 = time.perf_counter()
        time.sleep(np.max([0, .5 - (time2 - time1)]))
        i = i + 1

    time.sleep(3)

    # Clear the folder completely to have the same situation always
    print('deleting files')
    for file in os.listdir(newFolder):
        # if not f.endswith(".tiff"):
        #     continue
        os.remove(os.path.join(newFolder, file))
        time.sleep(0.001)
    print('Done')

    del inputData
    del inputDataFilename1
    del inputDataFilename2


if __name__ == '__main__':
    main()
