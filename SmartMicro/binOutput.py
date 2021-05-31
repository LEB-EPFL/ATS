# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:57:53 2020

@author: stepp

Application to write a value to a file in a remote location as fast as possible

"""

import os.path as ospath
import time
from datetime import datetime


def main():
    """ Use this to test a direct binary alternating value to the specified file. """
    path = "//lebnas1.epfl.ch/microsc125/Watchdog/"
    filename = 'binary_output.dat'
    fullFileDir = path + filename

    output = False
    output = bool(output)

    try:
        while True:
            time1 = time.perf_counter()
            file = open(fullFileDir, 'wb')
            file.write(bytearray(output))
            file.close()

            # This changes the value every some seconds
            if (time.perf_counter() % 3) < 1.5:
                output = False
                print('False False False')
            else:
                output = True
                print(output)
            time3 = time.perf_counter()
            time.sleep(0.5-(time3-time1))
    except KeyboardInterrupt:
        print('done')


def writeBin(output, printTime=0, path="//lebnas1.epfl.ch/microsc125/Watchdog/",
             filename='binary_output.dat'):
    """ Write a integer to a binary file that can be read by readBinNetwork.m from Matlab """

    if filename != '':
        fullFileDir = ospath.join(path, filename)
    else:
        fullFileDir = path

    file = open(fullFileDir, 'wb')
    file.write(bytearray(output))
    file.close()

    if printTime:
        now = datetime.now()
        currentTime = now.strftime("%H:%M:%S.%f")
        print(f"{output} written at {currentTime}")


if __name__ == "__main__":
    main()
