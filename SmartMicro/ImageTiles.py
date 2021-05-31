"""This is a script to test out image slicing using he image_slicer module
        in Python. Attention, the newest version of Pillow was downgraded when
        installing image_slicer using pip
   """

import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from skimage import io


def calculatePixel(posMN, overlap, targetSize, shape):
    """Get the corresponding pixel start end x/y values to the tile defined by row/column in posMN.

    Args:
        posMN ([type]): tile as in row/column
        overlap ([type]): overlap between tiles as defined by getTilePositions
        targetSize ([type]): Size of the tiles
        shape ([type]): shape of the tiled image

    Returns:
        [type]: tuple of start/end x/y pixels of the tile
    """
    posXY = (int(posMN[0]*(targetSize-overlap)), int(posMN[1]*(targetSize-overlap)),
             int(posMN[0]*(targetSize-overlap) + targetSize),
             int(posMN[1]*(targetSize-overlap) + targetSize))

    # shift the last one if it goes over the edge of the image
    if posMN[1]*(targetSize-overlap) + targetSize > shape[1]:
        shift = int(posMN[1]*(targetSize-overlap) + targetSize) - shape[1]
        posXY = (posXY[0], posXY[1] - shift, posXY[2], posXY[3] - shift)
        # print('Shifted vert for ', shift, 'pixel')

    if posMN[0]*(targetSize-overlap) + targetSize > shape[0]:
        shift = int(posMN[0]*(targetSize-overlap) + targetSize) - shape[0]
        posXY = (posXY[0] - shift, posXY[1], posXY[2] - shift, posXY[3])
        # print('Shifted hor for ', shift, 'pixel')

    return posXY


def getTilePositions(image, targetSize=128):
    """Generate tuples with the positions of tiles to split up an image with
    an overlap. Calculates the number of tiles in a way that allows for only
    full tiles to be needed.

    Args:
        filePath (PIL image): Image.open of a tiff file. Should be square and
        ideally from the geometric series (128, 256, 512, 1024, etc)
        targetSize (int, optional): target square size. Defaults to 128.

    Returns:
        [type]: dict with
        'posMN': tiles in row/column
        'px': tiles in pixels
        'overlap': overlap used between two tiles
        'numberTiles': number of tiles
        'stitch': number of pixels that will be discarded when stitching
    """

    # Check for the smallest overlap that gives a good result
    numberTiles = int(image.width/targetSize)+1
    cond = False

    while not cond and numberTiles < targetSize:
        overlap = ((numberTiles*targetSize-image.width)/numberTiles-1)
        if not overlap % 1:
            cond = True
        else:
            numberTiles = numberTiles + 1

    # For nxn tiles calculate the pixel positions considering the overlap
    numberTileRange = [range(0, numberTiles)]*2
    positions = {'mn': tuple(itertools.product(*numberTileRange)), 'px': [],
                 'overlap': overlap, 'stitch': int(overlap/2), 'n': numberTiles}

    for position in positions['mn']:
        positionXY = calculatePixel(position, overlap, targetSize, image.shape)
        positions['px'].append(positionXY)

    return positions


def getTilePositionsV2(image, targetSize=128):
    """Generate tuples with the positions of tiles to split up an image with
    an overlap. Calculates the number of tiles in a way that allows for only
    full tiles to be needed.

    Args:
        filePath (PIL image): Image.open of a tiff file. Should be square and
        ideally from the geometric series (128, 256, 512, 1024, etc)
        targetSize (int, optional): target square size. Defaults to 128.

    Returns:
        [type]: [description]
    """
    # Check for the smallest overlap that gives a good result
    numberTiles = int(image.shape[0]/targetSize)+1
    cond = False
    minOverlap = 35

    while not cond and numberTiles < targetSize and numberTiles > 1:
        overlap = ((numberTiles*targetSize-image.shape[0])/numberTiles-1)
        overlap = overlap - 1 if overlap % 2 else overlap
        if int(overlap) >= minOverlap:
            overlap = int(overlap)
            cond = True
        else:
            numberTiles = numberTiles + 1

    # For nxn tiles calculate the pixel positions considering the overlap
    numberTileRange = [range(0, numberTiles)]*2
    positions = {'mn': tuple(itertools.product(*numberTileRange)), 'px': [],
                 'overlap': overlap, 'stitch': int(overlap/2), 'n': numberTiles}

    for position in positions['mn']:
        positionXY = calculatePixel(position, overlap, targetSize,
                                    image.shape)
        positions['px'].append(positionXY)

    return positions


def stitchImage(data, positions, channel=0):
    """ stitch an image back together that has been tiled by NNfeeder.prepareNNImages """
    stitch = positions['stitch']
    # This has to take into account that the last tiles are sometimes shifted
    stitchedImageSize = positions['px'][-1][-1]
    stitchedImage = np.zeros([stitchedImageSize, stitchedImageSize])
    stitch1 = None if stitch == 0 else -stitch
    i = 0
    for position in positions['px']:
        stitchedImage[position[0]+stitch:position[2]-stitch,
                      position[1]+stitch:position[3]-stitch] = \
            data[i, stitch:stitch1, stitch:stitch1, channel]
        i = i + 1
    return stitchedImage


def main():
    """ Main method testing the tiling mechanism """
    path = '//lebnas1.epfl.ch/microsc125/Watchdog/test_image/Default'
    # fileName = 'img_channel000_position000_time000000000_z000.tif'
    fileName = 'moon.tif'
    filePath = path + '/' + fileName
    image = io.imread(filePath)

    positions = getTilePositionsV2(image, 128)

    plt.figure()
    axes = plt.axes()
    plt.draw()

    for position in positions['px']:
        setWidth = 0.5
        setColor = [0.7, 0.7, 0.7]
        Axes.axhline(axes, y=position[0], linewidth=setWidth, color=setColor)
        Axes.axvline(axes, x=position[1], linewidth=setWidth, color=setColor)
        Axes.axhline(axes, y=position[2], linewidth=setWidth, color=setColor)
        Axes.axvline(axes, x=position[3], linewidth=setWidth, color=setColor)

    numberTiles = positions['numberTiles']
    overlap = positions['overlap']
    print(numberTiles)
    print(numberTiles*(128-overlap)+overlap)

    plt.imshow(image)
    plt.pause(0.1)
    plt.show()


if __name__ == "__main__":
    main()
