"""Module for outsourcing the preprocessing of raw data to prepare for neural network.
Rescale to 81 nm/px, background subtraction and contrast enhancement.

Returns:
    [type]: [description]
"""


import numpy as np
from skimage import exposure, filters, transform

from SmartMicro.ImageTiles import getTilePositionsV2


def prepareNNImages(mitoFull, drpFull, model):
    """Preprocess raw iSIM images before running them throught the neural network.

    Args:
        mitoFull ([type]): full frame of the mito data as numpy array
        drpFull ([type]): full frame of the drp data as numpy array
        nnImageSize ([type]): image size that is needed for the neural network. Default is 128

    Returns:
        [type]: Returns a 3D numpy array that contains the data for the neural network and the
        positions dict generated by getTilePositions for tiling.
    """
    # Set iSIM specific values
    pixelCalib = 56  # nm per pixel
    sig = 121.5/81  # in pixel
    resizeParam = pixelCalib/81  # no unit
    try:
        nnImageSize = model.layers[0].input_shape[0][1]
    except AttributeError:
        nnImageSize = model
    positions = None

    # Preprocess the images
    if nnImageSize is None or drpFull.shape[1] > nnImageSize:
        # Adjust to 81nm/px
        mitoFull = transform.rescale(mitoFull, resizeParam)
        drpFull = transform.rescale(drpFull, resizeParam)
        # This leaves an image that is smaller then initially

        # gaussian and background subtraction
        mitoFull = filters.gaussian(mitoFull, sig, preserve_range=True)
        drpFull = (filters.gaussian(drpFull, sig, preserve_range=True)
                   - filters.gaussian(drpFull, sig*5, preserve_range=True))

        # Tiling
        if nnImageSize is not None:
            positions = getTilePositionsV2(drpFull, nnImageSize)
            contrastMax = 255
        else:
            contrastMax = 1

        # Contrast
        drpFull = exposure.rescale_intensity(
            drpFull, (np.min(drpFull), np.max(drpFull)), out_range=(0, contrastMax))
        mitoFull = exposure.rescale_intensity(
            mitoFull, (np.mean(mitoFull), np.max(mitoFull)),
            out_range=(0, contrastMax))

    else:
        positions = {'px': [(0, 0, drpFull.shape[1], drpFull.shape[1])],
                     'n': 1, 'overlap': 0, 'stitch': 0}

    # Put into format for the network
    if nnImageSize is not None:
        drpFull = drpFull.reshape(1, drpFull.shape[0], drpFull.shape[0], 1)
        mitoFull = mitoFull.reshape(1, mitoFull.shape[0], mitoFull.shape[0], 1)
        inputDataFull = np.concatenate((mitoFull, drpFull), axis=3)

        # Cycle through these tiles and make one array for everything
        i = 0
        inputData = np.zeros((positions['n']**2, nnImageSize, nnImageSize, 2), dtype=np.uint8())
        for position in positions['px']:
            inputData[i, :, :, :] = inputDataFull[:,
                                                  position[0]:position[2],
                                                  position[1]:position[3],
                                                  :]
            # inputData[i, :, :, 1] =  exposure.rescale_intensity(
            #    inputData[i, :, :, 1], (0, np.max(inputData[i, :, :, 1])),
            #    out_range=(0, 255))
            inputData[i, :, :, 0] = exposure.rescale_intensity(
                inputData[i, :, :, 0], (0, np.max(inputData[i, :, :, 0])),
                out_range=(0, 255))
            i = i+1
        inputData = inputData.astype('uint8')
    else:
        # This is now missing the tile-wise rescale_intensity for the mito channel.
        # Image shape has to be in multiples of 4, not even quadratic
        cropPixels = (mitoFull.shape[0] - mitoFull.shape[0] % 4,
                      mitoFull.shape[1] - mitoFull.shape[1] % 4)
        mitoFull = mitoFull[0:cropPixels[0], 0:cropPixels[1]]
        drpFull = drpFull[0:cropPixels[0], 0:cropPixels[1]]

        positions = getTilePositionsV2(mitoFull, 128)
        mitoFull = mitoFull.reshape(1, mitoFull.shape[0], mitoFull.shape[0], 1)
        drpFull = drpFull.reshape(1, drpFull.shape[0], drpFull.shape[0], 1)
        inputData = np.stack((mitoFull, drpFull), 3)


    return inputData, positions
