""" Module that implements some custom functions that are often used with Qt GUIs. """

import numpy as np
import pyqtgraph
from matplotlib import cm


def getImageItemcolormap(name='hot', alpha=False, opacity=(0.3, 0.85)):
    """ Get a LUT that can be used with a QImageItem from a matplotlib LUT name """
    myColormap = cm.ScalarMappable()
    myColormap.set_cmap(name)
    myColormap = myColormap.get_array()
    myColormap = myColormap[:255]
    if alpha:
        alpha = np.linspace(opacity[0], opacity[1], 255)
        myColormap[:, 3] = alpha
    lut = (myColormap * 255).view(np.ndarray).astype(int)
    return lut


def getQtcolormap(name='hot'):
    """ Deprecated used to get a LUT consisting of QColors """
    myColormap = cm.ScalarMappable()
    myColormap.set_cmap(name)
    myColormap = myColormap.get_array()
    lut = (myColormap * 255).view(np.ndarray).astype(int)
    # lut = [qRgb(i[0], i[1], i[2]) for i in lut]
    return lut


def getImageViewcolormap(name='hot'):
    """This can be used in a pyqtgrah ImageView to set a colormap that's known from matplotlib

    Args:
        name (str, optional): name of the matplotlib colormap. Defaults to 'hot'.

    Returns:
        [type]: [description]
    """
    myColormap = cm.ScalarMappable()
    myColormap.set_cmap(name)
    myColormap = myColormap.get_array()
    pos = np.linspace(0, 1, num=myColormap.shape[0])
    alpha = (np.linspace(0, 1, num=myColormap.shape[0])*255).astype(int)
    alpha = np.append(alpha, [255, 255, 0])
    lut = (myColormap * 255).view(np.ndarray).astype(int)
    lut[:, 3] = alpha
    print(lut[:, 3])
    lut = pyqtgraph.ColorMap(pos, lut)
    return lut


if __name__ == '__main__':
    pass
