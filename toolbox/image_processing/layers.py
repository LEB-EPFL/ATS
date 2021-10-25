from layeris.layer_image import LayerImage
import numpy as np
import matplotlib.colors as mplcolors
from matplotlib import cm


def screen_images(image1: np.ndarray, image2: np.ndarray, colormap1: cm.ScalarMappable = None,
                  colormap2: cm.ScalarMappable = None, get_array=True):
    ''' Screen two images in gray and red together for plotting'''
    image = image1/np.max(image1)
    if colormap1 is None:
        image = np.stack([image, image, image], axis=2)
    else:
        image = colormap1(image1)
    image = LayerImage.from_array(image)
    image2 = image2/np.max(image2)
    if colormap2 is None:
        image2 = np.stack([image2, np.zeros_like(image2),
                           np.zeros_like(image2)], axis=2)
    else:
        image2 = colormap2(image2)
        image2 = image2[:, :, 0:3]
    image.screen(image2)
    if get_array:
        return image.get_image_as_array()
    else:
        return image
