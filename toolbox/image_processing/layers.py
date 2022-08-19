from layeris.layer_image import LayerImage
import numpy as np
import matplotlib.colors as mplcolors
from matplotlib import cm


def screen_stacks(stack1: np.ndarray, stack2: np.ndarray, colormap1:cm.ScalarMappable = None, 
                  colormap2: np.ndarray = None):
    if len(stack1.shape) == 3:
        result = np.ndarray(list(stack1.shape) + [4])
    else:
        result = np.ndarray(stack1.shape)
    for id in range(stack1.shape[0]):
        result[id] = screen_images(stack1[id], stack2[id], colormap1=colormap1, colormap2=colormap2)
    return result


def screen_images(image1: np.ndarray, image2: np.ndarray, colormap1: cm.ScalarMappable = None,
                  colormap2: cm.ScalarMappable = None, get_array=True):
    ''' Screen two images in gray and red together for plotting'''
    if len(image1.shape) == 2:
        if colormap1 is None:
            image1 = np.stack([image1, image1, image1, np.ones_like(image1)], axis=2)
        else:
            image1 = colormap1(image1)
    image1 = LayerImage.from_array(image1)
    
    if len(image2.shape) == 2:
        if colormap2 is None:
            image2 = np.stack([image2, np.zeros_like(image2),
                            np.zeros_like(image2), np.ones_like(image2)], axis=2)
        else:
            # Does not work if input is uint8 for now
            image2 = colormap2(image2)
            # if image1.image_data.shape[-1] == 3:
            #     image2 = image2[:, :, 0:3]
    
    image1.screen(image2)

    if get_array:
        image1 = image1.get_image_as_array()
    return image1
