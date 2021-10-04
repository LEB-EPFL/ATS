from layeris.layer_image import LayerImage
import numpy as np


def screen_images(image1: np.ndarray, image2: np.ndarray, get_array=True):
    ''' Screen two images in gray and red together for plotting'''
    image = image1/np.max(image1)
    image = np.stack([image, image, image], axis=2)
    image = LayerImage.from_array(image)
    image2 = image2/np.max(image2)
    image2 = np.stack([image2, np.zeros_like(image2),
                       np.zeros_like(image2)], axis=2)
    image.screen(image2)
    if get_array:
        return image.get_image_as_array()
    else:
        return image
