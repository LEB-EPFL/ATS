
import numpy as np
import cv2


def prepare_decon(image, background=0.85):
    temp_fft = np.fft.fftshift(np.fft.fft2(image))
    filter_zone = get_filter_zone(temp_fft)
    temp_fft[filter_zone] = 0
    filtered_img = np.abs(np.fft.ifft2(np.fft.fftshift(temp_fft)))

    if background < 3:
        ret, mask = cv2.threshold(filtered_img.astype(np.uint16), 0, 1,
                                  cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        filtered_img = filtered_img - ret*background  # 80
    else:
        filtered_img = filtered_img - background
    filtered_img[filtered_img < 0] = 0
    return filtered_img


def get_filter_zone(temp_fft, y_range_param=15, x_range_param=100):
    height, width = temp_fft.shape
    x_center = round(1018*width/2048)
    y_center = round(360*height/2048)
    x_center2 = round(width-x_center)
    y_center2 = round(height-y_center)
    x_range = round(x_range_param*(width/2048))
    y_range = round(y_range_param*(height/2048))
    filter_zone = np.zeros(temp_fft.shape, dtype=bool)
    filter_zone[(y_center-y_range):(y_center+y_range),
                (x_center-x_range):(x_center+x_range)] = True
    filter_zone[(y_center2-y_range):(y_center2+y_range),
                (x_center2-x_range):(x_center2+x_range)] = True
    return filter_zone


def prepare_image(image, background=0.85, median=3, gaussian=1.5):
    prep_img = prepare_decon(image, background).astype(np.uint16)
    prep_img = cv2.medianBlur(prep_img, median)
    prep_img = cv2.GaussianBlur(prep_img, (0, 0), gaussian)
    return prep_img
