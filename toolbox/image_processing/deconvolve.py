
from skimage import restoration
import numpy as np
import cv2
from skimage import io
import matplotlib.pyplot as plt
from . import prepare

def main():
    from Analysis.tools import get_files
    folder = 'W:/Watchdog/microM_test/201208_cell_Int0s_30pc_488_50pc_561_band_5'
    files, _ = get_files(folder)

    for struct_file, nn_file in zip(files['network'], files['nn']):
        struct_img = io.imread(struct_file)
        nn_img = io.imread(nn_file)
        img = full_richardson_lucy(struct_img)
        # axs[3].imshow(nn_img)
        # plt.show()



def full_richardson_lucy(evt_image, intermediate:int=0):
    x, y = np.meshgrid(np.linspace(-3,3,11), np.linspace(-3,3,11))
    d = np.sqrt(x*x+y*y)
    sigma, mu = 4.0/2.3548, 0.0
    psf = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    psf = psf/np.sum(psf)  # *np.sum(evt_image)
    psf[psf < 0.000001] = 0

    evt_image = prepare_decon(evt_image)
    evt_image = evt_image.astype(np.uint16)

    if intermediate == True:
        return evt_image

    evt_image = cv2.medianBlur(evt_image, 5)
    evt_image = cv2.GaussianBlur(evt_image, (0,0), 2)
    if intermediate == 2:
        return evt_image
    evt_image = evt_image/65_535

    npad = np.ceil(3*sigma).astype(np.uint8)
    evt_image = cv2.copyMakeBorder(evt_image, npad, npad, npad, npad, cv2.BORDER_REPLICATE)
    evt_image = evt_image + 0.00001
    evt_decon = restoration.richardson_lucy(evt_image, psf)
    evt_decon = evt_decon[npad+1:-(npad*1)+1, npad+1:-(npad*1)+1]
    evt_decon = (evt_decon*65_535).astype(np.uint16)

    return evt_decon



if __name__ == '__main__':
    main()
