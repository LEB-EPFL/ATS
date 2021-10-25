import ctypes
hllDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit" +
                       "\\CUDA\\v11.2\\bin\\cusolver64_11.dll")
from typing import Tuple  # noqa: E402
from flowdec import data as fd_data  # noqa: E402
from flowdec import restoration as fd_restoration  # noqa: E402
from skimage import io  # noqa: E402
from scipy import ndimage, signal  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from .prepare import prepare_decon  # noqa: E402
from dataclasses import dataclass  # noqa: E402
import cv2  # noqa: E402

import time  # noqa: E402
import os  # noqa: E402
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main():
    from Analysis.tools import get_files
    folder = 'W:/Watchdog/microM_test/201208_cell_Int0s_30pc_488_50pc_561_band_5'
    files, _ = get_files(folder)

    algo = fd_restoration.RichardsonLucyDeconvolver(2).initialize()
    kernel = make_kernel(io.imread(files['network'][0]), sigma=3.8)
    for struct_file in files['network']:
        struct_img = io.imread(struct_file)
        _, axs = plt.subplots(1, 2)
        # axs[0].imshow(struct_img)
        t0 = time.perf_counter()
        struct_img = prepare_decon(struct_img)
        img = richardson_lucy(struct_img, algo=algo, kernel=kernel).astype(np.uint16)
        print(time.perf_counter()-t0)
        axs[0].imshow(struct_img.astype(np.uint16))
        axs[1].imshow(img)
        # plt.show()


@dataclass
class CudaParams():
    """ Class for storing CUDA parameters to be passed to the flowdec function"""

    kernel: np.ndarray
    algo: fd_restoration.RichardsonLucyDeconvolver
    shape: Tuple[int] = (100, 100)
    ndim: int = 2
    sigma: float = 3.9/2.335
    prepared: bool = False
    background: float = 0.85
    after_gaussian: float = 2

    def __init__(self, shape: Tuple[int] = shape, sigma: float = sigma, ndim: int = ndim,
                 background: float = background, prepared: bool = prepared):
        super().__init__()
        self.background = background
        self.kernel = make_kernel(np.zeros(shape), sigma=sigma)
        self.algo = fd_restoration.RichardsonLucyDeconvolver(ndim).initialize()

    def to_dict(self):
        class_dict = {'sigma': self.sigma,
                      'background': self.background,
                      'after_gaussian': self.after_gaussian}
        return class_dict


def richardson_lucy(image, params=None, algo=None, kernel=None, prepared=True, background=None):
    original_data_type = image.dtype
    if params is not None:
        algo, kernel, prepared = params.algo, params.kernel, params.prepared
        background = params.background
        # print(params)
    else:
        if algo is None:
            algo = fd_restoration.RichardsonLucyDeconvolver(2).initialize()
        if kernel is None:
            kernel = make_kernel(image, sigma=3.9/2.355)
        if background is None:
            print('no background specified, using 0.85')
            background = 0.85
    if not prepared:
        image = prepare_decon(image, background)
    # print('sigma: ', kernel['sigma'])
    res = algo.run(fd_data.Acquisition(data=image, kernel=kernel['kernel']), niter=30).data
    return res.astype(original_data_type)


def make_kernel(image, sigma=1.6):
    kernel = {'kernel': np.zeros_like(image, dtype=float),
              'sigma': sigma}
    for offset in [0, 1]:
        kernel['kernel'][tuple((np.array(kernel['kernel'].shape) - offset) // 2)] = 1
    kernel['kernel'] = ndimage.gaussian_filter(kernel['kernel'], sigma=sigma)
    return kernel


if __name__ == '__main__':
    main()
