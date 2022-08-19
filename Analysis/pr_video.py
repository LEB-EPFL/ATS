import tifffile
import matplotlib.pyplot as plt
import imageio
import ffmpeg
import os
import pyclesperanto_prototype as cle
from toolbox.image_processing import layers
from matplotlib.colors import LinearSegmentedColormap
from toolbox.image_processing.overlay import Overlay
import numpy as np

# for the interactive window:
# %load_ext autoreload
# %autoreload 2
# import os
# os.chdir("C:/Users/stepp/Documents/Software/EDA/Analysis")
# import pr_video
# import matplotlib.pyplot as plt
# images, nn_images = pr_video.load_data()
# images = pr_video.prepare_data(images, nn_images)
# plt.imshow(images[0])
# import pyclesperanto_prototype as cle
# from toolbox.image_processing import layers
# btr = pr_video.black_to_red_cm()
# norm = plt.Normalize(0, 255)
# ready = layers.screen_stacks(norm(images), norm(nn_images), None, btr)

# Annotation
# import napari
# viewer = napari.Viewer()
# viewer.add_image(ready)
# viewer.open('./PR_video/Annotations.csv')

# overlay = pr_video.add_circles(ready, viewer.layers["Annotations"].data)
# viewer.add_image(overlay)

# viewer.add_points(name="Annotations", ndim=3)
# viewer.layers["Annotations"].current_edge_width = 0.03
# viewer.layers["Annotations"].current_edge_color = "#ffffff64"
# viewer.layers["Annotations"].current_face_color = "#ffffff00"
# viewer.layers["Annotations"].current_size = 100


device = cle.select_device("GTX")
print("Used GPU: ", device)

file = r"\\lebnas1.epfl.ch\microsc125\iSIMstorage\Users\Willi\180420_drp_mito_Dora\sample1\sample1_cell_3_MMStack_Pos0_combine_decon.ome.tif"
crop = [70, 674]
nn_file = r"\\lebnas1.epfl.ch\microsc125\iSIMstorage\Users\Willi\180420_drp_mito_Dora\sample1\sample1_cell_3_MMStack_Pos0_combine_ffmodel_nn.ome.tif"


skip_frames = 3

def load_data(skip_frames = skip_frames):
    with tifffile.TiffFile(file) as tif:
        all_images = tif.asarray()
        images = all_images[::2]  # 2 for channels
        images = images[crop[0]:crop[1]]
        images = images[::skip_frames]
        for frame in range(images.shape[0]):
            images[frame] = images[frame]/images[frame].max()*255
        print(images.max())
        images = images.astype(np.uint8)
        del all_images

    with tifffile.TiffFile(nn_file) as tif:
        all_images = tif.asarray()
        nn_images = all_images[crop[0]:crop[1]]
        nn_images = nn_images[::skip_frames]
        nn_images = nn_images.astype(np.float64)
        nn_images = nn_images/np.max(nn_images)*255
        nn_images = nn_images.astype(np.uint8)
        del all_images
    
    nn_gpu = cle.push(nn_images)
    images_gpu = cle.push(images)

    x_scale = images.shape[1]/nn_images.shape[1]
    y_scale = images.shape[2]/nn_images.shape[2]
    nn_gpu = cle.scale(nn_gpu, factor_x=x_scale, factor_y=y_scale, auto_size=True)
    
    return images_gpu, cle.pull(nn_gpu)


def prepare_data(images, nn):
    images = cle.gaussian_blur(images, sigma_x=2, sigma_y=2)
    background = cle.gaussian_blur(images, sigma_x=100, sigma_y=100)
    mask = cle.greater_constant(images, constant=5)
    images = cle.multiply_images(images, mask)
    images = cle.divide_images(images, background)
    images = cle.pull(images)
    for frame in range(images.shape[0]):
        images[frame] = images[frame]/images[frame].max()*255
    images = images.astype(np.uint8)
    return images


def save_movie(images, out_file = './PR_video/video.gif', duration=1/24):
    imageio.mimsave(out_file, images, duration=duration)
    # stream = ffmpeg.input(out_file)
    folder = "c:/Users/stepp/Documents/Software/EDA/Analysis"
    mp4_file = '.'.join(out_file.split('.')[:-1]) + '.mp4'
    print(os.getcwd())
    print(mp4_file)
    os.system(f'C:/FFmpeg/bin/ffmpeg.exe -i {out_file} -y -vf scale=-4:720 -vcodec libx264 -pix_fmt yuv420p {mp4_file}')
    tif_file = '.'.join(out_file.split('.')[:-1]) + '.tif'
    tifffile.imsave(tif_file, images)


def add_circles(images, points):
    overlay = np.zeros(list(images.shape[:-1]) + [4], np.uint8)
    #sort the points
    points = points[np.argsort(points[:, 0])]
    context = Overlay(images.shape[1:-1])
    
    frame = points[0, 0]
    for idx in range(points.shape[0]):
        
        context.circle([points[idx, 2], points[idx, 1]], 40)
        frame = points[idx, 0]
        if idx == points.shape[0] - 1 or frame != points[idx + 1, 0]:
            overlay[int(frame), :, :] = context.get_image()
            context = Overlay(images.shape[1:-1])
    return overlay


def black_to_red_cm():
    cdict = {'red':  [[0.0, 0.0, 0.0],
                      [1., 1., 1.]],
            'green': [[0.0,  0.0, 0.0],
                      [1,  0.0, 0.0]],
            'blue':  [[0.0,  0.0, 0.0],
                      [1,  0.0, 0.0]],
            'alpha': [[0.0,  1., 1.],
                      [1.,  1., 1.]]}

    newcmp = LinearSegmentedColormap('testCmap', cdict, N=256)
    return newcmp