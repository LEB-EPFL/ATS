import data_locations
from SmartMicro.NNio import loadTifFolder
import matplotlib.pyplot as plt
import matplotlib.image
import tools
import numpy as np
import h5py
from tqdm import tqdm
from toolbox.plotting import colormaps, saving
import os


def main():
    extract_training_data()
    plt.show()


folder = os.path.join(data_locations.manuscript_figures, 'Suppl_Figures', 'NN')


@saving.saveplt('frames', folder=folder, formats=['pdf'])
def extract_training_data():
    frame_list = [1, 2340, 5093, 7939, 10334, 12234, 14234]
    file_list = list(data_locations.training_data.values())
    cmaps = [colormaps.black_to_red(), 'gray', 'viridis']
    _, axs = plt.subplots(len(file_list), len(frame_list),
                          figsize=(12, 8))
    plt.tight_layout()
    for row, file in enumerate(file_list):
        print(file)
        file_handle = h5py.File(file, 'r')
        dataset = file_handle[list(file_handle.keys())[0]]
        for column, frame in tqdm(enumerate(frame_list)):
            image = dataset[frame]
            axs[row, column].imshow(image, cmap=cmaps[row])
            axs[row, column].set_xticks([])
            axs[row, column].set_yticks([])
            axs[row, column].axis('off')


def save_decon_image():
    data = data_locations.caulo_folders['images']

    images = loadTifFolder(data['folder'], resizeParam=data_locations.resizeParam)

    fig, axs = plt.subplots(1, 3)

    for idx, frame in enumerate(data['frames']):
        # bact_image = images[1][frame].astype(np.uint16)
        bact_image = images[1][frame].astype(np.uint16)[data['crop'][0]:data['crop'][2],
                                                        data['crop'][1]:data['crop'][3]]
        bact_image = tools.deconvolve(bact_image, intermediate=2)
        matplotlib.image.imsave(data['output'] + str(frame) + '.png', bact_image, cmap='gray')
        axs[idx].imshow(bact_image, cmap='gray')

    plt.show()


if __name__ == "__main__":
    main()
