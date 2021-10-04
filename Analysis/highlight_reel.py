from tifffile.tifffile import FileSequence
from events import detect_events, filter_events, follow_events, get_event_frame
from data_handling import ATS_Data
import tools
from skimage import transform
import numpy as np
import pdb

import matplotlib.pyplot as plt
from toolbox.plotting import colormaps, saving
from toolbox.image_processing import prepare
from toolbox.image_processing import layers
# from layeris.layer_image import LayerImage
import os


def main(data):
    # data = load_dataset('caulo')
    prepare_data(data)
    detect_events(data)
    follow_events_batch(data)
    highlights = extract_event_highlight(data, save=True)
    plot_highlights(highlights, data)
    return data


def load_dataset(dataset: str) -> ATS_Data:
    data = ATS_Data()
    data.choose_dataset(dataset, 'highlight_reel')
    data.init_files('ats')
    return data


def prepare_data(data: ATS_Data):
    # pdb.set_trace()
    for idx, folder in enumerate(data.files_list):
        print(folder)
        # pdb.set_trace()
        folder.files['nn'] = transform.resize(folder.files['nn'],
                                              folder.files['peaks'].shape,
                                              preserve_range=True)
        folder.files['nn'] = folder.files['nn'].astype(np.uint8)

    return data


def detect_events(data: ATS_Data):
    for idx, files in enumerate(data.files_list):
        files.init_events(data.highlight_threshold)
    return data


def follow_events_batch(data: ATS_Data):
    for folder in data.files_list:
        follow_events(folder.events, folder.files['nn'], 100, data.hightlight_frame_size)


def extract_event_highlight(data: ATS_Data, save: bool = False):
    highlights = []
    image_id = 0
    for idx_folder, folder in enumerate(data.files_list):
        # if idx == 2:
        #     continue
        print(folder.folder)
        netw_imgs = folder.files['network']
        peak_imgs = folder.files['nn']
        for idx, event in folder.events.iterrows():
            # pdb.set_trace()
            frame = int(np.argmax(event.trace) + event.frame)
            pos = [int(event['weighted_centroid-0']), int(event['weighted_centroid-1'])]
            # netw_img = prepare.prepare_decon(netw_imgs[frame])
            netw_img = netw_imgs[frame]
            if data.dataset == 'caulo':
                netw_img = prepare.prepare_image(netw_img, background=1.01, median=1, gaussian=2)
            netw_subframe, _ = get_event_frame(netw_img, data.hightlight_frame_size,
                                               nn_image=peak_imgs[frame])
            peak_subframe, _ = get_event_frame(peak_imgs[frame], data.hightlight_frame_size,
                                               nn_image=peak_imgs[frame])
            # pdb.set_trace()
            netw_subframe = netw_subframe - np.min(netw_subframe)
            netw_subframe[netw_subframe < data.highlight_vmin] = data.highlight_vmin
            netw_subframe = netw_subframe - data.highlight_vmin
            if data.dataset == 'mito':
                netw_subframe = prepare.prepare_image(netw_subframe)
            # image = prepare.screen_images(netw_subframe, peak_subframe)
            image = layers.screen_images(netw_subframe, peak_subframe, get_array=False)
            highlight = {'image': image.get_image_as_array(),
                         'folder': folder.folder,
                         'frame': frame
                         }
            # pdb.set_trace()
            highlights.append(highlight)
            if save:
                directory = os.path.join(data.figure_folder, 'Suppl_Figures/Highlights',
                                         data.dataset)
                os.makedirs(directory, exist_ok=True)
                filename = data.dataset + '_' + str(image_id) + '.png'
                image.save(os.path.join(directory, filename))
                image_id += 1
    return highlights


def plot_highlights(highlights, data: ATS_Data):
    n_rows = int(np.ceil(len(highlights)/5))
    _, axs = plt.subplots(n_rows, 5)
    axs = axs.flatten()

    for idx, highlight in enumerate(highlights):
        image = highlight['image']

        axs[idx].imshow(image)
        axs[idx].set_xticks([])
        axs[idx].set_yticks([])
        axs[idx].axis('off')
    plt.tight_layout
    filename = data.dataset + '_highlights.pdf'
    plt.savefig(os.path.join(data.figure_folder + '/Suppl_Figures/Highlights', filename),
                **{"facecolor": None, "edgecolor": None, "transparent": True})
    plt.show()


if __name__ == '__main__':
    main()
