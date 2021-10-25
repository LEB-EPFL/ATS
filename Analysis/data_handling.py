from dataclasses import dataclass, field
from enum import Enum
import data_locations
import os
from typing import List
import glob
import re
from SmartMicro import NNio
import tools
from toolbox.plotting import unit_info
import tifffile
import pandas as pd
import pdb
from events import detect_events, filter_events


# TODO: Get the order from the Files in the folder and also write them to the DAtaset so we can
# use it here to load the data in the correct way.
@dataclass
class ImageFiles:
    mode: str
    folder: os.PathLike
    files: field(default_factory=list)
    times: field(default_factory=list)
    event_options: dict
    events: pd.DataFrame = pd.DataFrame()
    stack: bool = False
    decon_id: str = 'None'

    def __init__(self, ats_data, mode, folder, decon_id):
        self.mode = mode
        self.event_options = {'detection_thr': ats_data.detection_threshold,
                              'observation_time': ats_data.highlight_observation_time,
                              'min_distance': ats_data.min_event_distance}
        self.folder = folder
        self.decon_id = decon_id
        if decon_id.lower() == 'none':
            self.files, self.stack = get_files(folder)
        else:
            self.files, self.stack = get_files(folder, decon_id)

        if self.stack:
            self.times = NNio.loadTifStackElapsed(self.stack)
        else:
            self.times = tools.get_times(self.files['peaks'])
            self.files['network'], self.files['peaks'], self.files['nn'], self.files['decon'] = \
                NNio.loadTifFolder(folder, resizeParam=unit_info.sim_to_isim_factor,
                                   order=1, outputs=['decon'], decon_id=decon_id)
        print(self.files['network'])

    def init_events(self, threshold=None):
        if threshold is None:
            threshold = self.event_options['detection_thr']
        # print(self.folder)
        self.events = detect_events(self.files['nn'], self.times, threshold)
        self.events['folder'] = self.folder
        self.events = filter_events(self.events, self.event_options['observation_time'],
                                    self.event_options['min_distance'])


@dataclass
class ATS_Data:
    dataset: str = None
    slow_folders: List[os.PathLike] = field(default_factory=list)
    fast_folders: List[os.PathLike] = field(default_factory=list)
    ats_folders: List[os.PathLike] = field(default_factory=list)
    files_list: List[ImageFiles] = field(default_factory=list)
    detection_threshold: int = None
    highlight_threshold: int = None
    save_file: os.PathLike = None
    obervation_time: int = None
    highlight_observation_time: int = None
    min_event_distance: int = 10
    highlight_vmin: int = None
    hightlight_frame_size: int = None
    decon_background: float = None
    analysis_folder: os.PathLike = "c:/Users/stepp/Documents/05_Software/Analysis/"
    figure_folder: os.PathLike = "//lebsrv2.epfl.ch/LEB_PERSO/Willi-Stepp/ATS_Figures"

    def choose_dataset(self, dataset: str, measurement: str) -> None:
        self.dataset = dataset
        if dataset.lower() == "caulo":
            self.slow_folders = data_locations.caulo_folders['slow']
            self.fast_folders = data_locations.caulo_folders['fast']
            self.ats_folders = data_locations.caulo_folders['ats']
            # self.ats_folders.append('C:/Users/stepp/Documents/02_Raw/Caulobacter_iSIM/210506_01')
            self.detection_threshold = 90
            self.highlight_threshold = 100
            self.save_file = os.path.join(self.analysis_folder, measurement, 'caulo_events')
            self.observation_time = 60*60  # seconds
            self.highlight_observation_time = 90*60
            self.highlight_vmin = 1
            self.hightlight_frame_size = 30
            self.constriction_decon_background = 80
            self.highlight_decon_background = 80
        elif dataset.lower() == "mito":
            self.slow_folders = data_locations.mito_folders['slow']
            self.fast_folders = data_locations.mito_folders['fast']
            self.ats_folders = data_locations.mito_folders['ats']
            self.detection_threshold = 80
            self.highlight_threshold = 100
            self.save_file = os.path.join(self.analysis_folder, measurement, 'mito_events')
            self.observation_time = 20  # seconds
            self.highlight_observation_time = 60
            self.highlight_vmin = 8
            self.hightlight_frame_size = 30
            self.constriction_decon_background = 0.85
            self.highlight_decon_background = 0.92

    def init_files(self, mode, decon_id='img_*_decon*') -> List[ImageFiles]:
        self.files_list = []
        if mode.lower() == 'ats':
            folders = self.ats_folders
        elif mode.lower() == 'fast':
            folders = self.fast_folders
        elif mode.lower() == 'slow':
            folders = self.slow_folders
        for folder in folders:
            self.files_list.append(ImageFiles(self, mode, folder, decon_id=decon_id))
        return self.files_list


def get_files(folder, decon_id='img_*_decon*'):
    stack = False
    print(folder)
    if os.path.isfile(folder + '/img_channel001_position000_time000000000_z000.tif'):
        bact_filelist = sorted(glob.glob(folder + '/img_channel001*_z*'))
        ftsz_filelist = sorted(glob.glob(folder + '/img_channel000*.tif'))
        nn_filelist = sorted(glob.glob(folder + '/img_*_nn*'))
        decon_filelist = sorted(glob.glob(folder + '/img_*_decon*'))
    elif os.path.isfile(folder + '/img_channel000_position000_time000000000_z000.tif'):
        print('No channels here')
        filelist = sorted(glob.glob(folder + '/img_*.tif'))
        re_odd = re.compile(r".*time\d*[13579]_.*tif$")
        bact_filelist = [file for file in filelist if re_odd.match(file)]
        re_even = re.compile(r".*time\d*[02468]_.*")
        ftsz_filelist = [file for file in filelist if re_even.match(file)]
        nn_filelist = sorted(glob.glob(folder + '/img_*_nn*'))
        decon_filelist = sorted(glob.glob(folder + '/' + decon_id))
    else:
        print("Image stacks")
        files = sorted(glob.glob(folder + '*_crop.ome.tif'))[0]
        print(files)
        nn_file = files[:-8] + '_nn.ome.tif'
        decon_file = files[:-8] + '_decon.tiff'
        cropped_file = files
        ftsz_filelist, bact_filelist = NNio.loadTifStack(cropped_file)
        nn_filelist = tifffile.imread(nn_file)
        try:
            decon_filelist = tifffile.imread(decon_file)
        except FileNotFoundError:
            decon_filelist = False
            print('No decon file present')
        stack = cropped_file

    files = {'network': bact_filelist,
             'peaks': ftsz_filelist,
             'nn': nn_filelist,
             'decon': decon_filelist}

    return files, stack
