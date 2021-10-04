# Microscope settings
resizeParam = 56/81

# CAULOBACTER DATA
slow_folder = "C:/Users/stepp/Documents/02_Raw/Caulobacter_iSIM/slow/"
# samples = ["0", "1", "2", "3", "4", "6", "7", "8", "10", "11", "13"]
slow_samples = ["210526_FOV_5/Default0", "210526_FOV_5/Default1", "210526_FOV_5/Default2",
                "210526_FOV_5/Default3",
                "210602_FOV_7/Default0", "210602_FOV_7/Default1",
                "210602_FOV_7/Default2", "210602_FOV_7/Default3"]
slow_folders = [slow_folder + sample + '/' for sample in slow_samples]

fast_folder = "c:/Users/stepp/Documents/02_Raw/Caulobacter_iSIM/fast/"
fast_samples = ["FOV_5/Default0", "FOV_5/Default1", "FOV_5/Default2", "FOV_5/Default3",
                "FOV_6/Default0", "FOV_6/Default1", "FOV_6/Default2", "FOV_6/Default3",
                "210602_FOV_1/Default0", "210602_FOV_1/Default1", "210602_FOV_1/Default2",
                "210602_FOV_1/Default3"]
fast_folders = [fast_folder + sample + '/' for sample in fast_samples]

ats_folders = ['C:/Users/stepp/Documents/02_Raw/Caulobacter_iSIM/210414_06',
               'C:/Users/stepp/Documents/02_Raw/Caulobacter_iSIM/210416_02',
               'C:/Users/stepp/Documents/02_Raw/Caulobacter_iSIM/210416_03',
               'C:/Users/stepp/Documents/02_Raw/Caulobacter_iSIM/210421_21',
               'C:/Users/stepp/Documents/02_Raw/Caulobacter_iSIM/210615_10',
               'C:/Users/stepp/Documents/02_Raw/Caulobacter_iSIM/210615_11']

syncro_folders = [{
                   'folder': 'W:/Watchdog/bacteria/210506_weakSyncro/FOV_1/ATS_plot.pkl',
                   'crop': [4, 39], 'threshold': [80, 100], 'timeUnit': 'h',
                   'output': 'Suppl_Figures/syncro1_small.pdf'
                  },
                  {
                   'folder': 'c:/Users/stepp/Documents/05_Software/Analysis/'
                             '2106_Publication/syncro_2.pkl',
                   'crop': [1, 39], 'threshold': [90, 120], 'timeUnit': 'h',
                   'output': 'Suppl_Figures/syncro2_small.pdf'
                   # originally in 210512_3, now also in pkl file with newer SATS_GUI version
                  },
                  {
                   'folder': 'c:/Users/stepp/Documents/05_Software/Analysis/'
                             '2106_Publication/syncro_3.pkl',
                   'crop': [3, 100], 'threshold': [90, 120], 'timeUnit': 'h',
                   'output': 'Suppl_Figures/syncro3_small.pdf'
                   # originally in 210602_2
                  }]

image_data = {
              'folder': '//lebnas1.epfl.ch/microsc125/Watchdog/bacteria/'
                        '210602_syncro/FOV_2/Default',
              'frames': [5, 25, 45],
              'crop': [347, 120, 448, 221],
              'output': '//lebsrv2.epfl.ch/LEB_Perso/Willi-Stepp/ATS_Figures/Suppl_Figures/'
                        'syncro_frame_'
}

caulo_folders = {
    'slow': slow_folders,
    'fast': fast_folders,
    'ats': ats_folders,
    'syncro': syncro_folders,
    'images': image_data
}


slow_folder = '//lebnas1.epfl.ch/microsc125/iSIMstorage/Users/Dora/20201202_smartTest/analysis/'
slow_samples = ["cell0/", "cell3/", "cell4/"]
slow_folders = [slow_folder + sample for sample in slow_samples]

# The fast file are saved in stacks that are also not cropped very well. drp1 first
fast_folder = '//lebnas1.epfl.ch/microsc125/iSIMstorage/Users/Dora/20201213_fastMito/analysis/'
fast_samples = ['sample1/', 'sample2/', 'sample1_1/']  # , 'sample3/']
fast_folders = [fast_folder + sample for sample in fast_samples]
# exp = 'FOV_Int0_488nm_30mw_30pc_561nm_30mw_50pc_7/' # this has 3000 timepoint in tota

ats_folder = 'W:/Watchdog/microM_test/'
ats_samples = ['201208_cell_Int0s_30pc_488_50pc_561_band_6',
               '201208_cell_Int0s_30pc_488_50pc_561_band_4',
               '201208_cell_Int0s_30pc_488_50pc_561_band_5',
               '201208_cell_Int0s_30pc_488_50pc_561_band_10']
ats_folders = [ats_folder + sample for sample in ats_samples]


figure_1 = ''

mito_folders = {
    'slow': slow_folders,
    'fast': fast_folders,
    'ats': ats_folders
}

training_data = {'drp1': 'C:/Users/stepp/Documents/02_Raw/SmartMito/Drp1.h5',
                 'mito': 'C:/Users/stepp/Documents/02_Raw/SmartMito/Mito.h5',
                 'proc': 'C:/Users/stepp/Documents/02_Raw/SmartMito/Proc.h5'
                 }

manuscript_figures = '//lebsrv2.epfl.ch/LEB_PERSO/Willi-Stepp/ATS_Figures/'
