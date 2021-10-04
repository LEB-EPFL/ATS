import os
import matplotlib.pyplot as plt

SETTINGS = {"facecolor": None, "edgecolor": None, "transparent": True}


def saveplt(name, folder='./', formats=None):
    if formats is None:
        formats = []

    def savefig(plotting_function):

        def save_wrapper(*args, **kwargs):
            os.makedirs(folder, exist_ok=True)
            plotting_function(*args, **kwargs)
            if 'png' in formats:
                plt.savefig(os.path.join(folder, name + '.png'), **SETTINGS)
            plt.savefig(os.path.join(folder, name + '.pdf'), **SETTINGS)
            print('figure saved')
            return plotting_function
        return save_wrapper
    return savefig
