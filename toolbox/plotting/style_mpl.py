"""
    Make some special style options for matplotlib available with a clean and similar syntax
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
# import mplcyberpunk


SOLARIZED_COLORS = {
         "base03":  "#002B36",
         "base02":  "#073642",
         "base01":  "#586e75",
         "base00":  "#657b83",
         "base0":   "#839496",
         "base1":   "#93a1a1",
         "base2":   "#EEE8D5",
         "base3":   "#FDF6E3",
         "yellow":  "#B58900",
         "orange":  "#CB4B16",
         "red":     "#DC322F",
         "magenta": "#D33682",
         "violet":  "#6C71C4",
         "blue":    "#268BD2",
         "cyan":    "#2AA198",
         "green":   "#859900"
         }

PUBLICATION_COLORS = {
         'red':  '#d22a26',
         'darkblue': '#02353e',
         'lightblue': '#0293a4',
         'darkred': '#990000',  # '#ca5e70',
         'lightred': '#ff6250',  # 9e6bc4
         'green': '#8c9e46'
     }


DARK = {"03": SOLARIZED_COLORS["base03"],
        "02": SOLARIZED_COLORS["base02"],
        "01": SOLARIZED_COLORS["base01"],
        "00": SOLARIZED_COLORS["base00"],
        "0":  SOLARIZED_COLORS["base0"],
        "1":  SOLARIZED_COLORS["base1"],
        "2":  SOLARIZED_COLORS["base2"],
        "3":  SOLARIZED_COLORS["base3"]
        }

LIGHT = {"03": SOLARIZED_COLORS["base3"],
         "02": SOLARIZED_COLORS["base2"],
         "01": SOLARIZED_COLORS["base1"],
         "00": SOLARIZED_COLORS["base0"],
         "0":  SOLARIZED_COLORS["base00"],
         "1":  SOLARIZED_COLORS["base01"],
         "2":  SOLARIZED_COLORS["base02"],
         "3":  SOLARIZED_COLORS["base03"]
         }


def get_colors(style):
    if style == "solarized_dark":
        return DARK
    elif style == "solarized_light":
        return LIGHT


def set_mpl_font(size=10):
    # set Font

    mpl.rcParams['font.sans-serif'] = "arial"
    mpl.rcParams['font.family'] = "sans-serif"
    mpl.rcParams['font.size'] = size

    # MEDIUM_SIZE = size
    # SMALL_SIZE = size - 2
    # BIGGER_SIZE = size + 2

    # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    # plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    # plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def set_mpl_style(style: str):

    if "solarized" in style:
        if "dark" in style:
            rebase = DARK
        elif "light" in style:
            rebase = LIGHT
        else:
            print('You should specify light or dark for the solarized style')
            return None
        params = solarized_stylesheet(rebase)
        mpl.rcParams.update(params)
    elif "cyberpunk" in style:
        plt.style.use('cyberpunk')
    elif "publication" in style:
        plt.style.use("default")
        params = publication_stylesheet()
        mpl.rcParams.update(params)
    else:
        plt.style.use(style)


def publication_stylesheet():
    params = {

        "axes.prop_cycle": cycler('color', [PUBLICATION_COLORS['darkblue'],
                                            PUBLICATION_COLORS['green'],
                                            PUBLICATION_COLORS['red'],
                                            PUBLICATION_COLORS['lightblue'],
                                            PUBLICATION_COLORS['darkred'],
                                            PUBLICATION_COLORS['lightred']]),

        "savefig.facecolor": "None",
        "savefig.edgecolor": "None",
        "boxplot.medianprops.color": '#505050',
        "boxplot.boxprops.color": '#707070',
        "pdf.fonttype": 42
    }
    return params


def solarized_stylesheet(rebase):
    params = {"ytick.color": rebase["0"],  # 'k'
              "xtick.color": rebase["0"],  # 'k'
              "text.color": rebase["0"],  # 'k'
              "savefig.facecolor": rebase["02"],  # 'w'
              "patch.facecolor": SOLARIZED_COLORS["blue"],  # 'b'
              "patch.edgecolor": rebase["0"],  # 'k'
              "grid.color": rebase["03"],  # 'k'
              "figure.edgecolor": rebase["02"],  # 'w'
              "figure.facecolor": rebase["03"],  # '0.75'
              "axes.prop_cycle": cycler('color', [SOLARIZED_COLORS["blue"],
                                                  SOLARIZED_COLORS["green"],
                                                  SOLARIZED_COLORS["red"],
                                                  SOLARIZED_COLORS["cyan"],
                                                  SOLARIZED_COLORS["magenta"],
                                                  SOLARIZED_COLORS["orange"], rebase["0"]]),
              # ['b', 'g', 'r', 'c', 'm', 'y', 'k']
              "axes.edgecolor": rebase["03"],  # 'k'
              "axes.facecolor": rebase["02"],  # 'w'
              "axes.labelcolor": rebase["0"],  # 'k'

              "boxplot.whiskerprops.color": SOLARIZED_COLORS["base00"],
              "boxplot.whiskerprops.linewidth": .5,
              "boxplot.medianprops.color": SOLARIZED_COLORS["base00"],
              "boxplot.medianprops.linewidth": 1,
              "boxplot.boxprops.color": SOLARIZED_COLORS["base00"],
              "boxplot.boxprops.linewidth": 1,
              "boxplot.capprops.color": SOLARIZED_COLORS["base00"],
              "boxplot.capprops.linewidth": 1,

              }
    return params
