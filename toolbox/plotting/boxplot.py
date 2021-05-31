import matplotlib.pyplot as plt
import matplotlib as mpl


def boxplot(data,labels=None, showfliers=True, whis=None):
    old_x_tick_labelsize =mpl.rcParams["xtick.labelsize"]
    # print(mpl.rcParams["ytick.labelsize"])
    mpl.rcParams.update({"xtick.labelsize": 20})
    print(mpl.rcParams["xtick.labelsize"])
    # plt.rc('xtick', labelsize= 20)  # mpl.rcParams["axes.labelsize"])
    plt.boxplot(data, labels=labels, showfliers=showfliers, whis=whis)
    # plt.rc('xtick', labelsize=old_x_tick_labelsize)