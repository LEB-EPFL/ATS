from matplotlib.colors import LinearSegmentedColormap

def black_to_red(num_colors=255):
    colors = [(0, 0, 0), (1, 0, 0)] # first color is black, last is red
    black_to_red = LinearSegmentedColormap.from_list(
        "Custom", colors, N=num_colors)
    return black_to_red