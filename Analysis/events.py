import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from networkx.algorithms.assortativity.correlation import \
    numeric_assortativity_coefficient
from scipy.spatial.distance import pdist, squareform
from skimage import measure
import pdb
import tools


def detect_events(nn_filelist, times, detection_threshold):
    all_events = pd.DataFrame()
    for idx in range(nn_filelist.shape[0]):
        nn_image = nn_filelist[idx]
        # Check if there are events in this frame
        nn_image = nn_image*(nn_image > 20)
        labels = tools.distance_watershed(nn_image, sigma=0.05)
        fission_props = measure.regionprops_table(labels, intensity_image=nn_image,
                                                  properties=['label', 'max_intensity',
                                                              'weighted_centroid'])
        # Get rid of the events that are not high enough in nn value
        fission_props = pd.DataFrame(fission_props)
        fission_props = fission_props[fission_props['max_intensity'] > detection_threshold]
        # Record the frame that we are at
        fission_props['frame'] = idx
        fission_props['time'] = times[idx]
        all_events = all_events.append(fission_props, ignore_index=True, sort=False)
    return all_events


def filter_events(all_events, observation_time, min_distance):
    # Filter all these events so we get unique events
    distances = pdist(all_events[['weighted_centroid-0', 'weighted_centroid-1']],
                      metric='euclidean')
    distances = squareform(distances)
    events = pd.DataFrame()
    skip_events = []
    for (idx, event), distance in zip(all_events.iterrows(), distances):
        if idx not in skip_events:
            events_in_future = all_events[all_events.index > idx]
            events_in_observation_time = events_in_future[events_in_future['time'] < event['time']
                                                          + observation_time*1000]
            distance = distance[events_in_observation_time.index]
            events_to_close = events_in_observation_time[distance < min_distance]
            skip_events.extend(events_to_close.index)
            event['appearances'] = len(events_to_close.index) + 1
            if len(events_to_close.index) > 0:
                event['max_intensity'] = np.max([np.max(events_to_close['max_intensity']),
                                                event['max_intensity']])
            events = events.append(event, ignore_index=True, sort=False)
    return events


def follow_events(all_events, nn_images, frames, frame_size):  # put frame_size = 11 later
    ''' Follow up on one event and record the neural network data over time.'''
    all_events['trace'] = None
    for idx, event in all_events.iterrows():
        trace = np.zeros(frames)
        pos = [int(event['weighted_centroid-0']), int(event['weighted_centroid-1'])]
        for frame in range(frames):
            if int(event['frame']) + frame > nn_images.shape[0] - 1:
                break
            nn_image = nn_images[int(event['frame']) + frame]
            nn_subframe, pos = get_event_frame(nn_image, frame_size, nn_image, pos)
            trace[frame] = np.max(nn_subframe)
        all_events.at[idx, 'trace'] = trace
        if idx % 5 == 0:
            plt.plot(trace)
    return all_events


def get_event_frame(bact_image, frame_size, nn_image=None, pos=None):

    if pos is None:
        pos = list(zip(*np.where(nn_image == np.max(nn_image))))[0]
    max0 = pos[0] + frame_size + 1
    max1 = pos[1] + frame_size + 1

    if max0 > bact_image.shape[0] or max1 > bact_image.shape[1]:
        return False, False

    frame = bact_image[pos[0] - frame_size:pos[0] + frame_size + 1,
                       pos[1] - frame_size:pos[1] + frame_size + 1]

    return frame, pos
