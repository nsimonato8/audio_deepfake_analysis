"""
This module contains the code for the import of the audio file, and the relative transformation into a usable dataset.
"""
import pandas as pd
import librosa as lb
import os
import numpy as np

def prepare_raw_dataset(path_tracks: str, path_ground_truth: str, saving_path: str) -> None:
    """
    This function reads the dataset, file by file, and builds a Pandas DataFrame
    in which each row has bit_rate, label and values of the file. The index of the DataFrame
    is the track name. The data will then be printed to a .csv file.
    """
    res = []
    for p in os.listdir(path_tracks).sort():
        track, sr = lb.load(p)
        res.append((p, sr, track))

    res = np.vstack(res)
    tracks = pd.DataFrame(res, columns=["Filename", "SampleRate", "Track_np"])

    with open(path_ground_truth, "r") as f:
        ground_truth = [line.split(' ') for line in f.readlines()]
    ground_truth = pd.DataFrame(ground_truth, columns=["SpeakerID", "Filename", "-", "SystemID", "KEY"])

    dataset = pd.merge(tracks, ground_truth, on="Filename", how="inner")

    dataset.to_csv(saving_path)

    pass

def prepare_MFCC_dataset(df: pd.DataFrame) -> None:
    """
    This function reads the pandas DataFrame of the pre-processed dataset, and for each track
    computes the Mel-frequency spectrogram as a 2D numpy array. When finished, it writes the output
    to a .csv file.
    """
    pass