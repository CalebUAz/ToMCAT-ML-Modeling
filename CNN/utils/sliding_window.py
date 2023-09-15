import numpy as np
import mne
from utils.extract_EEG_features import get_eeg_frequency_band_data

def sliding_window(signals, valence, arousal, **options):
    """Convert an array of X, Y values into a dataset matrix for and CNN"""
    
    look_back = options.pop('look_back', None)
    dataX, dataValenceScore, dataArousalScore = [], [], []
    for i in range(len(signals) - look_back - 1):
        a = signals[i:(i+look_back)]
        dataX.append(a)
        dataValenceScore.append(valence[i:(i+look_back)])
        dataArousalScore.append(arousal[i:(i+look_back)])

    # Generate a single most frequently occurring label for each window
    dataValenceScore = [np.argmax(np.bincount(x)) for x in dataValenceScore]
    dataArousalScore = [np.argmax(np.bincount(x)) for x in dataArousalScore]

    return np.array(dataX), np.array(dataValenceScore), np.array(dataArousalScore)

def sliding_window_no_overlap(signals, valence, arousal, modality, **options):
    """Convert an array of X, Y values into a dataset matrix for a CNN"""

    look_back = options.pop('look_back', None)
    dataX, dataValenceScore, dataArousalScore = [], [], []

    if modality == 'eeg':
        # Convert EEG signals to frequency bands
        signals = get_eeg_frequency_band_data(signals)

    for i in range(0, len(signals) - look_back, look_back):  # Skip by 'look_back' for non-overlapping window
        a = signals[i:(i+look_back)]
        dataX.append(a)
        dataValenceScore.append(valence[i:(i+look_back)])
        dataArousalScore.append(arousal[i:(i+look_back)])

    # Generate a single most frequently occurring label for each window
    dataValenceScore = [np.argmax(np.bincount(x)) for x in dataValenceScore]
    dataArousalScore = [np.argmax(np.bincount(x)) for x in dataArousalScore]

    return np.array(dataX), np.array(dataValenceScore), np.array(dataArousalScore)


def sliding_window_get_sub_id(subject_id, **options):
    """Convert an array of X, Y values into a dataset matrix for a CNN"""

    look_back = options.pop('look_back', None)
    dataX =  []

    for i in range(0, len(subject_id) - look_back, look_back):  # Skip by 'look_back' for non-overlapping window
        a = subject_id[i:(i+look_back)]
        dataX.append(a)

    # Generate a single most frequently occurring subject id for each window
    dataX = [np.argmax(np.bincount(x)) for x in dataX]

    return np.array(dataX)
