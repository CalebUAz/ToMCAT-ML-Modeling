import numpy as np
import mne
from utils.extract_EEG_features import get_eeg_frequency_band_data
from utils.extract_EKG_features import get_ekg_features
from utils.extract_GSR_features import get_gsr_features

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

def sliding_window_no_overlap(signals, valence, arousal, modality, use_wavelet, use_emd, **options):
    """Convert an array of X, Y values into a dataset matrix for a CNN"""

    look_back = options.pop('look_back', None)
    dataX, dataValenceScore, dataArousalScore = [], [], []

    if modality == 'eeg':
        # Convert EEG signals to frequency bands
        signals = get_eeg_frequency_band_data(signals, use_wavelet, use_emd)
    elif modality == 'ekg':
        signals = get_ekg_features(signals)
    elif modality == 'gsr':
        signals = get_gsr_features(signals)

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

def sliding_window_no_subject_overlap(signals, valence, arousal, subject_id, modality, use_wavelet, use_emd, **options):
    #No overlap between windows and No overlap between subjects. 
    """Convert an array of X, Y values into a dataset matrix for a CNN"""

    look_back = options.pop('look_back', None)
    dataX, dataValenceScore, dataArousalScore = [], [], []
    
    if modality == 'eeg':
        # Convert EEG signals to frequency bands
        signals = get_eeg_frequency_band_data(signals, use_wavelet, use_emd)
    elif modality == 'ekg':
        signals = get_ekg_features(signals)
    elif modality == 'gsr':
        signals = get_gsr_features(signals)

    unique_subjects = np.unique(subject_id)
    
    for subject in unique_subjects:
        subject_indices = np.where(subject_id == subject)[0]
        
        for i in range(subject_indices[0], subject_indices[-1] - look_back + 1, look_back):
            a = signals[i:(i+look_back)]
            dataX.append(a)
            dataValenceScore.append(valence[i:(i+look_back)])
            dataArousalScore.append(arousal[i:(i+look_back)])

    print("Arousal score labels:", np.unique(dataValenceScore), "Valence score labels:", np.unique(dataArousalScore))

    # Generate a single most frequently occurring label for each window
    dataValenceScore = [np.argmax(np.bincount(x)) for x in dataValenceScore]
    dataArousalScore = [np.argmax(np.bincount(x)) for x in dataArousalScore]

    return np.array(dataX), np.array(dataValenceScore), np.array(dataArousalScore)
