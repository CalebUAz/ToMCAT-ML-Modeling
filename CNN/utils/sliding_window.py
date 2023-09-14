import numpy as np
import mne

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

    for i in range(0, len(signals) - look_back, look_back):  # Skip by 'look_back' for non-overlapping window
        if modality == 'eeg':
            sfreq = 500
            # Frequency bands definitions
            bands = {
                'Delta': (1, 4),
                'Theta': (4, 8),
                'Alpha': (8, 14),
                'Beta': (14, 30),
                'Gamma': (30, 40)
            }

            # Create channel names for MNE object
            n_channels = signals.shape[0]
            ch_names = ["eeg_channel_{}".format(i) for i in range(n_channels)]
            
            # Load your EEG data into an mne object
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * n_channels)
            raw = mne.io.RawArray(signals, info)

            # Apply band-pass filters for each frequency band and append to dataX
            for band, (fmin, fmax) in bands.items():
                filtered_signals = raw.copy().filter(fmin, fmax).get_data()
                
                # Window sliding for each channel's filtered data
                for channel_data in filtered_signals:
                    for i in range(0, len(channel_data) - look_back, look_back):
                        a_band = channel_data[i:(i+look_back)]
                        dataX.append(a_band.get_data())
                        dataValenceScore.append(valence[i:(i+look_back)])
                        dataArousalScore.append(arousal[i:(i+look_back)])
        else:
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
