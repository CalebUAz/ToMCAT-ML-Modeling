import mne
import numpy as np

def get_eeg_frequency_band_data(signals):
    # Frequency bands definitions
    bands = {
        'Delta': (1, 4),
        'Theta': (4, 8),
        'Alpha': (8, 14),
        'Beta': (14, 30),
        'Gamma': (30, 40)
    }
    sfreq = 500
    signals = signals.T
    # Create channel names for MNE object
    n_channels = signals.shape[0]
    ch_names = ["eeg_channel_{}".format(i) for i in range(n_channels)]
    
    # Load your EEG data into an mne object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * n_channels)
    raw = mne.io.RawArray(signals, info)

    # List to store the filtered data for each band
    all_band_data = []
    
    # Apply band-pass filters for each frequency band and store in the list
    for band, (fmin, fmax) in bands.items():
        filtered_data = raw.copy().filter(fmin, fmax).get_data()
        all_band_data.append(filtered_data)
    
    # Concatenate all frequency bands side by side (along channels axis)
    stacked_data = np.concatenate(all_band_data, axis=0)
    
    return stacked_data.T