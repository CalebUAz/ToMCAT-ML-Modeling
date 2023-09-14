import mne
import numpy as np

def get_eeg_frequency_band_data(signals, look_back):
    # Frequency bands definitions
    bands = {
        'Delta': (1, 4),
        'Theta': (4, 8),
        'Alpha': (8, 14),
        'Beta': (14, 30),
        'Gamma': (30, 40)
    }
    sfreq = 500
    dataX = []
    
    # Load your EEG data into an mne object (this assumes signals is a 1D array)
    info = mne.create_info(ch_names=["eeg_channel"], sfreq=sfreq, ch_types=["eeg"])
    raw = mne.io.RawArray([signals], info)
    
    for i in range(0, len(signals) - look_back, look_back):
        combined_bands = []
        
        for band, (fmin, fmax) in bands.items():
            filtered_signals = raw.copy().filter(fmin, fmax).get_data()[0]
            a_band = filtered_signals[i:(i+look_back)]
            combined_bands.append(a_band)
        
        # Horizontally stack bands for this window
        window_data = np.hstack(combined_bands)
        dataX.append(window_data)

    return dataX