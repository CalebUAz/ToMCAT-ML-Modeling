import mne
import numpy as np

def get_eeg_frequency_band_data(signals):
    # Frequency bands definitions
    bands = {
        'Delta': (1, 4),
        'Theta': (4, 8),
        'Alpha': (8, 14),
        'Beta': (14, 30),
        'Gamma': (30, 50)  # Adjusted upper limit to 50Hz for gamma
    }
    
    sfreq = 500
    signals = signals.T
    n_channels = signals.shape[0]
    ch_names = ["eeg_channel_{}".format(i) for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * n_channels)
    raw = mne.io.RawArray(signals, info)

    # Lists to store the PSD and DE features for each band
    all_psd_data = []
    all_de_data = []

    for band, (fmin, fmax) in bands.items():
        # Compute the PSD using mne's psd_multitaper method
        psd, freqs = mne.time_frequency.psd_multitaper(raw, fmin=fmin, fmax=fmax, bandwidth=2, tmin=None, tmax=None, n_jobs=1)
        
        # Average the PSD values across the frequencies within the band limits
        band_psd = psd.mean(axis=1)
        all_psd_data.append(band_psd)
        
        # Compute Differential Entropy (DE)
        normalized_psd = band_psd / band_psd.sum()
        de = -np.sum(normalized_psd * np.log2(normalized_psd), axis=1)
        all_de_data.append(de)

    # Stack the features side by side
    stacked_psd_data = np.vstack(all_psd_data)
    stacked_de_data = np.vstack(all_de_data)

    stacked_data = np.vstack([stacked_psd_data, stacked_de_data])

    return stacked_data.T