import numpy as np
import mne
import pywt
from PyEMD import EMD

def get_eeg_frequency_band_data(signals, use_wavelet, use_emd):
    # Frequency bands definitions
    bands = {
        'Theta': (4, 8),
        'Alpha': (8, 14),
        'Beta': (14, 30),
        'Gamma': (30, 40)
    }
    
    sfreq = 500
    signals = signals.T
    n_channels = signals.shape[0]
    ch_names = ["eeg_channel_{}".format(i) for i in range(n_channels)]

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * n_channels)
    raw = mne.io.RawArray(signals, info)

    all_band_data = []
    all_wavelet_data = []
    all_imf_data = []

    wavelet = 'db4'  # Daubechies 4 wavelet
    level = 4  # Level of decomposition

    for band, (fmin, fmax) in bands.items():
        filtered_data = raw.copy().filter(fmin, fmax).get_data()
        all_band_data.append(filtered_data)

    if use_wavelet:
        for channel_data in signals:
            coeffs = pywt.wavedec(channel_data, wavelet, level=level)
            reconstructed_signal = pywt.waverec(coeffs, wavelet)
            if len(reconstructed_signal) != len(channel_data):
                reconstructed_signal = reconstructed_signal[:len(channel_data)]
            all_wavelet_data.append(reconstructed_signal)

    if use_emd:
        for channel_data in signals:
            emd = EMD()
            imfs = emd(channel_data)
            first_four_imfs = imfs[:4, :] if imfs.shape[0] >= 4 else np.vstack((imfs, np.zeros((4 - imfs.shape[0], imfs.shape[1]))))
            all_imf_data.append(first_four_imfs)

    stacked_band_data = np.concatenate(all_band_data, axis=0)
    
    combined_data = stacked_band_data
    if use_wavelet:
        combined_data = np.concatenate((combined_data, np.array(all_wavelet_data)), axis=0)
    if use_emd:
        combined_data = np.concatenate((combined_data, np.vstack(all_imf_data)), axis=0)
        
    return combined_data.T
