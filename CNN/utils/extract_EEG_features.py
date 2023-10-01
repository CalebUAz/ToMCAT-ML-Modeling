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
            band_wavelet_data = []
            for channel_data in filtered_data:
                coeffs = pywt.wavedec(channel_data, wavelet, level=level)
                reconstructed_signal = pywt.waverec(coeffs, wavelet)
                if len(reconstructed_signal) != len(channel_data):
                    reconstructed_signal = reconstructed_signal[:len(channel_data)]
                band_wavelet_data.append(reconstructed_signal)
            all_wavelet_data.append(np.array(band_wavelet_data))

        if use_emd:
            band_imf_data = []
            for channel_data in filtered_data:
                emd = EMD()
                imfs = emd(channel_data)
                first_four_imfs = imfs[:4, :] if imfs.shape[0] >= 4 else np.vstack((imfs, np.zeros((4 - imfs.shape[0], imfs.shape[1]))))
                band_imf_data.append(first_four_imfs)
            all_imf_data.append(np.array(band_imf_data))

    stacked_band_data = np.concatenate(all_band_data, axis=0)
    
    combined_data = stacked_band_data
    if use_wavelet:
        stacked_wavelet_data = np.concatenate(all_wavelet_data, axis=0)
        combined_data = np.concatenate((combined_data, stacked_wavelet_data), axis=0)
    if use_emd:
        stacked_imf_data = np.concatenate(all_imf_data, axis=0)
        combined_data = np.concatenate((combined_data, stacked_imf_data), axis=0)
        
    return combined_data.T