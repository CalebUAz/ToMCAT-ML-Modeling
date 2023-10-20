import neurokit2 as nk

def get_ekg_features(signals):
    # signal is of type dataframe
    # print('signal shape:', signals.shape)
    ecg_cleaned = nk.ecg_clean(signals, sampling_rate=500, method="biosppy")
    rpeaks = nk.ecg_findpeaks(ecg_cleaned, method='pantompkins')

    signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=500)

    # Dropping the 'raw_ecg' column
    signals.drop(columns=['ECG_Raw'])
    # print('signal shape after dropping RAW:', signals.shape)
    signals.fillna(0, inplace=True)
    # print('signal shape after fill NA:', signals.shape)

    signal_arr = signals.values

    return signal_arr
