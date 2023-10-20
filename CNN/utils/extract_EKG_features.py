import neurokit2 as nk

def get_ekg_features(signals):
    # signal is of type dataframe
    ecg_cleaned = nk.ecg_clean(signals, sampling_rate=500, method="biosppy")
    rpeaks = nk.ecg_findpeaks(ecg_cleaned, method='pantompkins')

    signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=500)

    # Dropping the 'raw_ecg' column
    signal = signal.drop(columns=['ECG_Raw'])
    signals.fillna(0, inplace=True)

    signal_arr = signal.values

    return signal_arr