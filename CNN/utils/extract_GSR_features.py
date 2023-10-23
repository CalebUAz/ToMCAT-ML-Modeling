import neurokit2 as nk

def get_gsr_features(signals):
    # signal is of type dataframe
    # print('signal shape:', signals.shape)
    signals = nk.eda_process(signals, sampling_rate=500)
    signals = signals[0]

    # Dropping the 'EDA_Raw' column
    signals.drop(columns=['EDA_Raw'])
    # print('signal shape after dropping RAW:', signals.shape)
    signals.fillna(0, inplace=True)
    # print('signal shape after fill NA:', signals.shape)

    signal_arr = signals.values

    return signal_arr
