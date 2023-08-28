import numpy as np

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