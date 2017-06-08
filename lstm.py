import os
import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide messy TensorFlow warnings
warnings.filterwarnings("ignore")  # Hide messy Numpy warnings


def load_data(filename, seq_len, train_ratio, normalise_window):

    with open(filename, 'r') as f:
        data = f.read().split('\n')
    data = np.array(data).astype('float32')

    ''' the reason for this is so that when we 
    peel off the last data point as a label 
    below (y_train = ...), we still end up 
    with a nice even number, e.g 50 '''
    seq_len += 1
    result = np.zeros((len(data)-seq_len, seq_len, 1))

    # this shouldn't affect the sine wave data
    if normalise_window:
        data -= data.mean()
        data /= data.max()

    for i in range(len(data) - seq_len):
        result[i, :, 0] = data[i:i + seq_len]

    # create training and testing split
    split = int(round(train_ratio * len(data)))
    train = result[:split, :, :]
    test = result[split:, :, :]

    #np.random.shuffle(train)

    # take all but the last data point of each window as input
    x_train = train[:, :-1, :]
    # take the last data point of each window as label
    y_train = train[:, -1, :]
    # same procedure for test
    x_test = test[:, :-1, :]
    y_test = test[:, -1, :]

    return [x_train, y_train, x_test, y_test]
'''
def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1)
                             for p in window]
        normalised_data.append(normalised_window)
    return normalised_data
'''

def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_shape=(layers[1], layers[0]),
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model


def predict_point_by_point(model, data):
    # Predict each timestep given the last sequence of true data, in effect
    # only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted


def predict_sequence_full(model, data, window_size):
    # Shift the window by 1 new prediction each time, re-run predictions on
    # new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(
            curr_frame, [window_size - 1], predicted[-1], axis=0)
    return predicted


def predict_sequences_multiple(model, data, window_size, prediction_len):
    # Predict sequence of 50 steps before shifting prediction run forward by
    # 50 steps
    prediction_seqs = []
    for i in range(int(len(data) / prediction_len)):
        curr_frame = data[i * prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(
                curr_frame, [window_size - 1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs
