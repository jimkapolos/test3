import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense


def split_sequence(sequence, n_steps):
    """Split a univariate sequence into samples"""
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def create_lstm_model():
    """Create and train an LSTM model"""
    # input data
    raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    # prepare data
    n_steps = 3
    X, y = split_sequence(raw_seq, n_steps)
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    # model
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # train model
    model.fit(X, y, epochs=200, verbose=0)
    return model


if __name__ == "__main__":
    import inspect
    print("Function for model creation and training:")
    print(inspect.getsource(create_lstm_model))
    model = create_lstm_model()
    print("Model training completed. Use another script to save it.")
