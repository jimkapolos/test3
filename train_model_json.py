import json

def create_lstm_model():
    """
    This function creates and trains an LSTM model using TensorFlow/Keras.
    The model predicts the next value in a sequence.
    """
    from keras.models import Sequential
    from keras.layers import LSTM, Dense
    import numpy as np

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

    # Input data
    raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    # Prepare data
    n_steps = 3
    X, y = split_sequence(raw_seq, n_steps)
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    # Model
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Train model
    model.fit(X, y, epochs=200, verbose=0)
    return model


if __name__ == "__main__":
    model_metadata = {
        "function_name": "create_lstm_model",
        "framework": "TensorFlow",
        "description": "An LSTM model for sequence prediction",
        "parameters": {
            "n_steps": 3,
            "n_features": 1,
            "layers": [
                {"type": "LSTM", "units": 100, "activation": "relu"},
                {"type": "Dense", "units": 1}
            ],
            "optimizer": "adam",
            "loss": "mse",
            "epochs": 200
        }
    }

    with open("model_metadata.json", "w") as f:
        json.dump(model_metadata, f, indent=4)

    print("Model metadata has been saved to 'model_metadata.json'")
