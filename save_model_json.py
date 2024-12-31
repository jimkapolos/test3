import json
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense


def create_model_from_metadata(metadata):
    """Recreate the model based on metadata"""
    n_steps = metadata["parameters"]["n_steps"]
    n_features = metadata["parameters"]["n_features"]

    model = Sequential()
    for layer in metadata["parameters"]["layers"]:
        if layer["type"] == "LSTM":
            model.add(LSTM(layer["units"], activation=layer["activation"], input_shape=(n_steps, n_features)))
        elif layer["type"] == "Dense":
            model.add(Dense(layer["units"]))

    model.compile(optimizer=metadata["parameters"]["optimizer"], loss=metadata["parameters"]["loss"])
    return model


def find_next_version(output_dir):
    """Find the next available version for saving the model"""
    version = 1
    while os.path.exists(os.path.join(output_dir, f"{version}")):
        version += 1
    return version


def save_model(model, output_dir):
    """Save the trained model to the specified directory with versioning"""
    next_version = find_next_version(output_dir)
    save_path = os.path.join(output_dir, f"{next_version}")
    os.makedirs(save_path, exist_ok=True)
    tf.saved_model.save(model, save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    # Load metadata
    with open("model_metadata.json", "r") as f:
        metadata = json.load(f)

    # Recreate and train model
    model = create_model_from_metadata(metadata)
    print(f"Recreated model using framework: {metadata['framework']}")

    # Save model
    output_dir =  r"C:\Users\JIM\Desktop\work\model_save"
    save_model(model, output_dir)
