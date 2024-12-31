import os
import tensorflow as tf  # import torch


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
    tf.saved_model.save(model, save_path)  # torch.save(model.state_dict(),save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    # Developer provides the function or trained model directly
    from train_model import create_lstm_model

    trained_model = create_lstm_model()
    output_dir = r"C:\Users\JIM\Desktop\work\model_save"

    save_model(trained_model, output_dir)

