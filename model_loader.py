import torch
import tensorflow as tf
from config import (
    PYTORCH_MODEL_PATH,
    TENSORFLOW_MODEL_PATH
)
from model_architecture import CNN

def load_pytorch_model():
    model = CNN()
    checkpoint = torch.load(
        PYTORCH_MODEL_PATH,
        map_location="cpu"
    )
    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint
        
    model.eval()
    return model

def load_tensorflow_model():
    model = tf.keras.models.load_model(
        TENSORFLOW_MODEL_PATH
    )
    return model

# def load_model():
#     if FRAMEWORK == "pytorch":
#         return load_pytorch_model()
#     elif FRAMEWORK == "tensorflow":
#         return load_tensorflow_model()
#     else:
#         raise ValueError(
#             f"Unsupported framework: {FRAMEWORK}"
#         )