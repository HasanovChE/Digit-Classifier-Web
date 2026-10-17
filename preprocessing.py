import numpy as np
from PIL import Image, ImageOps
import torch
# from config import FRAMEWORK

def preprocess_canvas(image_data, framework):
    # 1. Convert Canvas RGBA array to PIL Image
    image = Image.fromarray(image_data.astype(np.uint8))
    
    # 2. Replace transparent background with white
    rgb = Image.new("RGB", image.size, "white")
    if image.mode == "RGBA":
        rgb.paste(image, mask=image.split()[3])
    else:
        rgb.paste(image)
        
    # 3. Convert to Grayscale
    gray_image = rgb.convert("L")
    
    # 4. Invert colors (White background -> Black background)
    inverted_image = ImageOps.invert(gray_image)
    
    # --- ROBUST CENTERING ---
    bbox = inverted_image.getbbox()
    if bbox is not None:
        cropped = inverted_image.crop(bbox)
        cropped.thumbnail((20, 20), Image.Resampling.LANCZOS)
        final_image = Image.new("L", (28, 28), 0)
        
        x = (28 - cropped.width) // 2
        y = (28 - cropped.height) // 2
        final_image.paste(cropped, (x, y))
    else:
        final_image = Image.new("L", (28, 28), 0)
    
    # 5. Convert to NumPy Array
    img_array = np.array(final_image)
    
    # 6. Normalize (Scale 0-255 to 0-1)
    img_normalized = img_array.astype(np.float32) / 255.0
    
    # 7. Convert to Tensor with correct dimensions
    if framework == "pytorch":
        # PyTorch: [Batch, Channel, Height, Width] -> [1, 1, 28, 28]
        tensor = torch.tensor(img_normalized).unsqueeze(0).unsqueeze(0)
    elif framework == "tensorflow":
        # TensorFlow: [Batch, Height, Width, Channel] -> [1, 28, 28, 1]
        import tensorflow as tf
        tensor = tf.convert_to_tensor(img_normalized)
        tensor = tf.expand_dims(tensor, axis=0)
        tensor = tf.expand_dims(tensor, axis=-1)
    else:
        raise ValueError(f"Unsupported framework: {framework}")
        
    return tensor, final_image