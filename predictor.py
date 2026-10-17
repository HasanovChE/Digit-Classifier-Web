import numpy as np
import torch
#from config import FRAMEWORK

def predict(model, image_tensor,framework):
    """
    Makes a prediction using the loaded model and preprocessed image tensor.
    Returns the top 3 predicted digits and their probabilities.
    """
    
    if framework == "pytorch":
        # Disable gradient calculation for inference to save memory and computations
        with torch.no_grad():
            logits = model(image_tensor)
            # Apply softmax to get probability distribution
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
            
    elif framework == "tensorflow":
        # Predict using the TensorFlow/Keras model
        # image_tensor.numpy() converts the tensor back to a NumPy array for TF
        probabilities = model.predict(image_tensor.numpy(), verbose=0)[0]
        
    else:
        raise ValueError(f"Unsupported framework: {framework}")

    # Get the indices of the top 3 highest probabilities in descending order
    top_indices = np.argsort(probabilities)[::-1][:3]
    
    results = []
    # Format the results into a list of dictionaries
    for index in top_indices:
        results.append({
            "digit": int(index),
            "probability": float(probabilities[index])
        })
        
    return results