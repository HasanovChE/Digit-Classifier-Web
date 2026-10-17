# Digit Classifier Web

A simple CNN-based web application that recognizes handwritten digits from a drawing canvas or uploaded image.

## Features

- CNN model trained for handwritten digit classification
- Saved trained model for inference
- Streamlit web interface
- Draw a digit directly on the canvas
- Upload an image for prediction
- Same preprocessing as used during training
- Predicted digit with confidence score
- Top-3 prediction probabilities
- Clear / reset drawing
- Prediction logging
- Live prediction support
- Ready for local execution and deployment

## Tech Stack

- Python
- TensorFlow / Keras
- CNN
- Streamlit
- NumPy
- Pillow
- Matplotlib
- Streamlit Drawable Canvas

## Project Structure

```text
digit-classifier/
├── app.py
├── train.py
├── model/
│   └── digit_classifier.keras
├── requirements.txt
├── predictions.csv
└── README.md
```

## How It Works

```text
User Drawing / Image
        ↓
Image Preprocessing
        ↓
CNN Model
        ↓
Class Probabilities
        ↓
Top-3 Predictions
        ↓
Predicted Digit
```

The input image is converted to grayscale, resized to the training image size, normalized, and reshaped before being passed to the CNN.

## Installation

```bash
git clone <repository-url>
cd digit-classifier

pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The application opens in the browser and allows the user to draw or upload a handwritten digit.

## Prediction Output

The application displays:

- Predicted digit
- Confidence percentage
- Top-3 predicted classes
- Probability chart

## Model

The classifier uses a Convolutional Neural Network consisting of convolution, pooling, flattening, and dense layers.

The trained model is saved and loaded by the web application, so retraining is not required for normal inference.

### Live Prediction

The application can update the prediction after the drawing changes.

### Prediction Logging

Predictions can be stored in `predictions.csv` for later analysis.

### Image Upload

Users can upload an external handwritten-digit image instead of drawing manually.

## Reproducibility

Training and preprocessing settings are kept consistent between model training and inference to ensure reliable predictions.

## Result

This project demonstrates a complete machine-learning workflow:

**CNN training → model saving → image preprocessing → web deployment → real-time inference → prediction visualization.**
