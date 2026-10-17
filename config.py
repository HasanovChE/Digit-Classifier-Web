from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "./models"
LOG_DIR = BASE_DIR / "logs"
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# SELECT MODEL FRAMEWORK
# FRAMEWORK= "pytorch"
# FRAMEWORK = "tensorflow"

# MODEL PATH
PYTORCH_MODEL_PATH = MODEL_DIR / "pytorch_cnn.pth"
TENSORFLOW_MODEL_PATH = MODEL_DIR / "tensorflow_cnn.keras"

# MNIST CONFIGURATION
NUM_CLASSES = 10
CLASS_NAMES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
IMAGE_SIZE = 28

# TRAINING PREPROCESSING
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# CANVAS
STROKE_WIDTH = 20
CANVAS_SIZE = 280