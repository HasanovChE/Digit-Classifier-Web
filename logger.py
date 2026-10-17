from datetime import datetime
from pathlib import Path
import pandas as pd
from config import LOG_DIR

# Define the path for the predictions CSV file
LOG_FILE = LOG_DIR / "predictions.csv"

def log_prediction(prediction, probabilities):
    """
    Logs every prediction with a timestamp to a CSV file.
    Expects the top prediction and the list of top-3 probability dictionaries.
    """
    timestamp = datetime.now().isoformat()
    
    # Create a dictionary for the new row of data
    row = {
        "timestamp": timestamp,
        "prediction": prediction,
        "top1_probability": probabilities[0]["probability"],
        "top2_digit": probabilities[1]["digit"],
        "top2_probability": probabilities[1]["probability"],
        "top3_digit": probabilities[2]["digit"],
        "top3_probability": probabilities[2]["probability"]
    }
    
    # Check if the log file already exists
    if LOG_FILE.exists():
        # Read existing data and append the new row
        df = pd.read_csv(LOG_FILE)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        # Create a new DataFrame if the file doesn't exist
        df = pd.DataFrame([row])
        
    # Save the updated DataFrame back to the CSV file
    df.to_csv(LOG_FILE, index=False)