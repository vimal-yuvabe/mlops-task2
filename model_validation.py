import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_squared_error
import requests
import os
import sys
import shutil

def download_model(url: str, save_path: str):
    """Download the model from the given URL."""
    print(f"Downloading model from {url}...")
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download model from {url}. HTTP status code: {response.status_code}")
    with open(save_path, 'wb') as f:
        f.write(response.content)
    print(f"Model downloaded to {save_path}")

def predict_model(model_path):
    # Load validation dataset
    validation_data = pd.read_csv('./test_data.csv')
    X_val = validation_data.drop(columns=['medv'])
    y_val = validation_data['medv']
    # Load the new model
    model = xgb.Booster()
    model.load_model(model_path)
    # Make predictions and calculate new accuracy
    dmatrix = xgb.DMatrix(X_val)
    predictions = model.predict(dmatrix)
    mse = mean_squared_error(y_val, predictions.round())
    return mse

def validate_model():
    """Validate the new model and compare its accuracy with the previous model."""
    # Read the model URL
    with open('./model_version.txt', 'r') as f:
        model_url = f.read().strip()

    # Download the new model
    new_model_path = './new_model.bst'
    download_model(model_url, new_model_path)

    current_mse = predict_model('./model.bst')
    new_mse = predict_model('./new_model.bst')

    # Exit if new accuracy is not better
    if new_mse >= current_mse:
        print("New model's mse is not better. Exiting.")
        sys.exit(1)
    else:
        # If new model is better, replace the old model
        print("New model is better. Updating deployment package.")
        shutil.move(new_model_path, './model.bst')
if __name__ == "__main__":
    validate_model()
