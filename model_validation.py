import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_squared_error
import requests
import os
import sys
from google.cloud import storage



def upload_best_model(model_name):
    """Uploads a file to the bucket."""
    # Create a storage client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket("tymestack-artifacts")

    # Create a new blob and upload the file
    blob = bucket.blob("housing-prediction/current.bst")
    blob.upload_from_filename(model_name)


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
    current_model_url = "https://storage.googleapis.com/tymestack-artifacts/housing-prediction/current.bst"
    best_model_url = "https://storage.googleapis.com/tymestack-artifacts/housing-prediction/best.bst"
    
    # Download the new model
   
    download_model(best_model_url, './best.bst')
    download_model(current_model_url,'./current.bst')

    current_mse = predict_model('./current.bst')
    new_mse = predict_model('./best.bst')

    # Exit if new accuracy is not better
    if new_mse >= current_mse:
        print("New model's mse is not better. Exiting.")
        sys.exit(1)
    else:
        # If new model is better, replace the old model
        print("New model is better. Updating deployment package.")
        # Upload the new model to bucket
        upload_best_model("./best.bst")
if __name__ == "__main__":
    # validate_model()
    upload_best_model("./current.bst")
