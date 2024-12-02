import requests

#REST End point URL
url = "https://housing-prediction-736790245655.us-central1.run.app/predict"
data = {
    "features":[
        12.8023,
        0,
        18.1,
        0,
        0.74,
        5.854,
        96.6,
        1.8956,
        24,
        666,
        20.2,
        240.52,
        23.79

    ]
}
resp = requests.post(url=url,json=data).json()
print(f"Prediciton output(medv):\n{resp}")