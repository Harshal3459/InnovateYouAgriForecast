import requests
import json

url = "http://127.0.0.1:5000/predict"
data = {                              #example data
    "market": "Mumbai",
    "variety": "Sugar",
    "days": 5
}

response = requests.post(url, json=data)
if response.status_code == 200:
    predictions = response.json()
    print(predictions)
    # Save the JSON response to a file
    with open('predictions.json', 'w') as json_file:
        json.dump(predictions, json_file, indent=4)
else:
    print(f"Error: {response.status_code}")
    print(response.text)
