import requests

# Endpoint URL
url = 'http://localhost:8080/predict'  # Update the URL if necessary

# Image file path
image_file = 'images/28.jpg'  # Replace with the path to your image file

# Open the image file
with open(image_file, 'rb') as f:
    # Prepare the request data
    files = {'image': f}

    # Send the POST request to the API
    response = requests.post(url, files=files)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Get the prediction from the response
        prediction = response.json()['prediction']
        print('Prediction:', prediction)
    else:
        print('Error:', response.text)