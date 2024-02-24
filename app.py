from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os


IMAGE_SIZE = [208,176] 
# Initialize Flask application
app = Flask(__name__)

# Load the saved TensorFlow CNN model
model = load_model('model.h5')

# Define the prediction endpoint
@app.route('/upload', methods=['POST'])
def predict():
    # Get the image data from the request
    image_data = request.files['image']
    
    # Preprocess the image data
    processed_image = preprocess_image(image_data)

    # Make prediction
    prediction = model.predict(processed_image)
    
    # Convert prediction to human-readable format
    predicted_class = decode_predictions(prediction)
    
    # Prepare the response
    response = {
        'dementia': predicted_class
    }
    
    # Return the response as JSON
    return jsonify(response)

def preprocess_image(image_data):
    # Decode the image data into an image object
    image = Image.open(image_data.stream)
    # Resize the image to fit the model's input shape
    image = image.convert('RGB')
    processed_image = image.resize(IMAGE_SIZE)
    # Convert the image to a numpy array and normalize pixel values
    processed_image = np.array(processed_image) / 255.0
    # Reshape and expand dimensions to match the model input shape
    processed_image = np.expand_dims(processed_image, axis=0)

    return processed_image

# Helper function to decode model predictions
def decode_predictions(prediction):
    labels = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']  # Example class labels
    predicted_class_index = np.argmax(prediction)
    predicted_class = labels[predicted_class_index]
    return predicted_class

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
