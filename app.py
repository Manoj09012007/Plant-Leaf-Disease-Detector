
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import os

app = Flask(__name__,template_folder="app/templates",
    static_folder="app/static")

# Load trained model
model = tf.keras.models.load_model(
    "models/plant_disease_model.keras"
)

# Load class names
with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Predict plant disease
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "No image uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "Please select an image"

    try:
        # Read uploaded image
        image = Image.open(file.stream).convert("RGB")

        # Resize image to model input size
        image = image.resize((128, 128))

        # Convert image into numpy array
        image_array = np.array(image)

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        # Predict disease
        predictions = model.predict(image_array)

        # Get predicted class
        predicted_index = int(np.argmax(predictions[0]))

        disease_name = class_names[predicted_index]

        # Calculate confidence
        confidence = float(predictions[0][predicted_index]) * 100

        return render_template(
            "index.html",
            prediction=disease_name,
            confidence=round(confidence, 2)
        )

    except Exception as e:
        return f"Prediction error: {e}"



if __name__ == "__main__":
    app.run(debug=True)
