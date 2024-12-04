from flask import Flask, request, render_template, jsonify
from keras.models import load_model
from keras_preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained model
model1 = load_model('./vggModel.h5', compile=False)

# class names
class_names = [
    "children crossing",
    "crossing",
    "Falling-Rocks-Ahead",
    "hospital",
    "level crossing with gates",
    "no honking",
    "no left turn",
    "no right turn",
    "no u turn",
    "roundabout",
    "speed limit 20",
    "speed limit 60",
    "speed limit 100",
    "stop"
]

# Function to make predictions
def predict_sign(image_path):
    img = load_img(image_path, target_size=(150, 150, 3))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model1.predict(img)
    predicted_class_index = np.argmax(prediction, axis=-1)[0]
    result = class_names[predicted_class_index]
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({"error": "No image selected"}), 400

    try:
        image_path = "./uploaded_image.jpg"
        image.save(image_path)

        prediction = predict_sign(image_path)

        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
