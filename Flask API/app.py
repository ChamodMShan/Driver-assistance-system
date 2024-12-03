from flask import Flask, request, render_template, jsonify
from keras.models import load_model
from keras_preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

model1 = load_model('vggModel_driver_behaviour.h5', compile=False)

class_names = [
    "c0: safe driving",
    "c1: texting - right",
    "c2: talking on the phone - right",
    "c3: texting - left",
    "c4: talking on the phone - left",
    "c5: operating the radio",
    "c6: drinking",
    "c7: reaching behind",
    "c8: hair and makeup",
    "c9: talking to passenger"
]

def predict_sign(image_path):
    img = load_img(image_path, target_size=(224, 224, 3))
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
        image_path = "uploaded_image.jpg"
        image.save(image_path)

        prediction = predict_sign(image_path)

        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
