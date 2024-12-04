import os
from flask import Flask, request, jsonify, render_template
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle

# Load the trained model
model = tf.keras.models.load_model('audio_classification_model.h5')

# Load the label encoder and scaler
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

app = Flask(__name__)

def extract_features(audio_path, n_mfcc=13):
    try:
        y, sr = librosa.load(audio_path, sr=44100)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        features = np.hstack([mfccs, chroma, spectral_contrast])
        return features
    except Exception as e:
        raise ValueError(f"Error processing audio file: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    audio_file = request.files['audio_file']
    audio_path = f'temp/{audio_file.filename}'
    audio_file.save(audio_path)

    try:
        features = extract_features(audio_path).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]

        return jsonify({'predicted_class': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)})
    finally:
        os.remove(audio_path)

if __name__ == '__main__':
    app.run(debug=True)
