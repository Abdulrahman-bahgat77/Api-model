from flask import Flask, request, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Precision, Recall
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
app = Flask(__name__)

# Define the class names
class_names = [
    'Pyramid_of_Djoser',
    'Hatshepsut Temple',
    'Wadi_Degla',
    'Azhar Mosque',
    'sant catrine',
    'The Great Temple of Ramesses II',
    'Great_Pyramid_of_Giza',
    'Amr Ibn Al-Aas Mosque',
    'Ramesseum',
    'Sphinx',
    'Cairo Tower',
    'Great_Hypostyle_Hall_of_Karnak',
    'cairo citadel',
    'Bibliotheca Alexandrina',
    'Statue of Ramesses II',
    'citadel of qaitbay 2',
    'Qaitbay Castle'
]

# Load the fine-tuned model
custom_objects = {'Precision': Precision, 'Recall': Recall}
model = load_model('model_fine-tune_EXV5.h5', custom_objects=custom_objects)


# Image preprocessing function
def preprocess_image(image):
    img = cv2.resize(image, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype='float32') / 255.0
    img = img.reshape(1, 224, 224, 3)
    return img


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        processed_img = preprocess_image(img)
        predictions = model.predict(processed_img)
        predicted_class_idx = np.argmax(predictions, axis=-1)[0]
        predicted_class = class_names[predicted_class_idx]

        return jsonify({'name': predicted_class})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Welcome to the Tourism Image Classification API',
        'endpoint': '/predict',
        'method': 'POST',
        'required': 'image file'
    })


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)