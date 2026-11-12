from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import io
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # reject uploads over 10 MB

# Load model once at startup
model = load_model('model/efficientnet_plant_disease.keras', compile=False)

# 23 classes in alphabetical order (matches effnet_test_gen.class_indices from training)
class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy',
]


def prettify(label: str) -> str:
    """Turn 'Tomato__Tomato_YellowLeaf__Curl_Virus' into 'Tomato - Tomato YellowLeaf Curl Virus'."""
    return label.replace('___', ' - ').replace('__', ' - ').replace('_', ' ').strip()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', error='No file uploaded.')

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', error='No file selected.')

    # Load and prepare the image - same pipeline as training
    try:
        img = image.load_img(io.BytesIO(file.read()), target_size=(224, 224))
    except Exception:
        return render_template('index.html', error='That file is not a valid image.')
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  # no-op for EfficientNet, kept for consistency with training

    # Predict
    preds = model.predict(x, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx]) * 100

    # Top-3 for richer UI
    top3_idx = preds.argsort()[-3:][::-1]
    top3 = [(prettify(class_names[i]), float(preds[i]) * 100) for i in top3_idx]

    return render_template(
        'result.html',
        disease=prettify(class_names[top_idx]),
        confidence=f"{confidence:.2f}",
        top3=top3,
    )


if __name__ == '__main__':
    # Debug mode exposes an interactive debugger, so it is opt-in for local use only
    app.run(host='0.0.0.0', port=8000, debug=os.getenv('FLASK_DEBUG') == '1')