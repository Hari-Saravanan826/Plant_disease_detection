# AgriVision AI: Plant Disease Detection Web App

A Flask web app that takes a photo of a plant leaf and returns the most likely disease with a confidence score and the top 3 predictions. The model is packaged in a Docker container and was deployed to a Google Cloud VM.

## What it does
- Upload a leaf image in the browser
- The app resizes it to 224x224 and runs it through an EfficientNetB0 classifier
- Returns the predicted class (for example "Tomato - Late blight"), the confidence, and the top 3 alternatives
- Covers 23 classes from the PlantVillage dataset (apple, corn, pepper, potato, tomato)

## What I built
I worked on the model and owned the deployment. On the model side, I trained part of the EfficientNetB0 classifier (about 99% test accuracy on the 23 classes). On the deployment side, I built everything that turns it into a working service:
- Flask app (`app.py`): model loading at startup, image preprocessing, prediction route, error handling for missing uploads
- HTML templates and CSS for the upload and result pages
- Dockerfile with the system libraries TensorFlow needs, served with Gunicorn
- Deployment of the container to a GCP Compute Engine VM

## Tech stack
Python 3.11, Flask, TensorFlow/Keras 2.18, EfficientNetB0, Gunicorn, Docker, Google Cloud (Compute Engine)

## Run locally with Docker
```bash
docker build -t plant-disease:1.0 .
docker run -p 8000:8000 plant-disease:1.0
```
Then open http://localhost:8000.

## Run without Docker
```bash
pip install -r requirements.txt
python app.py
```

## Known limitations and next steps
- The model was trained on clean PlantVillage lab images, so accuracy drops on photos taken in real fields (domain shift)
- Only 23 of the 38 PlantVillage classes are covered
- I am working on adding the remaining classes and field-image data (PlantDoc, FieldPlant) before redeploying
- The public demo endpoint is currently offline while I do this
