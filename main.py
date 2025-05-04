import os
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import numpy as np
import io
import gdown

MODEL_URL = "https://drive.google.com/uc?id=1at5e6eTviKQQqedffEF_X5fg1DWM-Ert"  # Replace with your actual model ID
MODEL_PATH = "gender_classifier_efficientnetb4.h5"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    print("Download completed.")

app = FastAPI()
model = load_model(MODEL_PATH)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((380, 380))  # size for EfficientNetB4
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = preprocess_image(contents)
        pred = model.predict(img)[0][0]
        label = "male" if pred > 0.5 else "female"
        confidence = float(pred if pred > 0.5 else 1 - pred)
        return JSONResponse(content={"gender": label, "confidence": round(confidence, 3)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})