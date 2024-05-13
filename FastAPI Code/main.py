from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

# Define model path and class names
MODEL_PATH = os.getenv("MODEL_PATH", "../saved_models/2/my_saved_model.keras")
CLASS_NAMES = ["glioma_tumor", "no_tumor", "meningioma_tumor", "pituitary_tumor"]
IMAGE_SIZE = (150, 150)

# Load the model with error handling
try:
    MODEL = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Ensure the loaded model has a 'predict' method
if not hasattr(MODEL, 'predict'):
    raise AttributeError("The loaded object does not have a 'predict' method.")

# Initialize FastAPI app
app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost,http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive!"}

def read_file_as_image(data, target_size=IMAGE_SIZE) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data))
        image = image.resize(target_size)
        return np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)  # Expand dimensions to fit model input

        # Ensure the model has 'predict' before using it
        if not hasattr(MODEL, 'predict'):
            raise AttributeError("Model does not have 'predict' method.")

        # Make predictions with the model
        predictions = MODEL.predict(img_batch)

        # Determine confidence and predicted class
        confidence = float(np.max(predictions[0]))
        predicted_class_index = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_class_index]

        # Map predicted class to human-readable format
        class_display_names = {
            "glioma_tumor": "Glioma Tumor",
            "no_tumor": "No Tumor",
            "meningioma_tumor": "Meningioma Tumor",
            "pituitary_tumor": "Pituitary Tumor"
        }
        predicted_class_display = class_display_names.get(predicted_class, "Unknown")

        return {
            "class": predicted_class_display,
            "confidence": confidence,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=int(os.getenv("PORT", 8000)))
