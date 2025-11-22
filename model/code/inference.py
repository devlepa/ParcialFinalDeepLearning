import json
import numpy as np
import tensorflow as tf
import joblib
import os

# ---------- CARGA DEL MODELO ----------
def model_fn(model_dir):
    model = tf.keras.models.load_model(os.path.join(model_dir, "yamnet_transfer_classifier.h5"))
    scaler = joblib.load(os.path.join(model_dir, "yamnet_scaler.joblib"))
    return {"model": model, "scaler": scaler}

# --------- PREPROCESAMIENTO ----------
def input_fn(request_body, content_type):
    data = json.loads(request_body)
    return np.array(data["input"])

# -------- PREDICCIÓN ----------
def predict_fn(input_data, model_dict):
    scaler = model_dict["scaler"]
    model = model_dict["model"]
    
    X_scaled = scaler.transform(input_data)
    preds = model.predict(X_scaled)
    return preds

# -------- POSTPROCESAMIENTO ----------
def output_fn(prediction, accept):
    return json.dumps({"prediction": prediction.tolist()})
