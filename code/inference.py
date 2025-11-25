import io
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import joblib
import librosa

################################################################################
# 1. Cargar modelo YAMNet + scaler + clasificador SavedModel
################################################################################

def model_fn(model_dir):
    """
    Carga TODO lo necesario cuando arranca el contenedor de SageMaker.
    - Clasificador SavedModel
    - Scaler
    - YAMNet (TF Hub)
    """

    print(">>>> Cargando modelo SavedModel...")
    classifier = tf.saved_model.load(model_dir)

    print(">>>> Cargando scaler...")
    scaler = joblib.load(f"{model_dir}/yamnet_scaler.joblib")

    print(">>>> Cargando YAMNet desde TF Hub...")
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

    return {
        "classifier": classifier,
        "scaler": scaler,
        "yamnet": yamnet
    }

################################################################################
# 2. Decodificar AUDIO (bytes → waveform 16kHz)
################################################################################

def decode_audio_bytes(audio_bytes):
    """
    Recibe audio en bytes (WAV o MP3)
    Devuelve waveform float32 normalizado a 16kHz.
    """
    audio_buffer = io.BytesIO(audio_bytes)

    # librosa carga mp3 y wav sin problema
    waveform, sr = librosa.load(audio_buffer, sr=16000)
    waveform = waveform.astype(np.float32)

    return waveform, sr

################################################################################
# 3. Extraer embeddings YAMNet y generar vector 2048-d
################################################################################

def extract_yamnet_features(waveform, yamnet_model):
    """
    Calcula embeddings YAMNet y retorna un vector (mean + std) → 2048 dims.
    """
    waveform_tf = tf.convert_to_tensor(waveform, dtype=tf.float32)

    _, embeddings, _ = yamnet_model(waveform_tf)
    embeddings = embeddings.numpy()

    mean = np.mean(embeddings, axis=0)
    std = np.std(embeddings, axis=0)

    return np.concatenate([mean, std])  # (2048,)

################################################################################
# 4. SageMaker Input FN
################################################################################

def input_fn(request_body, request_content_type):
    """
    Recibe audio crudo desde Lambda.
    ContentType = application/octet-stream
    """
    if request_content_type == "application/octet-stream":
        return request_body  # devolvemos los bytes tal cual

    raise ValueError("Content type no soportado: " + request_content_type)

################################################################################
# 5. Realizar inferencia completa (Audio → Embeddings → Scaler → Modelo)
################################################################################

def predict_fn(input_data, model_data):
    """
    input_data = audio bytes
    model_data = { classifier, scaler, yamnet }
    """

    audio_bytes = input_data

    # ---- Cargar recursos ----
    classifier = model_data["classifier"]
    scaler = model_data["scaler"]
    yamnet = model_data["yamnet"]

    # ---- 1. Decodificar audio ----
    waveform, sr = decode_audio_bytes(audio_bytes)

    # ---- 2. Extraer embeddings ----
    features = extract_yamnet_features(waveform, yamnet)

    # ---- 3. Normalizar con el scaler ----
    features_scaled = scaler.transform([features])  # → (1, 2048)

    # ---- 4. Ejecutar el modelo ----
    pred = classifier(features_scaled)
    pred = pred.numpy().flatten()

    class_id = int(np.argmax(pred))
    confidence = float(np.max(pred))

    # ---- 5. Regla de negocio ----
    if confidence < 0.6:
        return {
            "label": "unknown",
            "confidence": 0.0
        }

    # si tu notebook original tenía un diccionario class_names:
    class_names = [
        "air_conditioner", "car_horn", "children_playing", "dog_bark",
        "drilling", "engine_idling", "gun_shot", "jackhammer",
        "siren", "street_music"
    ]

    label = class_names[class_id]

    return {
        "label": label,
        "confidence": confidence
    }

################################################################################
# 6. Formatear salida JSON
################################################################################

def output_fn(prediction, accept):
    """
    Se encarga de devolver JSON al frontend a través de Lambda.
    """

    if accept == "application/json":
        return json.dumps(prediction), "application/json"

    # default
    return json.dumps(prediction), "application/json"

