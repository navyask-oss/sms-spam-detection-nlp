import json
import pickle
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

from preprocess import clean_text


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "spam_classifier.h5"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
LABEL_MAP_PATH = MODELS_DIR / "label_map.json"


class SpamPredictor:
    def __init__(self) -> None:
        if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists() or not LABEL_MAP_PATH.exists():
            raise FileNotFoundError(
                "Model artifacts are missing. Run `python src/train_model.py` first."
            )

        self.model = load_model(MODEL_PATH)
        with open(VECTORIZER_PATH, "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
            label_map = json.load(f)

        self.inverse_label_map = {int(v): k for k, v in label_map.items()}

    def predict(self, message: str) -> dict:
        cleaned = clean_text(message)
        features = self.vectorizer.transform([cleaned]).toarray()
        score = float(self.model.predict(features, verbose=0)[0][0])
        predicted_class = int(score >= 0.5)

        return {
            "message": message,
            "cleaned_message": cleaned,
            "label": self.inverse_label_map[predicted_class],
            "spam_probability": round(score, 4),
        }


def predict_message(message: str) -> dict:
    predictor = SpamPredictor()
    return predictor.predict(message)


if __name__ == "__main__":
    sample = "Congratulations! You have won a free ticket."
    print(predict_message(sample))
