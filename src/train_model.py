import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

from preprocess import load_dataset


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "spam.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "spam_classifier.h5"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
LABEL_MAP_PATH = MODELS_DIR / "label_map.json"


def build_model(input_dim: int) -> Sequential:
    model = Sequential(
        [
            Dense(256, activation="relu", input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DATA_PATH)
    dataset = dataset[dataset["label"].isin(["ham", "spam"])].copy()

    label_map = {"ham": 0, "spam": 1}
    y = dataset["label"].map(label_map).to_numpy()

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(dataset["clean_message"]).toarray()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=8, batch_size=32, validation_split=0.1, verbose=1)

    predictions = (model.predict(X_test) > 0.5).astype(int).flatten()
    print(classification_report(y_test, predictions, target_names=["ham", "spam"]))

    model.save(MODEL_PATH)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Vectorizer saved to: {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
