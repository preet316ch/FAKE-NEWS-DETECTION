from pathlib import Path
import re
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "fake_news_model.pkl"
VECTORIZER_PATH = BASE_DIR / "model" / "tfidf_vectorizer.pkl"

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _load():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            "Trained model files are missing. Run 'python train_model.py' first."
        )
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)

def predict_news(text: str) -> dict:
    model, vectorizer = _load()
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    prediction = int(model.predict(features)[0])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = float(max(probabilities) * 100)
    else:
        confidence = 0.0

    label = "REAL" if prediction == 1 else "FAKE"

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "message": (
            "The model classified this article as likely real based on patterns "
            "learned from the training data."
            if label == "REAL"
            else
            "The model classified this article as likely fake based on patterns "
            "learned from the training data."
        )
    }

if __name__ == "__main__":
    sample = "Scientists announced a new study about renewable energy."
    print(predict_news(sample))
