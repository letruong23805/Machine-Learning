import joblib
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Spam Classification API - Support Vector Machine (SVM)")

# Nạp model và vectorizer
MODEL_PATH = "svm_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
else:
    model = None
    vectorizer = None


class EmailInput(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "SVM Email Spam Classification API đang hoạt động!",
        "status": "Ready" if model else "Model not found. Please run train.py first."
    }


@app.post("/predict")
def predict_spam(data: EmailInput):
    if not model or not vectorizer:
        raise HTTPException(status_code=500, detail="Mô hình chưa được huấn luyện (thiếu file .pkl).")

    X_input = vectorizer.transform([data.text])
    prediction = model.predict(X_input)[0]
    probabilities = model.predict_proba(X_input)[0]

    return {
        "text": data.text,
        "prediction": str(prediction),
        "confidence": float(max(probabilities))
    }
