"""
api/main.py - Simple FastAPI REST API for heart disease prediction.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from src.predict import predict, load_model_meta

app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Patient(BaseModel):
    age:      float = Field(..., example=55)
    sex:      float = Field(..., example=1)
    cp:       float = Field(..., example=2)
    trestbps: float = Field(..., example=130)
    chol:     float = Field(..., example=250)
    fbs:      float = Field(..., example=0)
    restecg:  float = Field(..., example=1)
    thalach:  float = Field(..., example=165)
    exang:    float = Field(..., example=0)
    oldpeak:  float = Field(..., example=1.2)
    slope:    float = Field(..., example=2)
    ca:       float = Field(..., example=0)
    thal:     float = Field(..., example=2)


@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return load_model_meta()


@app.post("/predict")
def predict_endpoint(patient: Patient):
    try:
        return predict(patient.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)