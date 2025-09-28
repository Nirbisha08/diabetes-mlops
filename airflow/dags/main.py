from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import redis

app = FastAPI(title="Diabetes Prediction API", version="1.0")

# === Load run_id from Redis ===
try:
    r = redis.Redis(host="localhost", port=6379, db=0)
    run_id = r.get("diabetes:model_run_id")
    if run_id is None:
        raise ValueError("No model_run_id found in Redis")
    run_id = run_id.decode("utf-8")
    model_uri = f"runs:/{run_id}/final_xgboost_model"
    mlflow.set_tracking_uri('http://localhost:5000')
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"✅ Loaded model from run_id: {run_id}")
except Exception as e:
    model = None
    print(f"❌ Failed to load model: {e}")

# === Request Schema ===
class PatientData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.get("/")
def read_root():
    return {"message": "Welcome to the Diabetes Prediction API!"}

@app.post("/predict")
def predict(data: PatientData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not available")
    try:
        input_df = pd.DataFrame([data.dict()])
        prediction = model.predict(input_df)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")