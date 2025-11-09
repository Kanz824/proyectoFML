
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn
import io

# ===============================
# CARGA DEL MODELO Y TRANSFORMADOR
# ===============================
MODEL_PATH = "best_model_GradientBoosting.pkl"
#TRANSFORMER_PATH = "models/transformer.pkl"  # si usas ColumnTransformer guardado

model = joblib.load(MODEL_PATH)
#transformer = joblib.load(TRANSFORMER_PATH)

# ===============================
# CONFIGURACIÓN DE LA APP FASTAPI
# ===============================
app = FastAPI(
    title="Employee Attrition Prediction API",
    description="API para predecir la probabilidad de deserción de empleados (por registro o batch)",
    version="1.0.0",
)

# ===============================
# ESTRUCTURA DE DATOS DE ENTRADA
# ===============================
class EmployeeInput(BaseModel):
    Age: float
    DistanceFromHome: float
    MonthlyIncome: float
    NumCompaniesWorked: float
    PercentSalaryHike: float
    YearsAtCompany: float
    YearsInCurrentRole: float
    YearsWithCurrManager: float
    BusinessTravel: str
    Department: str
    EducationField: str
    JobRole: str
    MaritalStatus: str
    OverTime: str
    Gender: str
    JobLevel: int


# ===============================
# ENDPOINT DE PRUEBA
# ===============================
@app.get("/")
def home():
    return {"message": "🚀 Employee Attrition Prediction API is running!"}


# ===============================
# ENDPOINT: PREDICCIÓN INDIVIDUAL
# ===============================
@app.post("/predict")
def predict(data: EmployeeInput):
    df = pd.DataFrame([data.dict()])
    df_transformed = transformer.transform(df)
    prediction = model.predict(df_transformed)[0]
    prob = model.predict_proba(df_transformed)[0][1]
    return {
        "prediction": int(prediction),
        "probability_attrition": float(round(prob, 4)),
    }


# ===============================
# ENDPOINT: PREDICCIÓN POR LOTE (BATCH)
# ===============================
@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Permite subir un archivo CSV para predicciones por lotes.
    Retorna las predicciones y probabilidades.
    """
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    df_transformed = transformer.transform(df)
    predictions = model.predict(df_transformed)
    probabilities = model.predict_proba(df_transformed)[:, 1]

    results = pd.DataFrame({
        "prediction": predictions,
        "probability_attrition": probabilities
    })

    return results.to_dict(orient="records")


# ===============================
# PUNTO DE ENTRADA
# ===============================
if __name__ == "__main__":
    uvicorn.run("model_deploy:app", host="0.0.0.0", port=8000, reload=True)
