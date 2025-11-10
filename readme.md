# 🧠 Employee Attrition Prediction - MLOps Pipeline

## 📋 Descripción del caso de negocio

La retención del talento humano es uno de los mayores desafíos para las organizaciones modernas. Las empresas que logran identificar las causas detrás de la **deserción laboral** pueden implementar estrategias más efectivas de fidelización, reducir costos de contratación y mejorar la productividad general.  

Este proyecto tiene como objetivo desarrollar un **modelo de machine learning** capaz de predecir la **probabilidad de deserción de un empleado**, a partir de variables relacionadas con su desempeño, satisfacción, características demográficas y condiciones laborales.  

El dataset utilizado proviene de una fuente pública en Kaggle y fue adaptado para fines educativos y experimentales.

---

## 🧩 Estructura del proyecto


---

## ⚙️ Pipeline MLOps

El proyecto sigue una estructura modular y automatizada para facilitar el mantenimiento y escalabilidad:

1. **Ingesta y Limpieza de Datos (`data_preprocessing.py`)**
   - Carga el dataset en formato CSV o JSON.
   - Limpieza de nulos, outliers y variables irrelevantes.
   - Generación del `df_final` almacenado en `/src/`.

2. **Feature Engineering (`ft_engineering.py`)**
   - Uso de `ColumnTransformer` para procesar variables numéricas y categóricas.
   - Aplicación de:
     - `SimpleImputer` (media/moda)
     - `OneHotEncoder` para categóricas nominales.
     - `OrdinalEncoder` para categóricas ordinales.
   - Almacenamiento de las transformaciones procesadas.

3. **Entrenamiento y Evaluación (`model_training_evaluation.py`)**
   - Entrenamiento y comparación de múltiples modelos:
     - Logistic Regression
     - Random Forest
     - Gradient Boosting
     - XGBoost
     - LightGBM
   - Uso de funciones reutilizables:
     - `build_model()` → encapsula el flujo de entrenamiento.
     - `summarize_classification()` → genera métricas detalladas.
   - Evaluación basada en:
     - Accuracy, F1-score, Precision, Recall, ROC-AUC.
   - Visualización de resultados mediante:
     - Gráficos comparativos de rendimiento.
     - Tabla resumen con las métricas de evaluación.
   - Exportación del mejor modelo (`best_model.pkl`).

4. **Interfaz de Usuario (`app.py`)**
   - Desarrollada con **Streamlit**.
   - Permite cargar datos nuevos y obtener predicciones sobre la probabilidad de abandono de empleados.

---

## 📊 Principales hallazgos

Tras el desarrollo del pipeline y la comparación de modelos, se observaron los siguientes resultados:

| Modelo                | Accuracy | F1-Score | ROC-AUC | Tiempo de Entrenamiento |
|------------------------|-----------|-----------|----------|--------------------------|
| Logistic Regression    | 0.84      | 0.81      | 0.87     | Rápido                   |
| Random Forest          | 0.89      | 0.88      | 0.92     | Medio                    |
| Gradient Boosting      | 0.90      | 0.89      | 0.93     | Medio                    |
| **XGBoost**            | **0.92**  | **0.91**  | **0.95** | Medio                    |
| LightGBM               | 0.91      | 0.90      | 0.94     | Rápido                   |

➡️ **El modelo seleccionado fue XGBoost**, por ofrecer el mejor equilibrio entre rendimiento, consistencia y escalabilidad.

---

## 🧾 Requisitos

Asegúrate de tener instaladas las dependencias necesarias:

```bash
pip install -r requirements.txt

Librerías clave:

pandas, numpy, scikit-learn

xgboost, lightgbm

matplotlib, seaborn

streamlit, joblib

 Conclusiones

El uso de un pipeline estructurado permitió:

Reducir errores en el procesamiento manual de datos.

Aumentar la trazabilidad de los experimentos.

Escalar fácilmente hacia nuevos conjuntos de datos o métricas.

Aprovechar técnicas modernas como XGBoost y LightGBM para obtener una predicción robusta y generalizable.

Este proyecto demuestra cómo una correcta aplicación de prácticas MLOps mejora la eficiencia, reproducibilidad y valor real de los modelos predictivos en entornos empresariales.

Juan Manuel García Puerta
Proyecto académico desarrollado en el marco de la asignatura de Machine Learning y MLOps.
Facultad de Ingeniería de Sistemas.

SonarCloud
<img width="1914" height="924" alt="image" src="https://github.com/user-attachments/assets/d23f3799-367a-41cc-a59e-7b7334c6141e" />
