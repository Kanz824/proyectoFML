import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import json
import os

def build_datasets(data_path, features_path, target_col):
    # --- Leer el dataset limpio ---
    df = pd.read_csv(data_path)
    
    # --- Cargar lista de features seleccionadas ---
    with open(features_path, 'r') as f:
        features = json.load(f)

    # --- Separar variables predictoras (X) y objetivo (y) ---
    X = df[features]
    y = df[target_col]
    

    # --- Dividir datos en entrenamiento y prueba ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Identificar tipos de variables ---
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # Si tienes variables ordinales específicas, puedes listarlas aquí:
    ordinal_features = ['EducationField']  # Ejemplo, cámbialas según tu dataset

    # Asegurarte de que las ordinales estén en las categóricas
    categorical_features = [col for col in categorical_features if col not in ordinal_features]

    # --- Pipelines ---

    # Pipeline numérico: imputar y escalar
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Pipeline categórico nominal: imputar y one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Pipeline categórico ordinal: imputar y ordinal encode
    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder())
    ])

    # --- Column Transformer (ensamblar los pipelines) ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('ord', ordinal_transformer, ordinal_features)
        ]
    )

    print("✅ Preprocessor y datasets creados correctamente")
    print(f"🧩 Numéricas: {numeric_features}")
    print(f"🔤 Categóricas: {categorical_features}")
    print(f"🔢 Ordinales: {ordinal_features}")

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    # --- Rutas absolutas ---
    base_path = r'C:\Proyecto\mlops_pipeline\src'
    data_path = os.path.join(base_path, 'clean_data.csv')
    features_path = os.path.join(base_path, 'selected_features.json')

    # --- Ejecutar la función ---
    X_train, X_test, y_train, y_test, preprocessor = build_datasets(
        data_path=data_path,
        features_path=features_path,
        target_col='Attrition'
    )
