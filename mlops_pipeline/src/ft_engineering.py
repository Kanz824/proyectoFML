# ft_engineering.py
# -------------------------------------------------
# Este módulo realiza la ingeniería de características
# y retorna los conjuntos de entrenamiento y evaluación
# listos para los modelos supervisados.
# -------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def feature_engineering(df, target_col, ordinal_cols=None, categorical_cols=None, numeric_cols=None):
    """
    Realiza imputación, codificación y escalado sobre el dataset.
    Retorna X_train, X_test, y_train, y_test listos para entrenar modelos.
    """

    # --- Separar variables predictoras y objetivo ---
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # --- Definir transformadores individuales ---
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder())
    ])

    # --- Combinar los transformadores ---
    preprocessor = ColumnTransformer(transformers=[
        ('numeric', numeric_transformer, numeric_cols),
        ('categoric', categorical_transformer, categorical_cols),
        ('categoric_ordinales', ordinal_transformer, ordinal_cols)
    ])

    # --- Dividir datos ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- Aplicar transformaciones ---
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test, preprocessor


# Ejemplo de ejecución
if __name__ == "__main__":
    df = pd.read_csv('data/processed/train.csv')

    numeric_cols = ['edad', 'ingresos', 'puntaje_credito']
    categorical_cols = ['genero', 'ocupacion']
    ordinal_cols = ['nivel_educativo']  # por ejemplo: primaria < secundaria < universitaria

    X_train, X_test, y_train, y_test, preprocessor = feature_engineering(
        df,
        target_col='Estado',
        ordinal_cols=ordinal_cols,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols
    )

    print("Shape de X_train:", X_train.shape)
    print("Shape de X_test:", X_test.shape)
