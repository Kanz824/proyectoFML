import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report
)
from sklearn.model_selection import cross_val_score

# Modelos base
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Importa función de ft_engineering.py
from ft_engineering import build_datasets


# -------------------------------------------------------------
# Función auxiliar: resumen de métricas del modelo
# -------------------------------------------------------------
def summarize_classification(y_true, y_pred, model_name, clf, X_train, y_train):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    cv = cross_val_score(clf, X_train, y_train, cv=5).mean()

    return {
        'Modelo': model_name,
        'Accuracy': acc,
        'F1-score': f1,
        'Precision': prec,
        'Recall': rec,
        'CV_Score': cv
    }


# -------------------------------------------------------------
# Función para construir, entrenar y evaluar un modelo
# -------------------------------------------------------------
def build_model(model_name, model, preprocessor, X_train, y_train, X_test, y_test):
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics = summarize_classification(y_test, y_pred, model_name, clf, X_train, y_train)

    print(f"✅ {model_name} entrenado con éxito:")
    print(f"   Accuracy: {metrics['Accuracy']:.3f} | F1: {metrics['F1-score']:.3f} | CV: {metrics['CV_Score']:.3f}\n")

    return metrics, clf


# -------------------------------------------------------------
# Entrenamiento y evaluación general
# -------------------------------------------------------------
def train_and_evaluate_models():
    base_path = r'C:\Proyecto\mlops_pipeline\src'
    data_path = os.path.join(base_path, 'clean_data.csv')
    features_path = os.path.join(base_path, 'selected_features.json')

    # Cargar datasets
    X_train, X_test, y_train, y_test, preprocessor = build_datasets(
        data_path=data_path,
        features_path=features_path,
        target_col='Attrition'
    )

    # Definición de modelos
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "LightGBM": LGBMClassifier(random_state=42)
    }

    results = []
    trained_models = {}

    print("\n🚀 Iniciando entrenamiento y evaluación de modelos...\n")

    for name, model in models.items():
        metrics, clf = build_model(name, model, preprocessor, X_train, y_train, X_test, y_test)
        results.append(metrics)
        trained_models[name] = clf

    # Convertir resultados a DataFrame
    results_df = pd.DataFrame(results).sort_values(by='F1-score', ascending=False)
    print("\n📊 Resultados comparativos de modelos:\n")
    print(results_df)

    # Gráfico comparativo
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Modelo', y='F1-score', data=results_df, palette='viridis')
    plt.title('Comparación de modelos por F1-score', fontsize=14)
    plt.ylabel('F1-score')
    plt.xlabel('Modelo')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

    # Seleccionar mejor modelo
    best_model_name = results_df.iloc[0]['Modelo']
    best_model = trained_models[best_model_name]

    # Guardar modelo final
    model_path = os.path.join(base_path, f"best_model_{best_model_name}.pkl")
    joblib.dump(best_model, model_path)

    print(f"\n🏆 Mejor modelo: {best_model_name}")
    print(f"📦 Modelo guardado en: {model_path}")

    # Reporte detallado
    y_pred_final = best_model.predict(X_test)
    print("\n📋 Reporte de clasificación del mejor modelo:\n")
    print(classification_report(y_test, y_pred_final))

    # Guardar tabla resumen
    summary_path = os.path.join(base_path, "model_evaluation_summary.csv")
    results_df.to_csv(summary_path, index=False)
    print(f"\n📄 Tabla resumen guardada en: {summary_path}")

    return results_df, best_model


# -------------------------------------------------------------
# Ejecución principal
# -------------------------------------------------------------
if __name__ == "__main__":
    results_df, best_model = train_and_evaluate_models()
