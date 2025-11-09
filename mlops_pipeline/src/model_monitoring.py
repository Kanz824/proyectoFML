import os
import json
import joblib
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chisquare, entropy
from sklearn.preprocessing import LabelEncoder
from datetime import datetime


# -------------------------------------------------------
# Funciones de cálculo de Data Drift
# -------------------------------------------------------

def psi(expected, actual, buckets=10):
    """Population Stability Index (PSI)."""
    def scale_range(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-6)

    expected_perc = np.histogram(scale_range(expected), bins=buckets)[0] / len(expected)
    actual_perc = np.histogram(scale_range(actual), bins=buckets)[0] / len(actual)
    psi_val = np.sum((expected_perc - actual_perc) * np.log((expected_perc + 1e-6) / (actual_perc + 1e-6)))
    return psi_val


def jensen_shannon(p, q):
    """Jensen-Shannon divergence entre dos distribuciones."""
    p = np.array(p) / np.sum(p)
    q = np.array(q) / np.sum(q)
    m = (p + q) / 2
    return (entropy(p, m) + entropy(q, m)) / 2


def calculate_drift_metrics(df_old, df_new):
    """Calcula métricas de drift para todas las variables."""
    metrics = []

    for col in df_old.columns:
        if df_old[col].dtype in ['float64', 'int64']:
            ks_stat, ks_pvalue = ks_2samp(df_old[col], df_new[col])
            psi_val = psi(df_old[col], df_new[col])
            js_val = jensen_shannon(np.histogram(df_old[col], bins=20)[0],
                                    np.histogram(df_new[col], bins=20)[0])

            metrics.append({
                'Variable': col,
                'Tipo': 'Numérica',
                'KS_Statistic': ks_stat,
                'PSI': psi_val,
                'JSD': js_val,
                'Chi2': np.nan
            })
        else:
            le = LabelEncoder()
            old_enc = le.fit_transform(df_old[col].astype(str))
            new_enc = le.transform(df_new[col].astype(str))
            chi2_val, _ = chisquare(np.bincount(old_enc), np.bincount(new_enc))

            metrics.append({
                'Variable': col,
                'Tipo': 'Categórica',
                'KS_Statistic': np.nan,
                'PSI': np.nan,
                'JSD': np.nan,
                'Chi2': chi2_val
            })
    return pd.DataFrame(metrics)


# -------------------------------------------------------
# Simulación de monitoreo
# -------------------------------------------------------

def run_data_drift_monitoring():
    base_path = r"C:\Proyecto\mlops_pipeline\src"
    data_path = os.path.join(base_path, "clean_data.csv")

    # Cargar dataset base
    df = pd.read_csv(data_path)

    # Simular nuevos datos (con ruido)
    df_new = df.copy()
    for col in df_new.select_dtypes(include=[np.number]).columns:
        df_new[col] = df_new[col] * np.random.uniform(0.9, 1.1, size=len(df_new))

    # Calcular métricas de drift
    metrics_df = calculate_drift_metrics(df, df_new)

    # Detectar alertas
    metrics_df['Alerta'] = metrics_df.apply(lambda x:
                                            '🔴 Crítico' if (x['PSI'] > 0.25 or x['KS_Statistic'] > 0.1)
                                            else ('🟡 Moderado' if x['PSI'] > 0.1 else '🟢 Estable'), axis=1)

    # Guardar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(base_path, f"data_drift_report_{timestamp}.csv")
    metrics_df.to_csv(output_path, index=False)

    print("✅ Monitoreo completado.")
    print(f"📄 Reporte guardado en: {output_path}")

    # Retornar para visualización posterior
    return metrics_df


if __name__ == "__main__":
    results = run_data_drift_monitoring()
    print("\n📊 Resultados de drift:")
    print(results)
