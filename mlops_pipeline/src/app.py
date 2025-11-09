import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Monitoreo de Data Drift", layout="wide")

st.title("📊 Monitoreo de Data Drift del Modelo")
st.markdown("""
Esta aplicación permite visualizar métricas de **drift** y detectar desviaciones significativas 
entre la población histórica y los datos actuales del modelo.
""")

# Cargar el archivo más reciente generado por model_monitoring.py
base_path = r"C:\Proyecto\mlops_pipeline\src"
files = [f for f in os.listdir(base_path) if f.startswith("data_drift_report")]
if not files:
    st.error("❌ No se encontraron reportes de drift. Ejecuta model_monitoring.py primero.")
    st.stop()

latest = max(files)
st.sidebar.success(f"Reporte más reciente: {latest}")

df = pd.read_csv(os.path.join(base_path, latest))

# Filtro de tipo
tipo = st.sidebar.multiselect("Filtrar por tipo de variable", df["Tipo"].unique(), default=df["Tipo"].unique())
df_filtered = df[df["Tipo"].isin(tipo)]

st.dataframe(df_filtered)

# Gráficos comparativos
st.subheader("📈 Métricas de Drift por Variable")

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
sns.barplot(x="Variable", y="PSI", data=df_filtered, ax=axes[0], palette="coolwarm")
axes[0].set_title("PSI (Population Stability Index)")
axes[0].tick_params(axis='x', rotation=45)

sns.barplot(x="Variable", y="KS_Statistic", data=df_filtered, ax=axes[1], palette="viridis")
axes[1].set_title("Kolmogorov-Smirnov Statistic")
axes[1].tick_params(axis='x', rotation=45)

sns.barplot(x="Variable", y="JSD", data=df_filtered, ax=axes[2], palette="magma")
axes[2].set_title("Jensen-Shannon Divergence")
axes[2].tick_params(axis='x', rotation=45)

st.pyplot(fig)

# Alertas
st.subheader("🚨 Indicadores de Alerta")
critical = df_filtered[df_filtered["Alerta"] == "🔴 Crítico"]
moderate = df_filtered[df_filtered["Alerta"] == "🟡 Moderado"]

if not critical.empty:
    st.error(f"⚠️ Se detectaron {len(critical)} variables con drift crítico.")
    st.dataframe(critical)
elif not moderate.empty:
    st.warning(f"🟡 Hay {len(moderate)} variables con drift moderado.")
else:
    st.success("🟢 No se detectaron desviaciones significativas.")

st.markdown("---")
st.markdown("**Sugerencias:** Si existen variables con drift crítico, considera reentrenar el modelo o revisar las fuentes de datos.")
