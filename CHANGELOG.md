# Changelog

## 1.0.0 — 2026-10-25
- Se incorpora el dataset público Wisconsin Diagnostic Breast Cancer exportado a CSV.
- Se agrega limpieza reproducible, estandarización de etiquetas y detección de duplicados exactos.
- Se incorporan dos pipelines: regresión logística con escalado y Random Forest.
- Se implementa Grid Search con validación cruzada estratificada de cinco particiones.
- Se registran métricas, hiperparámetros y modelos mediante MLflow.
- Se agregan matrices de confusión, curva ROC y reporte técnico.
- Se fija `random_state=42` para facilitar la reproducción de resultados.
