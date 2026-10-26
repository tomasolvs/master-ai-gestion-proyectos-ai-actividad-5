# Actividad 5 — Gestión de datos, entrenamiento y MLflow

* Nombre: Tomás Olivares
* Nombre del curso: Gestión de proyectos de inteligencia artificial
* Nombre del profesor: Luis Ariel Vázquez Piña
* Módulo: 2 - Semana: 6
* Fecha:  25/10/2026

[Notebook en Google Colab vinculado a este repo](https://colab.research.google.com/drive/1aT_pVbE2ckVUE96HTKsko-pcDsmVLCY9?usp=sharing)

## Objetivo
Preparar y versionar un dataset público, entrenar dos clasificadores, ajustar hiperparámetros con Grid Search y validación cruzada, y registrar experimentos con MLflow.

## Dataset
Wisconsin Diagnostic Breast Cancer, disponible mediante `sklearn.datasets.load_breast_cancer`. El conjunto incluye 569 observaciones y 30 variables numéricas derivadas de imágenes de aspirados de masa mamaria. La variable objetivo se expresa como `maligno` o `benigno`. Fuente de referencia: UCI Machine Learning Repository, *Breast Cancer Wisconsin (Diagnostic)*: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic ; documentación de scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

## Estructura
- `datos/datos_ini/`: archivo original exportado del dataset público.
- `datos/datos_limp/`: archivo estandarizado para el análisis.
- `fuentes/datos_prep.py`: función de limpieza y control de calidad.
- `fuentes/train.py`: Grid Search, evaluación y registro de experimentos.
- `entrena.ipynb`: notebook de preparación, entrenamiento y evaluación.
- `evidencias/`: métricas, tablas y gráficos generados.
- `reportes/`: reporte técnico PDF.
- `CHANGELOG.md`: historial de cambios.

## Ejecución
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python fuentes/datos_prep.py
```
La función de limpieza se puede ejecutar desde un notebook o desde Python importando `limpiar_dataset`.

Para entrenar y registrar los experimentos en MLflow:
```bash
python fuentes/train.py
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```
Abre `http://127.0.0.1:5000`. El script registra parámetros, métricas y modelo como artefacto en cada ejecución. Los experimentos usan `random_state=42`, partición estratificada 80/20 y validación cruzada estratificada de cinco folds.

Para ejecutar la evaluación sin MLflow (solo como prueba técnica):
```bash
python fuentes/train.py --no-mlflow
```

## Decisiones metodológicas
- Se compara regresión logística con Random Forest.
- Se utiliza F1 como criterio de selección en Grid Search para equilibrar precisión y sensibilidad.
- Se reportan accuracy, precision, recall, F1 y ROC-AUC.
- La partición de prueba se reserva para evaluación final y no se usa para elegir hiperparámetros.
- El dataset no contiene valores nulos en las variables usadas. La limpieza elimina duplicados exactos si los hubiera; no se inventan valores ni se imputan datos inexistentes.
- El dataset es educativo y no debe usarse para diagnósticos médicos.

## Reproducibilidad y versionamiento
Consulta `CHANGELOG.md`. Los archivos CSV se conservan en el repositorio y los cambios de código se controlan con Git. Para datasets mayores, puede incorporarse DVC. Evita publicar datos personales o sensibles en repositorios públicos.
