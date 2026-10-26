# Actividad 5 — Gestión de datos, entrenamiento y MLflow

* Nombre: Tomás Olivares
* Nombre del curso: Gestión de proyectos de inteligencia artificial
* Nombre del profesor: Luis Ariel Vázquez Piña
* Módulo: 2 - Semana: 6
* Fecha:  25/10/2026

[Notebook en Google Colab vinculado a este repo](https://colab.research.google.com/drive/1aT_pVbE2ckVUE96HTKsko-pcDsmVLCY9?usp=sharing)

## Descripción

Proyecto de clasificación con el conjunto de datos Wisconsin Diagnostic Breast Cancer. Se prepararon los datos, se entrenaron y compararon Regresión Logística y Random Forest, se ajustaron hiperparámetros mediante validación cruzada y se registraron parámetros, métricas y modelos con MLflow.

## Estructura del proyecto

Actividad5/
├── datos/
│   ├── datos_ini/
│   │   └── breast_cancer_original.csv
│   ├── datos_limp/
│   │   └── breast_cancer_limpio.csv
│   └── README.md
├── evidencias/
│   ├── mlflow_detalle.txt
│   ├── mlflow_resultados.csv
│   ├── Random_Forest.pkl
│   ├── Regression_Logistica.pkl
│   ├── reporte_Regression_Logistica.csv
│   └── resultados_modelos.csv
├── fuentes/
│   ├── datos_prep.py
│   └── train.py
├── CHANGELOG.md
├── README.md
├── requirements.txt
└── Notebook de la actividad

## Dataset

Se utiliza el conjunto Wisconsin Diagnostic Breast Cancer. El archivo limpio contiene 569 registros y 30 variables predictoras, con la variable objetivo `diagnostico`.

La revisión del archivo limpio reportó:
- Valores nulos: 0
- Duplicados exactos: 0
- Registros benignos: 357
- Registros malignos: 212

## Modelos y evaluación

Se compararon dos clasificadores mediante una partición de prueba estratificada y GridSearchCV con validación cruzada estratificada de cinco particiones.

| Métrica | Regresión Logística | Random Forest |
|---|---:|---:|
| Accuracy | 0.973684 | 0.956140 |
| Precision | 0.972603 | 0.958904 |
| Recall | 0.986111 | 0.972222 |
| F1 | 0.979310 | 0.965517 |
| ROC-AUC | 0.995701 | 0.993056 |

Regresión Logística obtuvo mejores resultados en las cinco métricas reportadas. La comparación es descriptiva y corresponde a la partición de prueba documentada en el notebook.

## Entorno e instalación

El proyecto se ejecutó en Google Colab. Las dependencias se encuentran en `requirements.txt`.

Para instalarlas desde Colab, con el archivo disponible en `/content`:

```python
%pip install -r /content/requirements.txt