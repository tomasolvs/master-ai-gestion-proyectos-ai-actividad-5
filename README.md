# Actividad 5 — Gestión de datos, entrenamiento y MLflow

* Nombre: Tomás Olivares
* Nombre del curso: Gestión de proyectos de inteligencia artificial
* Nombre del profesor: Luis Ariel Vázquez Piña
* Módulo: 2 - Semana: 6
* Fecha:  25/10/2026

[Notebook en Google Colab vinculado a este repo](https://colab.research.google.com/drive/1aT_pVbE2ckVUE96HTKsko-pcDsmVLCY9?usp=sharing)

## Objetivo
Preparar un conjunto de datos público, comparar dos algoritmos de clasificación, ajustar hiperparámetros con validación cruzada y registrar parámetros, métricas y modelos mediante MLflow.

## Dataset
Se utiliza el conjunto **Wisconsin Diagnostic Breast Cancer**, disponible en `scikit-learn`. Contiene 569 registros y 30 variables numéricas predictoras. La variable objetivo se expresa como `diagnostico`, con las clases `maligno` y `benigno`.

El dataset tiene fines académicos. Los resultados no deben utilizarse para diagnóstico ni decisiones clínicas.

## Estructura del proyecto

```text
Actividad5/
├── datos/
│   ├── datos_ini/          # Dataset original
│   └── datos_limp/         # Dataset preparado
├── evidencias/
│   ├── mlflow_detalle.txt
│   ├── mlflow_resultados.csv
│   ├── resultados_modelos.csv
│   └── capturas/           # Añadir capturas de MLflow
├── fuentes/
│   ├── datos_prep.py
│   └── train.py
├── entrena.ipynb
├── requirements.txt
├── README.md
├── CHANGELOG.md
└── reporte_tecnico.pdf
```

Los nombres de archivos y carpetas deben ajustarse a los que efectivamente se suban al repositorio. El ZIP de evidencias generado en Colab contiene `mlflow_detalle.txt`, `mlflow_resultados.csv`, `resultados_modelos.csv`, `reporte_Regresion_Logistica.csv` y los modelos `Regresion_Logistica.pkl` y `Random_Forest.pkl`.

## Entorno y ejecución

Se recomienda Google Colab con Python 3.10 o posterior. Para instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar el notebook `entrena.ipynb` de arriba hacia abajo. El flujo prepara los datos, divide el conjunto en entrenamiento y prueba, ajusta los modelos mediante `GridSearchCV` con validación cruzada estratificada de cinco particiones, calcula métricas en el conjunto de prueba y registra los resultados en MLflow.

Dependencias principales:
- pandas
- scikit-learn
- matplotlib
- mlflow

## Modelos evaluados
- **Regresión Logística**, con estandarización de variables y búsqueda de `C` y `class_weight`.
- **Random Forest**, con búsqueda de número de árboles, profundidad máxima y tamaño mínimo de hoja.

La partición de prueba es del 20 %, con `random_state=42`. La búsqueda de hiperparámetros optimiza F1.

## Resultados reportados
Los resultados de la ejecución documentada son:

| Métrica | Regresión Logística | Random Forest |
|---|---:|---:|
| Accuracy | 0.973684 | 0.956140 |
| Precision | 0.972603 | 0.958904 |
| Recall | 0.986111 | 0.972222 |
| F1 | 0.979310 | 0.965517 |
| ROC-AUC | 0.995701 | 0.993056 |

En esta partición, Regresión Logística obtiene mejores valores en las cinco métricas mostradas. Esta es una comparación descriptiva de una partición de prueba; no constituye por sí sola una prueba de significancia estadística.

## MLflow y trazabilidad
El experimento se llama `Actividad5_Comparacion_Clasificadores`. En la sesión de Colab documentada, el backend local se configuró como `sqlite:////content/mlflow.db`. MLflow registra parámetros, métricas y los archivos de modelo como artefactos. El archivo SQLite está dentro del entorno temporal de Colab; para conservar la trazabilidad fuera de la sesión, guardar las evidencias exportadas y las capturas de la interfaz.

Las ejecuciones históricas pueden incluir intentos fallidos y ejecuciones repetidas. Para el resumen final se muestran las ejecuciones `FINISHED` más recientes por modelo; no es necesario borrar el historial.

## Evidencias y entregables
1. Dataset original y dataset limpio.
2. Scripts de preparación y entrenamiento.
3. Notebook ejecutado con salidas visibles.
4. `README.md` y `CHANGELOG.md`.
5. Capturas de MLflow con las ejecuciones exitosas, métricas, parámetros y artefactos.
6. Reporte técnico en PDF.
7. ZIP final con el proyecto y sus evidencias.

## Fuente del dataset
UCI Machine Learning Repository, Wisconsin Diagnostic Breast Cancer; también está disponible a través de `sklearn.datasets.load_breast_cancer`:
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
