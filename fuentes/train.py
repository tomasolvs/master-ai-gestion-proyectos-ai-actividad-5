"""Entrena y compara regresión logística y Random Forest con MLflow."""
from pathlib import Path
import argparse, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
                             RocCurveDisplay, classification_report)
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datos/datos_limp/breast_cancer_limpio.csv"
OUT = ROOT / "evidencias"
OUT.mkdir(exist_ok=True)

def main(use_mlflow=True):
    df = pd.read_csv(DATA)
    X = df.drop(columns=["diagnostico", "id_registro"])
    y = df["diagnostico"].map({"maligno": 0, "benigno": 1})
    if y.isna().any(): raise ValueError("La variable objetivo contiene categorías inesperadas.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        "Regresion_Logistica": (
            Pipeline([("escalador", StandardScaler()),
                      ("modelo", LogisticRegression(max_iter=5000, random_state=42))]),
            {"modelo__C": [0.1, 1.0, 10.0],
             "modelo__class_weight": [None, "balanced"]}
        ),
        "Random_Forest": (
            Pipeline([("modelo", RandomForestClassifier(random_state=42, n_jobs=-1))]),
            {"modelo__n_estimators": [100, 200],
             "modelo__max_depth": [None, 5, 10],
             "modelo__min_samples_leaf": [1, 2]}
        )
    }
    mlflow = None
    if use_mlflow:
        try:
            import mlflow
            import mlflow.sklearn
        except ImportError as e:
            raise RuntimeError("MLflow no está instalado. Ejecuta: pip install -r requirements.txt") from e
        tracking_db = ROOT / "mlflow.db"

        mlflow.set_tracking_uri(
            f"sqlite:///{tracking_db}"
        )

        mlflow.set_experiment(
            "Actividad5_Comparacion_Clasificadores"
        )
    rows, fitted = [], {}
    for name, (pipe, grid) in models.items():
        run_context = mlflow.start_run(run_name=name) if use_mlflow else _NullContext()
        with run_context as run:
            search = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1, refit=True)
            search.fit(X_train, y_train)
            best = search.best_estimator_
            pred = best.predict(X_test)
            proba = best.predict_proba(X_test)[:, 1]
            metrics = {
                "accuracy": float(accuracy_score(y_test, pred)),
                "precision": float(precision_score(y_test, pred, zero_division=0)),
                "recall": float(recall_score(y_test, pred, zero_division=0)),
                "f1": float(f1_score(y_test, pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, proba)),
                "cv_f1_mean": float(search.best_score_),
                "cv_f1_std": float(search.cv_results_["std_test_score"][search.best_index_]),
            }
            rows.append({"modelo": name, **metrics,
                         "mejores_hiperparametros": json.dumps(search.best_params_, ensure_ascii=False)})
            fitted[name] = (best, pred, proba)
            if use_mlflow:
                mlflow.log_params({k: str(v) for k,v in search.best_params_.items()})
                mlflow.log_param("cv", "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)")
                mlflow.log_param("criterio_gridsearch", "f1")
                mlflow.log_param("random_state", 42)
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(best, artifact_path="modelo")
                report = classification_report(y_test, pred, output_dict=True, zero_division=0)
                pd.DataFrame(report).transpose().to_csv(OUT/f"reporte_{name}.csv")
    results = pd.DataFrame(rows).sort_values("f1", ascending=False)
    results.to_csv(OUT/"resultados_modelos.csv", index=False)
    # Confusion matrices
    for name, (model, pred, proba) in fitted.items():
        disp = ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=["maligno","benigno"])
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"Matriz de confusión — {name}")
        plt.tight_layout(); plt.savefig(OUT/f"matriz_confusion_{name}.png", dpi=160); plt.close()
    # ROC comparison
    fig, ax = plt.subplots(figsize=(7,5))
    for name, (model, pred, proba) in fitted.items():
        RocCurveDisplay.from_predictions(y_test, proba, name=name, ax=ax)
    ax.plot([0,1],[0,1],"--", label="Referencia aleatoria")
    ax.set_title("Curvas ROC — conjunto de prueba"); fig.tight_layout()
    fig.savefig(OUT/"curvas_roc.png", dpi=160); plt.close(fig)
    print(results.to_string(index=False))
    return results

class _NullContext:
    def __enter__(self): return None
    def __exit__(self, *args): return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-mlflow", action="store_true", help="Valida el entrenamiento sin registrar en MLflow")
    args = parser.parse_args()
    main(use_mlflow=not args.no_mlflow)
