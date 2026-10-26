# Datos del proyecto

`datos_ini/breast_cancer_original.csv` es una exportación del dataset público Wisconsin Diagnostic Breast Cancer disponible mediante scikit-learn.
`datos_limp/breast_cancer_limpio.csv` es la versión normalizada por `fuentes/datos_prep.py`.

Variables predictoras: 30 medidas numéricas de características de núcleos celulares. `id_registro` identifica filas y se excluye del entrenamiento. `diagnostico` es la variable objetivo (`maligno` o `benigno`).

No hay datos personales del usuario. El dataset se usa solo para fines educativos, no clínicos.
