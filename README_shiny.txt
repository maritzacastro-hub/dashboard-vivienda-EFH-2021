Dashboard EFH 2021 en Shiny for Python

Archivos:
- app_shiny.py : aplicación principal en Shiny for Python.
- requirements_shiny.txt : dependencias mínimas sugeridas.

Estructura esperada en la misma carpeta del app:
- app_shiny.py
- public_results/
    - eda_summary.csv
    - eda_meta.csv
    - eda_num_stats.csv
    - eda_num_hist.csv.gz
    - eda_cat_counts.csv.gz
    - eda_biv_num.csv
    - eda_biv_cat.csv.gz
    - eda_corr.csv.gz
    - metrics_by_fold.csv
    - model_lr_18.joblib

Instalación sugerida:
1) pip install -r requirements_shiny.txt
2) shiny run --reload app_shiny.py

Qué cambia respecto al dashboard original:
- La app abre en una portada académica, no en la calculadora.
- La calculadora se mantiene como pestaña propia.
- Se separan descriptivos, correlaciones, modelos y conclusiones.
- La estética es más sobria y apropiada para exposición.

Notas:
- Si faltan archivos en public_results, la app no debería caerse: mostrará mensajes de ausencia de datos.
- La calculadora aplica la misma lógica base del dashboard original: transformación log(1+x) para montos monetarios y validación opcional de coherencia.
