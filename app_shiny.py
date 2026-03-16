from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shinywidgets import output_widget, render_widget


BASE_DIR = Path(__file__).resolve().parent
PUBLIC_RESULTS = BASE_DIR / "public_results"
MODEL_PATH = PUBLIC_RESULTS / "model_lr_18.joblib"

HOGARES_UNICOS_ANALISIS = 2313
PCT_VIV_PROPIA_HOME = 66.07
MODELO_FINAL_HOME = "Regresión logística binaria"

ORDERED_VARS = [
    "act_fijo",
    "act_var",
    "cap_pen_ent",
    "edad_pr",
    "hr_trabajadas_pr",
    "yoprinm_pr",
    "ypenh",
    "ysubh",
    "t_cc",
    "t_tbco",
    "u_cheq",
    "u_pac",
    "u_pat",
    "u_tbco",
    "u_prepago",
    "est_civil_pr",
    "numh",
    "ocuph",
]

NUMERIC_VARS = [
    "act_fijo",
    "act_var",
    "cap_pen_ent",
    "edad_pr",
    "hr_trabajadas_pr",
    "yoprinm_pr",
    "ypenh",
    "ysubh",
]

BINARY_VARS = [
    "t_cc",
    "t_tbco",
    "u_cheq",
    "u_pac",
    "u_pat",
    "u_tbco",
    "u_prepago",
]

CATEGORICAL_VARS = ["est_civil_pr", "numh", "ocuph"]

VAR_LABELS = {
    "yoprinm_pr": "Ingreso mensual de la persona de referencia del hogar",
    "act_fijo": "Monto total de activos financieros de renta fija",
    "act_var": "Monto invertido por el hogar en instrumentos de renta variable",
    "cap_pen_ent": "Saldo en cuenta de capitalización individual",
    "edad_pr": "Edad de la persona de referencia",
    "hr_trabajadas_pr": "Horas trabajadas por la persona de referencia",
    "ypenh": "Ingreso mensual del hogar por pensiones",
    "ysubh": "Ingreso mensual del hogar por subsidios",
    "numh": "Número de miembros del hogar",
    "ocuph": "Número de miembros del hogar que trabajan",
    "u_pac": "Uso de pago automático a cuenta corriente",
    "t_tbco": "Posee tarjetas de crédito bancarias",
    "u_pat": "Uso de pago automático a tarjetas de crédito",
    "u_tbco": "Uso de tarjetas de crédito bancarias como medio de pago",
    "u_cheq": "Uso de cheques como medio de pago",
    "t_cc": "Posee cuenta corriente",
    "u_prepago": "Uso de instrumentos de prepago",
    "est_civil_pr": "Estado civil de la persona de referencia",
}

ESTADO_CIVIL = [
    "Soltero(a)",
    "Casado(a)",
    "Conviviente o pareja",
    "Divorciado(a)",
    "Viudo(a)",
    "Separado(a)",
]

VARIABLE_CHOICES = {v: VAR_LABELS.get(v, v) for v in ORDERED_VARS}

APP_CSS = """
:root {
  --accent: #163b65;
  --accent-soft: #edf4fb;
  --muted: #6c757d;
}
.navbar-brand {
  font-weight: 700;
}
.card, .value-box {
  border-radius: 1rem !important;
}
.section-note {
  background: #f8fafc;
  border: 1px solid #dbe6f2;
  border-radius: 14px;
  padding: 14px 16px;
  margin-bottom: 1rem;
}
.hero-box {
  background: linear-gradient(135deg, #163b65 0%, #255d94 100%);
  color: white;
  border-radius: 18px;
  padding: 1.35rem 1.5rem;
  margin-bottom: 1rem;
}
.hero-box h2 {
  margin: 0 0 .35rem 0;
  font-size: 1.65rem;
}
.helper-text {
  color: var(--muted);
  font-size: .95rem;
}
.metric-caption {
  color: var(--muted);
  font-size: .9rem;
}
.prob-badge {
  display: inline-block;
  padding: .45rem .8rem;
  border-radius: 999px;
  font-weight: 700;
}
.prob-low { background: #fdecec; color: #8b1e1e; }
.prob-mid { background: #fff4db; color: #8b5e00; }
.prob-high { background: #e9f8ef; color: #13653a; }

.value-box .card-title,
.value-box .value-box-title {
  font-size: 1.25rem !important;
  font-weight: 600 !important;
}

.value-box .value-box-value {
  font-size: 2.3rem !important;
  font-weight: 700 !important;
}

.prob-main-value {
  font-size: 2.8rem;
  font-weight: 700;
  color: #163b65;
  line-height: 1.1;
}

.calc-card-wrap {
  height: 100%;
}

.calc-card-wrap .card {
  min-height: 170px;
  height: 100%;
}

.prob-card-wrap .card-header {
  background: #163b65;
  color: white;
  font-weight: 700;
}

.interp-card-wrap .card-header {
  background: #edf4fb;
  color: #163b65;
  font-weight: 700;
}

.prob-bar-wrap {
  width: 100%;
  height: 18px;
  background: #e9ecef;
  border-radius: 999px;
  overflow: hidden;
  margin-top: 14px;
}

.prob-bar-fill {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, #7b2cbf 0%, #3a0ca3 100%);
  transition: width 0.3s ease;
}
"""


def money_to_model(x: float | int) -> float:
    return float(np.log1p(max(float(x), 0.0)))


def read_csv_if_exists(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def _strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _rename_first_match(df: pd.DataFrame, target: str, candidates: list[str]) -> pd.DataFrame:
    if target in df.columns:
        return df
    for c in candidates:
        if c in df.columns:
            return df.rename(columns={c: target})
    return df

def normalize_model_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Modelo" not in df.columns:
        return df

    df = df.copy()
    df["Modelo"] = (
        df["Modelo"]
        .astype(str)
        .str.strip()
        .str.replace(r"^MLP seleccionadas(?:\s*\(Group-CV\))?$", "MLP", regex=True)
        .str.replace(r"^LR Ridge \(splines\)(?:\s*\(Group-CV\))?$", "Regresión logística binaria", regex=True)
    )
    return df

def normalize_public_tables(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    for key, df in list(data.items()):
        if isinstance(df, pd.DataFrame) and not df.empty:
            data[key] = _strip_cols(df)

    meta = data.get("meta", pd.DataFrame())
    if not meta.empty:
        meta = _rename_first_match(meta, "var", ["variable", "Variable", "VAR", "feature", "col", "name"])
        meta = _rename_first_match(meta, "type", ["tipo", "Tipo", "TYPE", "dtype", "var_type"])
    else:
        meta = pd.DataFrame(columns=["var", "type"])
    data["meta"] = meta

    cat_counts = data.get("cat_counts", pd.DataFrame())
    if not cat_counts.empty:
        cat_counts = _rename_first_match(cat_counts, "var", ["variable", "Variable", "VAR", "feature", "name"])
        cat_counts = _rename_first_match(cat_counts, "category", ["categoria", "Categoría", "cat", "nivel", "level", "value"])
        cat_counts = _rename_first_match(cat_counts, "n", ["count", "Count", "freq", "frequency", "N"])
        if "macrozona" not in cat_counts.columns:
            cat_counts["macrozona"] = "All"
    else:
        cat_counts = pd.DataFrame(columns=["macrozona", "var", "category", "n"])
    data["cat_counts"] = cat_counts

    num_hist = data.get("num_hist", pd.DataFrame())
    if not num_hist.empty:
        num_hist = _rename_first_match(num_hist, "var", ["variable", "Variable", "VAR", "feature", "name"])
        num_hist = _rename_first_match(num_hist, "bin_left", ["left", "bin_l", "lower", "li", "min"])
        num_hist = _rename_first_match(num_hist, "bin_right", ["right", "bin_r", "upper", "ls", "max"])
        num_hist = _rename_first_match(num_hist, "count", ["n", "Count", "freq", "frequency", "N"])
        if "macrozona" not in num_hist.columns:
            num_hist["macrozona"] = "All"
    else:
        num_hist = pd.DataFrame(columns=["macrozona", "var", "bin_left", "bin_right", "count"])
    data["num_hist"] = num_hist

    num_stats = data.get("num_stats", pd.DataFrame())
    if not num_stats.empty:
        num_stats = _rename_first_match(num_stats, "var", ["variable", "Variable", "VAR", "feature", "name"])
        if "macrozona" not in num_stats.columns:
            num_stats["macrozona"] = "All"
    else:
        num_stats = pd.DataFrame(columns=["macrozona", "var"])
    data["num_stats"] = num_stats

    for nm in ["biv_num", "biv_cat"]:
        df = data.get(nm, pd.DataFrame())
        if not df.empty:
            df = _rename_first_match(df, "var", ["variable", "Variable", "VAR", "feature", "name"])
            if "macrozona" not in df.columns:
                df["macrozona"] = "All"
        else:
            df = pd.DataFrame(columns=["macrozona", "var"])
        data[nm] = df

    corr_long = data.get("corr_long", pd.DataFrame())
    if not corr_long.empty:
        corr_long = _rename_first_match(corr_long, "var1", ["x", "v1", "variable1", "Variable1"])
        corr_long = _rename_first_match(corr_long, "var2", ["y", "v2", "variable2", "Variable2"])
        corr_long = _rename_first_match(corr_long, "corr", ["cor", "correlacion", "correlation", "value"])
    else:
        corr_long = pd.DataFrame(columns=["var1", "var2", "corr"])
    data["corr_long"] = corr_long

    return data


def load_app_data(base_dir: Path) -> dict[str, Any]:
    data: dict[str, Any] = {
        "summary": pd.DataFrame({"n_registros": [0], "pct_viv_propia": [0.0]}),
        "meta": pd.DataFrame(columns=["var", "type"]),
        "num_stats": pd.DataFrame(),
        "num_hist": pd.DataFrame(),
        "cat_counts": pd.DataFrame(),
        "biv_num": pd.DataFrame(),
        "biv_cat": pd.DataFrame(),
        "corr_long": pd.DataFrame(),
        "metrics_fold": pd.DataFrame(),
        "have_eda": False,
        "have_metrics": False,
        "eda_error": None,
        "metrics_error": None,
        "have_model": False,
        "model_error": None,
        "model": None,
    }

    try:
        data["summary"] = read_csv_if_exists(base_dir / "eda_summary.csv")
        data["meta"] = read_csv_if_exists(base_dir / "eda_meta.csv")
        data["num_stats"] = read_csv_if_exists(base_dir / "eda_num_stats.csv")
        data["num_hist"] = read_csv_if_exists(base_dir / "eda_num_hist.csv.gz", compression="gzip")
        data["cat_counts"] = read_csv_if_exists(base_dir / "eda_cat_counts.csv.gz", compression="gzip")
        data["biv_num"] = read_csv_if_exists(base_dir / "eda_biv_num.csv")
        data["biv_cat"] = read_csv_if_exists(base_dir / "eda_biv_cat.csv.gz", compression="gzip")
        data["corr_long"] = read_csv_if_exists(base_dir / "eda_corr.csv.gz", compression="gzip")
        data = normalize_public_tables(data)
        data["have_eda"] = not data["summary"].empty
    except Exception as e:  # pragma: no cover - defensive fallback
        data["eda_error"] = str(e)

    try:
        data["metrics_fold"] = read_csv_if_exists(base_dir / "metrics_by_fold.csv")
        data["metrics_fold"] = normalize_model_labels(data["metrics_fold"])
        data["have_metrics"] = not data["metrics_fold"].empty
    except Exception as e:  # pragma: no cover - defensive fallback
        data["metrics_error"] = str(e)

    try:
        if MODEL_PATH.exists():
            data["model"] = joblib.load(MODEL_PATH)
            data["have_model"] = True
        else:
            data["model_error"] = f"No se encontró el modelo en {MODEL_PATH}"
    except Exception as e:  # pragma: no cover - defensive fallback
        data["model_error"] = str(e)

    meta = data["meta"]
    if not meta.empty and {"var", "type"}.issubset(meta.columns):
        available_vars = set(meta["var"].astype(str).tolist())
    else:
        available_vars = set(ORDERED_VARS)
    data["available_vars"] = [v for v in ORDERED_VARS if v in available_vars]
    return data


DATA = load_app_data(PUBLIC_RESULTS)


def label_var(v: str) -> str:
    return VAR_LABELS.get(v, v)


def empty_figure(message: str):
    fig = px.scatter()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 16},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(template="plotly_white", height=400)
    return fig


def pseudo_series_from_hist(var_name: str) -> pd.Series:
    h = DATA["num_hist"]
    if h.empty:
        return pd.Series([], dtype=float, name=var_name)
    h = h[(h["macrozona"] == "All") & (h["var"] == var_name)].copy()
    if h.empty:
        return pd.Series([], dtype=float, name=var_name)
    h["count"] = pd.to_numeric(h["count"], errors="coerce").fillna(0).astype(int)
    h["bin_left"] = pd.to_numeric(h["bin_left"], errors="coerce")
    h["bin_right"] = pd.to_numeric(h["bin_right"], errors="coerce")
    h = h.dropna(subset=["bin_left", "bin_right"])
    if h.empty:
        return pd.Series([], dtype=float, name=var_name)
    h["bin_mid"] = (h["bin_left"] + h["bin_right"]) / 2.0
    vals = np.repeat(h["bin_mid"].to_numpy(), h["count"].to_numpy())
    return pd.Series(vals, name=var_name)


def categorical_table(var_name: str) -> pd.DataFrame:
    raw = DATA["cat_counts"]
    if raw.empty:
        return pd.DataFrame(columns=["category", "n", "porcentaje"])
    raw = raw[(raw["macrozona"] == "All") & (raw["var"] == var_name)].copy()
    if raw.empty:
        return pd.DataFrame(columns=["category", "n", "porcentaje"])
    raw["category"] = raw["category"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    raw["n"] = raw["n"].astype(str).str.replace(r"[^\d\.-]", "", regex=True)
    raw["n"] = pd.to_numeric(raw["n"], errors="coerce").fillna(0).astype(int)
    raw = raw.groupby("category", as_index=False)["n"].sum()
    raw["category_num"] = pd.to_numeric(raw["category"], errors="coerce")
    raw = raw.sort_values(["category_num", "n"], ascending=[True, False]).drop(columns=["category_num"])
    den = int(raw["n"].sum())
    raw["porcentaje"] = 0 if den == 0 else (raw["n"] / den * 100).round(2)
    return raw


def get_numeric_summary(var_name: str) -> dict[str, float] | None:
    num_stats = DATA["num_stats"]
    if num_stats.empty:
        return None
    row = num_stats[(num_stats["macrozona"] == "All") & (num_stats["var"] == var_name)]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def prepare_input_row(values: dict[str, Any]) -> pd.DataFrame:
    row = {
        "act_fijo": money_to_model(values["act_fijo"]),
        "act_var": money_to_model(values["act_var"]),
        "cap_pen_ent": money_to_model(values["cap_pen_ent"]),
        "edad_pr": int(values["edad_pr"]),
        "hr_trabajadas_pr": int(values["hr_trabajadas_pr"]),
        "yoprinm_pr": money_to_model(values["yoprinm_pr"]),
        "ypenh": money_to_model(values["ypenh"]),
        "ysubh": money_to_model(values["ysubh"]),
        "t_cc": int(values["t_cc"]),
        "t_tbco": int(values["t_tbco"]),
        "u_cheq": int(values["u_cheq"]),
        "u_pac": int(values["u_pac"]),
        "u_pat": int(values["u_pat"]),
        "u_tbco": int(values["u_tbco"]),
        "u_prepago": int(values["u_prepago"]),
        "est_civil_pr": values["est_civil_pr"],
        "numh": int(values["numh"]),
        "ocuph": int(values["ocuph"]),
    }
    df = pd.DataFrame([row])
    return df[ORDERED_VARS]


def metric_summary_table(df_metrics: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["AUC", "Sensibilidad", "Especificidad", "Precision"] if c in df_metrics.columns]
    if df_metrics.empty or not cols:
        return pd.DataFrame()
    out = df_metrics.groupby("Modelo", dropna=False)[cols].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


summary = DATA["summary"]
summary_n = int(summary.loc[0, "n_registros"]) if not summary.empty and "n_registros" in summary.columns else 0
summary_pct = PCT_VIV_PROPIA_HOME
metrics_summary = metric_summary_table(DATA["metrics_fold"])
best_model_name = (
    DATA["metrics_fold"].groupby("Modelo")["AUC"].mean().idxmax()
    if DATA["have_metrics"] and "AUC" in DATA["metrics_fold"].columns and not DATA["metrics_fold"].empty
    else "Regresión logística Ridge"
)
best_auc_value = (
    float(DATA["metrics_fold"].groupby("Modelo")["AUC"].mean().max())
    if DATA["have_metrics"] and "AUC" in DATA["metrics_fold"].columns and not DATA["metrics_fold"].empty
    else np.nan
)


app_ui = ui.page_navbar(
    ui.nav_panel(
        "Inicio",
        ui.div(
            {"class": "hero-box"},
            ui.h2("Factores que influyen en la probabilidad de tenencia de vivienda propia"),
            ui.p(
                "Aplicación en Shiny para presentar resultados de la EFH 2021, "
                "comparar modelos y simular la probabilidad estimada de vivienda propia."
            ),
        ),
        ui.layout_columns(
            ui.value_box(
                "Hogares únicos analizados",
                f"{HOGARES_UNICOS_ANALISIS:,}",
                "Unidad principal utilizada para la comparación de modelos.",
                theme="primary"
            ),
            ui.value_box(
                "Vivienda propia",
                f"{summary_pct:.2f}%",
                "Proporción a nivel de hogares únicos",
                theme="blue"
            ),
            ui.value_box(
                "Variables del modelo",
                f"{len(ORDERED_VARS)}",
                "8 numéricas, 7 binarias y 3 categóricas.",
                theme=ui.value_box_theme(bg="#5b3c88", fg="white")
            ),
            ui.value_box(
                "Modelo final seleccionado",
                MODELO_FINAL_HOME,
                "Seleccionado por equilibrio entre desempeño, estabilidad e interpretabilidad.",
                theme=ui.value_box_theme(bg="#1f6f4a", fg="white")
            ),
            col_widths=(3, 3, 3, 3),
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Objetivo del dashboard"),
                ui.markdown(
                    """
                    - Mostrar los resultados descriptivos de las 18 variables seleccionadas.
                    - Comparar el desempeño de la regresión logística Ridge frente a MLP.
                    - Mantener una calculadora para simular perfiles de hogares.
                    """
                ),
                full_screen=True,
            ),
            ui.card(
                ui.card_header("Estructura del dashboard"),
                ui.markdown(
                    """
                    1. **Perfil descriptivo** para explorar cada variable.
                    2. **Correlaciones** para visualizar relaciones bivariadas.
                    3. **Modelos** para comparar métricas por fold.
                    4. **Calculadora** para estimar la probabilidad del perfil ingresado.
                    """
                ),
                full_screen=True,
            ),
            col_widths=(6, 6),
        ),
        ui.card(
            ui.card_header("¿Para qué sirve este dashboard?"),
            ui.p(
                "Este dashboard permite visualizar de forma integrada los resultados descriptivos, "
                "la comparación entre modelos predictivos y la simulación de perfiles de hogares a partir "
                "de resultados públicos agregados almacenados en la carpeta public_results y de un modelo "
                "entrenado guardado como joblib. Su propósito es apoyar la interpretación de los hallazgos "
                "sin exponer microdatos crudos de la EFH."
            ),
        ),
    ),
    ui.nav_panel(
        "Perfil descriptivo",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_selectize(
                    "var_desc",
                    "Variable a explorar",
                    choices={v: label_var(v) for v in DATA["available_vars"]},
                    selected=(DATA["available_vars"][0] if DATA["available_vars"] else ORDERED_VARS[0]),
                    width="100%",
                ),
                ui.p(
                    "Selecciona una variable y revisa su comportamiento univariado. "
                    "Para variables monetarias y continuas, la app reconstruye series sintéticas desde histogramas públicos.",
                    class_="helper-text",
                ),
                title="Exploración",
                width="300px",
                open="desktop",
            ),
            ui.output_ui("desc_note"),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Distribución principal"),
                    output_widget("desc_plot_main"),
                    full_screen=True,
                ),
                ui.card(
                    ui.card_header("Vista complementaria"),
                    output_widget("desc_plot_aux"),
                    full_screen=True,
                ),
                col_widths=(6, 6),
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Resumen numérico / frecuencias"),
                    ui.output_data_frame("desc_table"),
                    full_screen=True,
                ),
                ui.card(
                    ui.card_header("Lectura rápida"),
                    ui.output_ui("desc_text"),
                    full_screen=True,
                ),
                col_widths=(7, 5),
            ),
        ),
    ),
    ui.nav_panel(
        "Correlaciones",
        ui.div(
            {"class": "section-note"},
            "Se muestra la matriz de correlación entre las variables disponibles en los resultados públicos. ",
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Matriz de correlación"),
                output_widget("corr_plot"),
                full_screen=True,
            ),
            ui.card(
                ui.card_header("Correlaciones absolutas más altas"),
                ui.output_data_frame("corr_top_table"),
                full_screen=True,
            ),
            col_widths=(8, 4),
        ),
    ),
    ui.nav_panel(
        "Modelos",
        ui.div(
            {"class": "section-note"},
            "Comparación entre Regresión Logística Ridge y MLP usando las métricas por fold disponibles en metrics_by_fold.csv.",
        ),
        ui.layout_columns(
            ui.value_box("Mejor modelo según AUC", best_model_name, "Promedio out-of-fold.", theme=ui.value_box_theme(bg="#29895d", fg="white")),
            ui.value_box(
                "Mejor AUC promedio",
                (f"{best_auc_value:.3f}" if not np.isnan(best_auc_value) else "No disponible"),
                "Calculado sobre metrics_by_fold.csv.",
                theme="primary",
            ),
            ui.value_box("Comparación", "Regresión logística binaria vs MLP", "Ambos modelos se visualizan en la misma pestaña.", theme=ui.value_box_theme(bg="#6a499a", fg="white")),
            col_widths=(4, 4, 4),
        ),
        ui.layout_columns(
            ui.card(ui.card_header("AUC por fold"), output_widget("plot_auc"), full_screen=True),
            ui.card(ui.card_header("Sensibilidad por fold"), output_widget("plot_sens"), full_screen=True),
            col_widths=(6, 6),
        ),
        ui.layout_columns(
            ui.card(ui.card_header("Especificidad por fold"), output_widget("plot_spec"), full_screen=True),
            ui.card(ui.card_header("Precisión por fold"), output_widget("plot_prec"), full_screen=True),
            col_widths=(6, 6),
        ),
        ui.card(
            ui.card_header("Resumen estadístico por modelo"),
            ui.output_data_frame("model_summary_table"),
            full_screen=True,
        ),
    ),
    ui.nav_panel(
        "Calculadora",
        ui.layout_sidebar(
            ui.sidebar(
                ui.p("Completa o ajusta las características del hogar para estimar la probabilidad de tenencia de vivienda propia.",class_="helper-text",
                ),
                
                ui.input_slider("edad_pr", label_var("edad_pr"), min=16, max=90, value=16),
                ui.input_slider("numh", label_var("numh"), min=1, max=12, value=1),
                ui.input_slider("ocuph", label_var("ocuph"), min=0, max=3, value=0),
                ui.input_numeric("yoprinm_pr", label_var("yoprinm_pr"), value=0, min=0, step=50000),
                ui.input_numeric("ypenh", label_var("ypenh"), value=0, min=0, step=50000),
                ui.input_numeric("ysubh", label_var("ysubh"), value=0, min=0, step=50000),
                ui.input_numeric("act_fijo", label_var("act_fijo"), value=0, min=0, step=500000),
                ui.input_numeric("act_var", label_var("act_var"), value=0, min=0, step=500000),
                ui.input_numeric("cap_pen_ent", label_var("cap_pen_ent"), value=0, min=0, step=500000),
                ui.input_slider("hr_trabajadas_pr", label_var("hr_trabajadas_pr"), min=0, max=80, value=0),
                ui.input_select(
                    "est_civil_pr",
                    label_var("est_civil_pr"),
                    choices={"": "Selecciona..."} | {x: x for x in ESTADO_CIVIL},
                    selected="",
                ),
                ui.input_checkbox_group(
                    "instrumentos",
                    "Instrumentos financieros utilizados / poseídos",
                    choices={v: label_var(v) for v in BINARY_VARS},
                    selected=[],
                ),
                
                title="Perfil del hogar",
                width="350px",
                open="desktop",
            ),
            ui.row(
                ui.column(
                    4,
                    ui.div(
                        ui.card(
                            ui.card_header("Probabilidad estimada de tenencia de vivienda propia"),
                            ui.div(ui.output_text("prob_value"), class_="prob-main-value"),
                            ui.output_ui("prob_bar"),
                        ),
                        class_="calc-card-wrap prob-card-wrap",
                    ),
                ),
                ui.column(
                    8,
                    ui.div(
                        ui.card(
                            ui.card_header("Descripción del resultado"),
                            ui.output_ui("prob_interpretation"),
                        ),
                        class_="calc-card-wrap interp-card-wrap",
                    ),
                ),
            ),
        ),
    ),

    title="Dashboard EFH 2021",
    id="main_nav",
    selected="Inicio",
    theme=ui.Theme(preset="flatly"),
    header=ui.tags.style(APP_CSS),
    window_title="Dashboard EFH 2021 - Shiny",
)


def server(input: Inputs, output: Outputs, session: Session) -> None:
    @reactive.effect
    def _sync_ocuph_with_numh() -> None:
        max_ocuph = int(input.numh())
        current_ocuph = int(input.ocuph())
        ui.update_slider(
            "ocuph",
            min=0,
            max=max_ocuph,
            value=min(current_ocuph, max_ocuph),
            session=session,
        )

    @reactive.calc
    def current_var() -> str:
        selected = input.var_desc()
        return selected if selected else ORDERED_VARS[0]

    @reactive.calc
    def current_series() -> pd.Series:
        var = current_var()
        return pseudo_series_from_hist(var)

    @render.ui
    def desc_note():
        var = current_var()
        label = label_var(var)
        meta_type = "numérica" if var in NUMERIC_VARS else "categórica/binaria"
        return ui.div(
            {"class": "section-note"},
            ui.strong(f"Variable seleccionada: {label}. "),
            f"Tipo interpretado: {meta_type}.",
        )

    @render_widget
    def desc_plot_main():
        var = current_var()
        if var in NUMERIC_VARS and var not in {"numh", "ocuph"}:
            serie = current_series()
            if serie.empty:
                return empty_figure("No hay datos disponibles para esta variable.")
            fig = px.histogram(
                serie.to_frame(),
                x=var,
                nbins=30,
                template="plotly_white",
                title=f"Histograma de {label_var(var)}",
            )
            fig.update_layout(height=420, xaxis_title=label_var(var), yaxis_title="Frecuencia")
            return fig

        table = categorical_table(var)
        if table.empty:
            return empty_figure("No hay frecuencias disponibles para esta variable.")
        fig = px.bar(
            table.head(25),
            x="category",
            y="n",
            text="porcentaje",
            template="plotly_white",
            title=f"Frecuencias de {label_var(var)}",
        )
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(height=420, xaxis_title="Categoría", yaxis_title="Frecuencia")
        return fig

    @render_widget
    def desc_plot_aux():
        var = current_var()
        if var in NUMERIC_VARS and var not in {"numh", "ocuph"}:
            serie = current_series()
            if serie.empty:
                return empty_figure("No hay datos disponibles para esta variable.")
            fig = px.box(
                serie.to_frame(),
                y=var,
                template="plotly_white",
                title=f"Boxplot de {label_var(var)}",
            )
            fig.update_layout(height=420, yaxis_title=label_var(var))
            return fig

        table = categorical_table(var)
        if table.empty:
            return empty_figure("No hay categorías disponibles para esta variable.")
        fig = px.pie(
            table.head(10),
            names="category",
            values="n",
            hole=0.55,
            template="plotly_white",
            title=f"Composición porcentual de {label_var(var)}",
        )
        fig.update_layout(height=420)
        return fig

    @render.data_frame
    def desc_table():
        var = current_var()
        if var in NUMERIC_VARS and var not in {"numh", "ocuph"}:
            serie = current_series()
            if serie.empty:
                return render.DataGrid(pd.DataFrame({"mensaje": ["Sin datos disponibles"]}), width="100%")
            desc = serie.describe(percentiles=[0.25, 0.5, 0.75]).round(3).reset_index()
            desc.columns = ["estadístico", "valor"]
            return render.DataGrid(desc, width="100%")

        table = categorical_table(var)
        if table.empty:
            return render.DataGrid(pd.DataFrame({"mensaje": ["Sin datos disponibles"]}), width="100%")
        table = table.rename(columns={"category": "categoría", "n": "frecuencia", "porcentaje": "porcentaje"})
        return render.DataGrid(table, width="100%")

    @render.ui
    def desc_text():
        var = current_var()
        label = label_var(var)
        if var in NUMERIC_VARS and var not in {"numh", "ocuph"}:
            srow = get_numeric_summary(var)
            if srow is None:
                return ui.p("No hay resumen público disponible para esta variable.")
            porc_cero = float(srow.get("porc_cero", 0.0))
            porc_pos = float(srow.get("porc_pos", 0.0))
            n_total = int(float(srow.get("n", 0)))
            return ui.TagList(
                ui.p(f"La variable {label.lower()} se resume sobre {n_total:,} registros públicos."),
                ui.tags.ul(
                    ui.tags.li(f"{porc_cero:.2f}% de los hogares no presentan monto en esta variable."),
                    ui.tags.li(f"{porc_pos:.2f}% sí presentan monto positivo en esta variable."),
                    ui.tags.li("La forma exacta observada en el gráfico proviene de una reconstrucción basada en histogramas agregados."),
                ),
            )

        table = categorical_table(var)
        if table.empty:
            return ui.p("No hay frecuencias públicas disponibles para esta variable.")
        top_row = table.iloc[0]
        return ui.TagList(
            ui.p(f"La categoría más frecuente en {label.lower()} es {top_row['category']}.") ,
            ui.tags.ul(
                ui.tags.li(f"Frecuencia: {int(top_row['n']):,} hogares."),
                ui.tags.li(f"Participación: {float(top_row['porcentaje']):.2f}% del total de la variable."),
            ),
        )

    @render_widget
    def corr_plot():
        corr_long = DATA["corr_long"]
        if corr_long.empty:
            return empty_figure("No se encontró la matriz de correlación en public_results.")
        corr_mat = corr_long.pivot(index="var1", columns="var2", values="corr")
        corr_mat = corr_mat.loc[[v for v in ORDERED_VARS if v in corr_mat.index], [v for v in ORDERED_VARS if v in corr_mat.columns]]
        fig = px.imshow(
            corr_mat,
            text_auto=True,
            color_continuous_scale="Blues",
            template="plotly_white",
            aspect="auto",
        )
        fig.update_layout(height=720)
        return fig

    @render.data_frame
    def corr_top_table():
        corr_long = DATA["corr_long"]
        if corr_long.empty:
            return render.DataGrid(pd.DataFrame({"mensaje": ["Sin matriz de correlación disponible"]}), width="100%")
        corr = corr_long.copy()
        corr = corr[corr["var1"] != corr["var2"]].copy()
        corr["pair"] = corr.apply(lambda r: " | ".join(sorted([str(r["var1"]), str(r["var2"])])), axis=1)
        corr["abs_corr"] = corr["corr"].abs()
        corr = corr.sort_values("abs_corr", ascending=False).drop_duplicates("pair")
        corr = corr.head(12)[["var1", "var2", "corr", "abs_corr"]].copy()
        corr["var1"] = corr["var1"].map(label_var)
        corr["var2"] = corr["var2"].map(label_var)
        corr.columns = ["Variable 1", "Variable 2", "Correlación", "|Correlación|"]
        return render.DataGrid(corr.round(3), width="100%")

    def plot_metric(metric_name: str, title: str):
        df_metrics = DATA["metrics_fold"]
        if df_metrics.empty or metric_name not in df_metrics.columns or "Modelo" not in df_metrics.columns:
            return empty_figure(f"No hay datos de {metric_name}.")
        dfp = df_metrics[df_metrics["Modelo"].astype(str).isin(["Regresión logística binaria", "MLP"])].copy()
        if dfp.empty or "Fold" not in dfp.columns:
            return empty_figure(f"No hay comparaciones válidas para {metric_name}.")
        dfp["Fold"] = pd.to_numeric(dfp["Fold"], errors="coerce")
        dfp = dfp.dropna(subset=["Fold", metric_name])
        if dfp.empty:
            return empty_figure(f"No hay comparaciones válidas para {metric_name}.")
        fig = px.line(
            dfp,
            x="Fold",
            y=metric_name,
            color="Modelo",
            markers=True,
            template="plotly_white",
            title=title,
        )
        fig.update_layout(height=380)
        return fig

    @render_widget
    def plot_auc():
        return plot_metric("AUC", "Distribución por fold - AUC")

    @render_widget
    def plot_sens():
        return plot_metric("Sensibilidad", "Distribución por fold - Sensibilidad")

    @render_widget
    def plot_spec():
        return plot_metric("Especificidad", "Distribución por fold - Especificidad")

    @render_widget
    def plot_prec():
        return plot_metric("Precision", "Distribución por fold - Precisión")

    @render.data_frame
    def model_summary_table():
        if metrics_summary.empty:
            return render.DataGrid(pd.DataFrame({"mensaje": ["No hay métricas por fold disponibles"]}), width="100%")
        return render.DataGrid(metrics_summary.round(4), width="100%")

    @reactive.calc
    def calculator_values() -> dict[str, Any]:
        instruments = set(input.instrumentos() or [])
        values = {
            "edad_pr": input.edad_pr(),
            "numh": input.numh(),
            "ocuph": input.ocuph(),
            "yoprinm_pr": input.yoprinm_pr() or 0,
            "ypenh": input.ypenh() or 0,
            "ysubh": input.ysubh() or 0,
            "act_fijo": input.act_fijo() or 0,
            "act_var": input.act_var() or 0,
            "cap_pen_ent": input.cap_pen_ent() or 0,
            "hr_trabajadas_pr": input.hr_trabajadas_pr(),
            "est_civil_pr": input.est_civil_pr(),
            "t_cc": "t_cc" in instruments,
            "t_tbco": "t_tbco" in instruments,
            "u_cheq": "u_cheq" in instruments,
            "u_pac": "u_pac" in instruments,
            "u_pat": "u_pat" in instruments,
            "u_tbco": "u_tbco" in instruments,
            "u_prepago": "u_prepago" in instruments,
        }
    
        values["ocuph"] = min(int(values["ocuph"]), int(values["numh"]))
    
        if int(values["ocuph"]) == 0:
            values["yoprinm_pr"] = 0
            values["hr_trabajadas_pr"] = 0
    
        if int(values["hr_trabajadas_pr"]) == 0 and float(values["yoprinm_pr"]) > 0:
            values["yoprinm_pr"] = 0
    
        return values

    @reactive.calc
    def calc_warnings_list() -> list[str]:
        raw_income = float(input.yoprinm_pr() or 0)
        raw_hours = int(input.hr_trabajadas_pr())
        ocuph = int(input.ocuph())
        warnings: list[str] = []
        if ocuph == 0 and (raw_income > 0 or raw_hours > 0):
            warnings.append(
                "Ocupados = 0 pero existe ingreso principal o horas trabajadas. El modo estricto corrige esta incoherencia."
            )
        if raw_hours == 0 and raw_income > 0:
            warnings.append(
                "Horas trabajadas = 0 junto con ingreso principal positivo. Esto puede representar un caso atípico."
            )
        if int(input.numh()) < ocuph:
            warnings.append("El número de ocupados no puede superar al número de miembros del hogar.")
        return warnings

    @reactive.calc
    def prediction_result() -> dict[str, Any]:
        values = calculator_values()
    
        if not values.get("est_civil_pr"):
            return {
                "ok": False,
                "message": "Completa el perfil del hogar para calcular la probabilidad."
            }
    
        if not DATA["have_model"]:
            return {"ok": False, "message": DATA["model_error"] or "Modelo no disponible."}
    
        x_in = prepare_input_row(values)
    
        try:
            p = float(DATA["model"].predict_proba(x_in)[0, 1])
            return {
                "ok": True,
                "prob": p,
                "x_in": x_in,
            }
        except Exception as e:  # pragma: no cover - defensive fallback
            return {"ok": False, "message": str(e)}

    @render.text
    def prob_value():
        res = prediction_result()
        return f"{100 * res['prob']:.2f}%" if res.get("ok") else "--"

    @render.text
    def prob_caption():
        res = prediction_result()
        return f"p = {res['prob']:.6f}" if res.get("ok") else "No se pudo calcular la probabilidad."
        
    @render.ui
    def prob_bar():
        res = prediction_result()
        if not res.get("ok"):
            return ui.div(
                ui.div(class_="prob-bar-fill", style="width: 0%;"),
                class_="prob-bar-wrap",
            )
    
        pct = max(0, min(100, 100 * float(res["prob"])))
        return ui.div(
            ui.div(class_="prob-bar-fill", style=f"width: {pct:.2f}%;"),
            class_="prob-bar-wrap",
        )
    
    @render.ui
    def prob_interpretation():
        res = prediction_result()
        if not res.get("ok"):
            return ui.p(res.get("message", "Completa el perfil del hogar para visualizar el resultado."))
    
        return ui.p(
            "La cifra mostrada corresponde a la probabilidad estimada por el modelo para el perfil de hogar ingresado."
        )

    @render.ui
    def calc_warnings():
        warnings = calc_warnings_list()
        if not warnings:
            return ui.p("No se detectaron incoherencias relevantes en el perfil ingresado.")
        return ui.tags.ul(*[ui.tags.li(w) for w in warnings])

    @render.data_frame
    def input_row_table():
        res = prediction_result()
        if not res.get("ok"):
            return render.DataGrid(pd.DataFrame({"mensaje": [res.get("message", "No disponible")]}), width="100%")
        df = res["x_in"].T.reset_index()
        df.columns = ["variable", "valor_ingresado_modelo"]
        df["variable"] = df["variable"].map(label_var)
        return render.DataGrid(df, width="100%")


app = App(app_ui, server)
