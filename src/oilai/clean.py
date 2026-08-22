"""Construcción del panel mensual de producción por campo.

Decisiones de limpieza (documentadas para la tesis):

1. **Agregación municipal.** Un campo que se extiende sobre varios municipios
   se reporta en una fila por municipio. La unidad de análisis es el campo, así
   que sumamos la producción y conservamos el municipio de mayor aporte.
2. **Operadora.** Puede cambiar a lo largo del tiempo (cesión de contratos). Se
   conserva la del mes, no una única por campo, porque el cambio de operadora es
   una covariable potencialmente informativa.
3. **Normalización por días del mes.** La producción mensual en barriles no es
   comparable entre meses de 28 y 31 días. Derivamos `bpd` (barriles por día
   calendario), que es la variable que modelamos.
4. **No se rellenan huecos.** Un mes ausente puede significar campo cerrado o
   simplemente no reportado; imputar introduciría señal falsa. El panel marca
   explícitamente los huecos con `es_hueco`.
"""

from __future__ import annotations

import calendar

import pandas as pd

from .config import PANEL_PARQUET
from .ingest import download

NUMERIC_COLS = ["produccion_bls", "latitud", "longitud"]


def _to_float(series: pd.Series) -> pd.Series:
    """'508,922.00' -> 508922.0"""
    return pd.to_numeric(
        series.astype("string").str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def _collapse_municipios(df: pd.DataFrame) -> pd.DataFrame:
    """Suma la producción de un mismo campo repartida en varios municipios."""
    df = df.sort_values("produccion_bls", ascending=False)
    grouped = df.groupby(["campo", "fecha"], as_index=False).agg(
        produccion_bls=("produccion_bls", "sum"),
        operadora=("operadora", "first"),      # first = municipio de mayor aporte
        contrato=("contrato", "first"),
        tipo_contrato=("tipo_contrato", "first"),
        departamento=("departamento", "first"),
        municipio=("municipio", "first"),
        latitud=("latitud", "first"),
        longitud=("longitud", "first"),
        n_municipios=("municipio", "nunique"),
    )
    return grouped


def _add_panel_features(df: pd.DataFrame) -> pd.DataFrame:
    """Variables derivadas por campo: días del mes, bpd y edad productiva."""
    dias = df["fecha"].dt.days_in_month
    df["dias_mes"] = dias
    df["bpd"] = df["produccion_bls"] / dias

    df = df.sort_values(["campo", "fecha"])
    primera = df.groupby("campo")["fecha"].transform("min")
    df["meses_desde_inicio"] = (
        (df["fecha"].dt.year - primera.dt.year) * 12
        + (df["fecha"].dt.month - primera.dt.month)
    )

    # Un hueco es un mes sin reporte entre dos meses reportados del mismo campo.
    esperado = df.groupby("campo")["meses_desde_inicio"].transform(
        lambda s: s.diff().fillna(1)
    )
    df["meses_desde_reporte_previo"] = esperado.astype(int)
    df["es_hueco"] = df["meses_desde_reporte_previo"] > 1

    return df


def build_panel(force: bool = False) -> pd.DataFrame:
    """Panel mensual campo x fecha, listo para modelar."""
    if PANEL_PARQUET.exists() and not force:
        return pd.read_parquet(PANEL_PARQUET)

    raw = download()

    df = raw.copy()
    for col in NUMERIC_COLS:
        df[col] = _to_float(df[col])

    df["vigencia"] = pd.to_numeric(df["vigencia"], errors="coerce").astype("Int64")
    df["mes"] = pd.to_numeric(df["mes"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["vigencia", "mes", "campo", "produccion_bls"])

    df["fecha"] = pd.to_datetime(
        dict(year=df["vigencia"], month=df["mes"], day=1)
    )

    for col in ("campo", "operadora", "contrato", "departamento", "municipio"):
        df[col] = df[col].astype("string").str.strip().str.upper()

    panel = _collapse_municipios(df)
    panel = _add_panel_features(panel)

    panel.to_parquet(PANEL_PARQUET, index=False)
    return panel


if __name__ == "__main__":
    p = build_panel(force=True)
    print(f"panel: {len(p):,} filas x {len(p.columns)} columnas")
    print(f"campos: {p.campo.nunique()}  |  {p.fecha.min():%Y-%m} a {p.fecha.max():%Y-%m}")
    print(f"guardado en {PANEL_PARQUET}")
