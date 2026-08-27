"""Benchmark con orígenes alineados en calendario.

La Fase 1 evaluó cada campo en orígenes definidos por su **posición** en la serie
(a un cuarto, a la mitad, al final de su historia). Eso vale para comparar
modelos ajustados campo a campo, pero no sirve aquí: un modelo global se entrena
con todos los campos simultáneamente y necesita un corte temporal único, o
estaría aprendiendo del futuro de un campo para predecir el pasado de otro.

Este módulo redefine el protocolo con **cortes de calendario comunes**. Todos los
modelos —líneas base, Arps y el modelo global— se evalúan sobre exactamente los
mismos campos, orígenes y horizontes, de modo que la comparación es directa.

Regla de entrenamiento del modelo global: solo entran muestras cuyo **objetivo ya
había ocurrido** en la fecha de corte (`fecha_objetivo <= corte`). No basta con
filtrar por el origen: una muestra con origen anterior al corte pero objetivo
posterior contiene precisamente el dato que se quiere predecir.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .clean import build_panel
from .config import REPORTS
from .evaluate import escala_naive
from .features import construir_muestras
from .models.baselines import ArpsModel, Drift, MediaMovil, Naive
from .models.global_ml import ModeloGlobal

CORTES = [
    pd.Timestamp("2023-03-01"),
    pd.Timestamp("2024-03-01"),
    pd.Timestamp("2025-03-01"),
]

HORIZONTE = 12
MIN_HISTORIA = 24


def modelos_referencia() -> list:
    """Las líneas base de la Fase 1, para comparar bajo el nuevo protocolo."""
    return [Naive(), MediaMovil(3), Drift(), ArpsModel(ventana=24), ArpsModel(ventana=36)]


def _historia(g: pd.DataFrame, corte: pd.Timestamp) -> pd.DataFrame:
    return g[g.fecha <= corte].sort_values("fecha")


def _meses_desde(inicio: pd.Timestamp, fechas: pd.Series) -> np.ndarray:
    return (
        (fechas.dt.year - inicio.year) * 12 + (fechas.dt.month - inicio.month)
    ).to_numpy(float)


def evaluar_referencias(
    panel: pd.DataFrame, corte: pd.Timestamp
) -> pd.DataFrame:
    """Predicciones de las líneas base para los 12 meses tras el corte."""
    fin = corte + pd.DateOffset(months=HORIZONTE)
    filas = []

    for campo, g in panel.groupby("campo", sort=False):
        hist = _historia(g, corte)
        if len(hist) < MIN_HISTORIA:
            continue

        futuro = g[(g.fecha > corte) & (g.fecha <= fin)].sort_values("fecha")
        if futuro.empty:
            continue

        inicio = hist.fecha.iloc[0]
        t_hist = _meses_desde(inicio, hist.fecha)
        q_hist = hist.bpd.to_numpy(float)
        t_fut = _meses_desde(inicio, futuro.fecha)
        q_fut = futuro.bpd.to_numpy(float)

        # h en meses de calendario desde el corte, comparable entre campos.
        h = (
            (futuro.fecha.dt.year - corte.year) * 12
            + (futuro.fecha.dt.month - corte.month)
        ).to_numpy(int)

        esc = escala_naive(q_hist)

        for modelo in modelos_referencia():
            try:
                pred = modelo.fit(t_hist, q_hist).predict(t_fut)
            except Exception:
                continue
            for j in range(len(q_fut)):
                filas.append(
                    {
                        "campo": campo,
                        "modelo": modelo.nombre,
                        "origen": corte,
                        "h": int(h[j]),
                        "y": float(q_fut[j]),
                        "yhat": float(pred[j]),
                        "escala": esc,
                    }
                )

    return pd.DataFrame(filas)


def evaluar_global(
    panel: pd.DataFrame, corte: pd.Timestamp, verbose: bool = True
) -> tuple[pd.DataFrame, ModeloGlobal]:
    """Entrena el modelo global con datos anteriores al corte y lo evalúa."""
    muestras = construir_muestras(
        panel[panel.fecha <= corte + pd.DateOffset(months=HORIZONTE)],
        horizontes=range(1, HORIZONTE + 1),
        min_historia=MIN_HISTORIA,
    )

    # Entrenamiento: solo lo ya ocurrido en la fecha de corte.
    entrenamiento = muestras[muestras.fecha_objetivo <= corte]
    prueba = muestras[muestras.origen == corte]

    if verbose:
        print(
            f"  entrenamiento: {len(entrenamiento):,} muestras "
            f"({entrenamiento.campo.nunique()} campos) | "
            f"prueba: {len(prueba):,} ({prueba.campo.nunique()} campos)",
            flush=True,
        )

    modelo = ModeloGlobal().fit(entrenamiento)
    pred = modelo.predict_bpd(prueba)

    # La escala de MASE se calcula con la historia previa al corte, igual que
    # para las líneas base, para que las cifras sean comparables.
    escalas = {}
    for campo, g in panel.groupby("campo", sort=False):
        hist = _historia(g, corte)
        if len(hist) >= MIN_HISTORIA:
            escalas[campo] = escala_naive(hist.bpd.to_numpy(float))

    out = pd.DataFrame(
        {
            "campo": prueba.campo.to_numpy(),
            "modelo": modelo.nombre,
            "origen": corte,
            "h": prueba.h.to_numpy(int),
            "y": prueba.bpd_real.to_numpy(float),
            "yhat": pred,
            "escala": prueba.campo.map(escalas).to_numpy(float),
        }
    )
    return out, modelo


def main(verbose: bool = True) -> pd.DataFrame:
    panel = build_panel()
    trozos = []
    importancias = None

    for corte in CORTES:
        if verbose:
            print(f"\ncorte {corte:%Y-%m}", flush=True)
        t0 = time.perf_counter()

        trozos.append(evaluar_referencias(panel, corte))
        global_df, modelo = evaluar_global(panel, corte, verbose)
        trozos.append(global_df)
        importancias = modelo.importancias()

        if verbose:
            arboles = modelo.mejor_iteracion or "sin parada temprana"
            print(f"  árboles: {arboles} | {time.perf_counter() - t0:.1f}s", flush=True)

    df = pd.concat(trozos, ignore_index=True)

    # Solo se comparan campos que todos los modelos pudieron pronosticar.
    llave = ["campo", "origen", "h"]
    n_modelos = df.modelo.nunique()
    completos = df.groupby(llave).modelo.nunique() == n_modelos
    df = df.merge(
        completos[completos].reset_index()[llave], on=llave, how="inner"
    )

    df.to_parquet(REPORTS / "backtest_global.parquet", index=False)
    if importancias is not None:
        importancias.to_csv(REPORTS / "importancia_variables.csv", header=["ganancia_pct"])
    return df
