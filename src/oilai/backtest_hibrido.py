"""Evaluación de los modelos híbridos (Fase 4).

Se reutiliza el protocolo de calendario de la Fase 3 —mismos cortes, mismos
campos, mismos horizontes— para que las cifras sean directamente comparables con
el modelo global.

Dos detalles de eficiencia y corrección que conviene explicar:

* **Las muestras se construyen una sola vez** para todo el panel y luego se
  filtran por fecha. Es equivalente a reconstruirlas por corte, porque todas las
  variables son causales: truncar el futuro no altera una variable calculada en
  un origen anterior. La prueba `test_las_variables_no_miran_al_futuro` sostiene
  esa equivalencia, y `test_filtrar_equivale_a_reconstruir` la comprueba directamente.

* **Los pesos de la vía B se estiman en un origen de validación anterior al
  corte** (doce meses antes). En ese origen los valores reales ya se conocen en
  la fecha de corte, así que no hay fuga; usar el propio corte para estimar los
  pesos y luego medir sobre él sería hacer trampa.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .backtest_global import (
    CORTES,
    HORIZONTE,
    MIN_HISTORIA,
    evaluar_referencias,
)
from .clean import build_panel
from .config import REPORTS
from .features import construir_muestras
from .models.global_ml import ModeloGlobal
from .models.hibrido import (
    VARIABLES_ARPS,
    CombinacionPorRegimen,
    agregar_variable_arps,
    ajustes_arps,
)

# Modelos base que entran en la combinación convexa: los tres que ganaron en
# algún régimen a lo largo de las fases anteriores.
BASES_COMBINACION = ["Naive", "Arps-24m", "ML-global"]

MESES_VALIDACION_PESOS = 12

CORTES_TAMANO = [0, 500, 5000, 50000, np.inf]
ETIQUETAS_TAMANO = ["<0.5k", "0.5-5k", "5-50k", ">50k"]


class HibridoArpsML(ModeloGlobal):
    """Vía A: el modelo global con el pronóstico de Arps como variable."""

    nombre = "Híbrido-Arps-ML"


def clase_tamano(panel: pd.DataFrame, origen: pd.Timestamp) -> pd.Series:
    """Clase de tamaño de cada campo según los doce meses previos al origen.

    Debe calcularse con datos anteriores al origen: usar la media de toda la
    serie metería información posterior en una variable de estratificación.
    """
    ventana = panel[
        (panel.fecha <= origen)
        & (panel.fecha > origen - pd.DateOffset(months=12))
    ]
    media = ventana.groupby("campo").bpd.mean()
    return pd.cut(media, CORTES_TAMANO, labels=ETIQUETAS_TAMANO)


def _predecir_ml(
    muestras: pd.DataFrame,
    origen: pd.Timestamp,
    clase: type[ModeloGlobal] = ModeloGlobal,
) -> tuple[pd.DataFrame, ModeloGlobal]:
    """Entrena con lo ya ocurrido en `origen` y predice en ese origen."""
    entrenamiento = muestras[muestras.fecha_objetivo <= origen]
    prueba = muestras[muestras.origen == origen]

    modelo = clase().fit(entrenamiento)
    salida = pd.DataFrame(
        {
            "campo": prueba.campo.to_numpy(),
            "h": prueba.h.to_numpy(int),
            "y": prueba.bpd_real.to_numpy(float),
            "yhat": modelo.predict_bpd(prueba),
        }
    )
    return salida, modelo


def _ancho(
    referencias: pd.DataFrame, ml: pd.DataFrame, clases: pd.Series
) -> pd.DataFrame:
    """Una fila por (campo, h) con una columna de predicción por modelo."""
    tabla = referencias.pivot_table(
        index=["campo", "h"], columns="modelo", values="yhat"
    ).reset_index()

    reales = referencias.groupby(["campo", "h"]).agg(
        y=("y", "first"), escala=("escala", "first")
    ).reset_index()

    tabla = tabla.merge(reales, on=["campo", "h"])
    tabla = tabla.merge(
        ml.rename(columns={"yhat": "ML-global"}).drop(columns="y"),
        on=["campo", "h"],
        how="inner",
    )
    tabla["clase"] = tabla.campo.map(clases)
    return tabla.dropna(subset=BASES_COMBINACION + ["y"])


def main(verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = build_panel()

    if verbose:
        print("construyendo muestras y ajustes de Arps...", flush=True)
    muestras = construir_muestras(panel, horizontes=range(1, HORIZONTE + 1),
                                  min_historia=MIN_HISTORIA)
    ajustes = ajustes_arps(panel)
    muestras_arps = agregar_variable_arps(muestras, ajustes)

    if verbose:
        cobertura = muestras_arps.arps_rel.notna().mean() * 100
        print(f"  {len(muestras):,} muestras · Arps disponible en {cobertura:.1f}%",
              flush=True)

    resultados = []
    pesos_por_corte = []

    for corte in CORTES:
        validacion = corte - pd.DateOffset(months=MESES_VALIDACION_PESOS)
        if verbose:
            print(f"\ncorte {corte:%Y-%m} (pesos aprendidos en {validacion:%Y-%m})",
                  flush=True)
        t0 = time.perf_counter()

        # --- Referencias y modelo global en el corte ---
        ref_corte = evaluar_referencias(panel, corte)
        ml_corte, _ = _predecir_ml(muestras, corte)
        hib_corte, modelo_hib = _predecir_ml(muestras_arps, corte, HibridoArpsML)

        # --- Mismos modelos en el origen de validación, para estimar pesos ---
        ref_val = evaluar_referencias(panel, validacion)
        ml_val, _ = _predecir_ml(muestras, validacion)

        ancho_val = _ancho(ref_val, ml_val, clase_tamano(panel, validacion))
        combinador = CombinacionPorRegimen(BASES_COMBINACION).fit(ancho_val)

        ancho_corte = _ancho(ref_corte, ml_corte, clase_tamano(panel, corte))
        ancho_corte["Híbrido-regimen"] = combinador.predict(ancho_corte)

        tabla = combinador.tabla_pesos()
        tabla["corte"] = corte
        pesos_por_corte.append(tabla)

        # --- Se recoge todo en formato largo ---
        base = ancho_corte[["campo", "h", "y", "escala", "clase"]].copy()
        base["origen"] = corte

        for modelo in BASES_COMBINACION + ["Híbrido-regimen"]:
            trozo = base.copy()
            trozo["modelo"] = modelo
            trozo["yhat"] = ancho_corte[modelo].to_numpy()
            resultados.append(trozo)

        # La vía A se une por (campo, h) para evaluarse sobre el mismo conjunto.
        trozo = base.merge(
            hib_corte[["campo", "h", "yhat"]], on=["campo", "h"], how="inner"
        )
        trozo["modelo"] = HibridoArpsML.nombre
        resultados.append(trozo)

        if verbose:
            arboles = modelo_hib.mejor_iteracion or "sin parada temprana"
            print(f"  {len(ancho_corte):,} pares (campo, h) · árboles vía A: {arboles}"
                  f" · {time.perf_counter() - t0:.1f}s", flush=True)

    df = pd.concat(resultados, ignore_index=True)

    # Solo se comparan pares que todos los modelos pudieron pronosticar.
    llave = ["campo", "origen", "h"]
    n_modelos = df.modelo.nunique()
    completos = df.groupby(llave).modelo.nunique() == n_modelos
    df = df.merge(completos[completos].reset_index()[llave], on=llave, how="inner")

    pesos = pd.concat(pesos_por_corte, ignore_index=True)

    df.to_parquet(REPORTS / "backtest_hibrido.parquet", index=False)
    pesos.to_csv(REPORTS / "pesos_hibrido.csv", index=False)
    return df, pesos
