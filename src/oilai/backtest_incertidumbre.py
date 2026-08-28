"""Evaluación de intervalos de predicción y de la detección de anomalías.

Diseño de las particiones, que es donde se juega la validez:

    |........ entrenamiento ........|.. calibración ..|.... prueba ....|
                              C-12                  C                C+12

* **Entrenamiento**: muestras cuyo objetivo ocurrió antes de `C-12`.
* **Calibración**: orígenes en `[C-12, C)`, con objetivo ya ocurrido en `C`. El
  modelo no vio estas muestras, que es lo que exige la predicción conformal por
  particiones: calibrar sobre datos de entrenamiento daría márgenes optimistas y
  por tanto intervalos demasiado estrechos.
* **Prueba**: orígenes en `[C, C+12)`. Se usan doce meses de orígenes en lugar de
  uno solo porque estimar una cobertura con precisión exige muchas
  observaciones, y porque así los tres cortes cubren de 2023 a 2026 de forma
  continua y sin solapamiento.

El coste de esta separación es que el modelo de puntos se entrena con doce meses
menos de datos que el de la Fase 3, así que sus pronósticos son algo peores. Es
el precio de unos intervalos honestos y conviene declararlo.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .anomalias import detectar, evolucion_posterior, tasa_por_mes, validar
from .backtest_global import CORTES, HORIZONTE, MIN_HISTORIA
from .clean import build_panel
from .config import REPORTS
from .features import construir_muestras
from .incertidumbre import (
    NIVEL,
    ModeloCuantil,
    _alfas,
    aplicar_conformal,
    aplicar_conformal_grupo,
    calibrar_conformal,
    calibrar_conformal_grupo,
    resumen_intervalos,
)
from .models.global_ml import ModeloGlobal

MESES_CALIBRACION = 12
MESES_PRUEBA = 12

# Árboles fijos para todos los modelos de esta fase: iguala el presupuesto entre
# el modelo de puntos y los cuantílicos, de modo que la comparación entre métodos
# no dependa de cuánto se entrenó cada uno.
N_ARBOLES = 400

# Método con el que se construyen los intervalos que disparan alertas.
METODO_ALERTAS = "Conformal-clase"


_PANEL: pd.DataFrame | None = None


def _panel_cache() -> pd.DataFrame:
    global _PANEL
    if _PANEL is None:
        _PANEL = build_panel()
    return _PANEL


def _particiones(
    muestras: pd.DataFrame, corte: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inicio_cal = corte - pd.DateOffset(months=MESES_CALIBRACION)
    fin_prueba = corte + pd.DateOffset(months=MESES_PRUEBA)

    entrena = muestras[muestras.fecha_objetivo <= inicio_cal]
    calibra = muestras[
        (muestras.origen >= inicio_cal)
        & (muestras.origen < corte)
        & (muestras.fecha_objetivo <= corte)
    ]
    prueba = muestras[(muestras.origen >= corte) & (muestras.origen < fin_prueba)]

    return entrena, calibra, prueba


def evaluar_corte(
    muestras: pd.DataFrame, corte: pd.Timestamp, verbose: bool = True
) -> pd.DataFrame:
    """Intervalos por los tres métodos, sobre las mismas observaciones."""
    entrena, calibra, prueba = _particiones(muestras, corte)

    if verbose:
        print(
            f"  entrena {len(entrena):,} · calibra {len(calibra):,} · "
            f"prueba {len(prueba):,}",
            flush=True,
        )

    if calibra.empty or prueba.empty:
        return pd.DataFrame()

    # --- Modelo de puntos y márgenes conformales ---
    punto = ModeloGlobal(n_estimators=N_ARBOLES).fit(entrena)

    # La clase se calcula con los doce meses ANTERIORES a la calibración, de
    # modo que sea causal tanto para calibrar como para predecir.
    from .backtest_hibrido import clase_tamano

    clases = clase_tamano(_panel_cache(), corte - pd.DateOffset(months=MESES_CALIBRACION))

    residuos = pd.DataFrame(
        {
            "h": calibra.h.to_numpy(int),
            "grupo": calibra.campo.map(clases).astype(object).to_numpy(),
            "residuo": calibra.y.to_numpy(float) - punto.predict_log(calibra),
        }
    )
    margenes = calibrar_conformal(residuos, NIVEL)
    margenes_grupo, margenes_h = calibrar_conformal_grupo(residuos, NIVEL)

    pred_log = punto.predict_log(prueba)
    ancla = prueba.ancla_bpd.to_numpy(float)
    lo_c, hi_c = aplicar_conformal(pred_log, ancla, prueba.h.to_numpy(int), margenes)

    grupo_prueba = prueba.campo.map(clases).astype(object).to_numpy()
    lo_g, hi_g = aplicar_conformal_grupo(
        pred_log, ancla, prueba.h.to_numpy(int), grupo_prueba,
        margenes_grupo, margenes_h,
    )

    # --- Regresión cuantílica, con el mismo conjunto de entrenamiento ---
    baja, alta = _alfas(NIVEL)
    q_baja = ModeloCuantil(baja, n_estimators=N_ARBOLES).fit(entrena)
    q_alta = ModeloCuantil(alta, n_estimators=N_ARBOLES).fit(entrena)

    lo_q = np.clip(ancla * np.exp(q_baja.predict_log(prueba)), 0.0, None)
    hi_q = np.clip(ancla * np.exp(q_alta.predict_log(prueba)), 0.0, None)

    # El cuantil bajo puede cruzar al alto: son modelos independientes y nada se
    # lo impide. Se ordenan, que es la corrección estándar y no altera cobertura.
    lo_q, hi_q = np.minimum(lo_q, hi_q), np.maximum(lo_q, hi_q)

    base = pd.DataFrame(
        {
            "campo": prueba.campo.to_numpy(),
            "origen": prueba.origen.to_numpy(),
            "fecha_objetivo": prueba.fecha_objetivo.to_numpy(),
            "h": prueba.h.to_numpy(int),
            "y": prueba.bpd_real.to_numpy(float),
            "punto": np.clip(ancla * np.exp(pred_log), 0.0, None),
            "corte": corte,
        }
    )

    salida = []
    base["clase"] = grupo_prueba

    for metodo, lo, hi in (
        ("Conformal", lo_c, hi_c),
        ("Conformal-clase", lo_g, hi_g),
        ("Cuantílica", lo_q, hi_q),
    ):
        trozo = base.copy()
        trozo["metodo"] = metodo
        trozo["lo"] = lo
        trozo["hi"] = hi
        salida.append(trozo)

    return pd.concat(salida, ignore_index=True)


def main(verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = build_panel()

    if verbose:
        print("construyendo muestras...", flush=True)
    muestras = construir_muestras(
        panel, horizontes=range(1, HORIZONTE + 1), min_historia=MIN_HISTORIA
    )

    trozos = []
    for corte in CORTES:
        if verbose:
            print(f"\ncorte {corte:%Y-%m}", flush=True)
        t0 = time.perf_counter()
        trozos.append(evaluar_corte(muestras, corte, verbose))
        if verbose:
            print(f"  {time.perf_counter() - t0:.1f}s", flush=True)

    intervalos = pd.concat(trozos, ignore_index=True)
    intervalos.to_parquet(REPORTS / "intervalos.parquet", index=False)

    # --- Anomalías: se monitorea el mes siguiente ---
    # Se usa la conformal condicionada por clase: con la marginal, los campos
    # medianos quedaban sobrecubiertos y casi nunca disparaban alerta, mientras
    # los pequeños disparaban de más. La tasa de alerta debe ser comparable entre
    # tamaños para que la lista de avisos sea utilizable.
    un_mes = intervalos[
        (intervalos.metodo == METODO_ALERTAS) & (intervalos.h == 1)
    ].copy()
    alertas = detectar(un_mes)
    alertas.to_parquet(REPORTS / "alertas.parquet", index=False)

    evolucion = evolucion_posterior(panel, alertas)
    evolucion.to_parquet(REPORTS / "evolucion_alertas.parquet", index=False)

    if verbose:
        print(
            f"\n{len(alertas):,} observaciones monitoreadas · "
            f"{int(alertas.anomalia_baja.sum())} alertas de caída",
            flush=True,
        )

    return intervalos, alertas, evolucion
