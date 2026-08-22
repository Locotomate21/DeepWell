"""Backtesting walk-forward y métricas de error.

Protocolo de evaluación (el punto metodológico central de la tesis):

* **Origen rodante.** Para cada campo se eligen varios orígenes de pronóstico.
  En cada uno el modelo se entrena SOLO con datos anteriores al origen y
  pronostica h = 1..H meses. Nunca hay información del futuro en el
  entrenamiento.
* **Horizontes separados.** El error se reporta por horizonte, porque un modelo
  puede ser bueno a 1 mes y malo a 12; promediarlos oculta eso.
* **MASE.** Escala el error por el del Naive estacional sobre el histórico de
  entrenamiento, lo que permite comparar campos de 500 bpd con campos de
  120.000 bpd. Es la métrica de decisión; MAE y RMSE se reportan en bpd para
  interpretabilidad operativa.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def smape(y: np.ndarray, yhat: np.ndarray) -> float:
    """sMAPE simétrico en %. Evita la explosión de MAPE cuando y→0."""
    denom = (np.abs(y) + np.abs(yhat)) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denom > 0, np.abs(y - yhat) / denom, 0.0)
    return float(np.mean(ratio) * 100)


def sesgo(y: np.ndarray, yhat: np.ndarray) -> float:
    """Error medio con signo: positivo = el modelo sobreestima."""
    return float(np.mean(yhat - y))


def escala_naive(q_train: np.ndarray) -> float:
    """Denominador de MASE: MAE del Naive de un paso sobre el entrenamiento."""
    if len(q_train) < 2:
        return np.nan
    d = float(np.mean(np.abs(np.diff(q_train))))
    return d if d > 0 else np.nan


@dataclass
class Ventana:
    """Una evaluación: un campo, un origen, un horizonte."""

    campo: str
    modelo: str
    origen: pd.Timestamp
    h: int
    y: float
    yhat: float
    escala: float


def backtest_campo(
    t: np.ndarray,
    q: np.ndarray,
    fechas: pd.Series,
    modelos: list,
    horizonte: int = 12,
    n_origenes: int = 3,
    min_train: int = 24,
) -> list[Ventana]:
    """Corre el walk-forward de un campo para todos los modelos."""
    n = len(q)
    if n < min_train + horizonte:
        return []

    # Orígenes equiespaciados en el tramo donde caben train y horizonte.
    ultimo = n - horizonte
    origenes = np.unique(
        np.linspace(min_train, ultimo, n_origenes).astype(int)
    )

    filas: list[Ventana] = []
    for corte in origenes:
        t_tr, q_tr = t[:corte], q[:corte]
        t_te, q_te = t[corte : corte + horizonte], q[corte : corte + horizonte]
        esc = escala_naive(q_tr)

        for modelo in modelos:
            try:
                pred = modelo.fit(t_tr, q_tr).predict(t_te)
            except Exception:  # un modelo que falla no debe tumbar el benchmark
                continue

            for j, (yv, yh) in enumerate(zip(q_te, pred), start=1):
                filas.append(
                    Ventana(
                        campo="",
                        modelo=modelo.nombre,
                        origen=fechas.iloc[corte - 1],
                        h=j,
                        y=float(yv),
                        yhat=float(yh),
                        escala=esc,
                    )
                )
    return filas


def resumen(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega los resultados crudos del backtest en una tabla comparativa."""
    df = df.copy()
    df["ae"] = (df.y - df.yhat).abs()
    df["se"] = (df.y - df.yhat) ** 2

    out = df.groupby("modelo").apply(
        lambda g: pd.Series(
            {
                "MAE_bpd": g.ae.mean(),
                "RMSE_bpd": np.sqrt(g.se.mean()),
                "sMAPE_%": smape(g.y.to_numpy(), g.yhat.to_numpy()),
                "MASE": (g.ae / g.escala).replace([np.inf, -np.inf], np.nan).mean(),
                "sesgo_bpd": (g.yhat - g.y).mean(),
                "n": len(g),
            }
        ),
        include_groups=False,
    )
    return out.sort_values("MASE")
