"""Líneas base ingenuas.

Sin estas referencias no se puede afirmar que un modelo "funciona": en series de
producción muy suaves, repetir el último valor observado es sorprendentemente
difícil de batir a horizontes cortos. Todo modelo del benchmark implementa la
misma interfaz `fit`/`predict`.
"""

from __future__ import annotations

import numpy as np


class Naive:
    """Repite el último caudal observado (random walk)."""

    nombre = "Naive"

    def fit(self, t: np.ndarray, q: np.ndarray) -> "Naive":
        self._ultimo = float(q[-1])
        return self

    def predict(self, t_fut: np.ndarray) -> np.ndarray:
        return np.full(len(t_fut), self._ultimo)


class MediaMovil:
    """Repite el promedio de los últimos `k` meses: más robusto al ruido."""

    def __init__(self, k: int = 3):
        self.k = k
        self.nombre = f"Media-{k}m"

    def fit(self, t: np.ndarray, q: np.ndarray) -> "MediaMovil":
        self._nivel = float(np.mean(q[-self.k :]))
        return self

    def predict(self, t_fut: np.ndarray) -> np.ndarray:
        return np.full(len(t_fut), self._nivel)


class Drift:
    """Extrapola la recta entre el primer y el último punto (deriva lineal)."""

    nombre = "Drift"

    def fit(self, t: np.ndarray, q: np.ndarray) -> "Drift":
        self._t_fin = float(t[-1])
        self._q_fin = float(q[-1])
        span = float(t[-1] - t[0])
        self._pend = (q[-1] - q[0]) / span if span > 0 else 0.0
        return self

    def predict(self, t_fut: np.ndarray) -> np.ndarray:
        pred = self._q_fin + self._pend * (np.asarray(t_fut, float) - self._t_fin)
        return np.clip(pred, 0.0, None)


class ArpsModel:
    """Envuelve el ajuste de Arps en la interfaz del benchmark.

    `ventana` limita el ajuste a los últimos N meses: Arps describe la
    declinación del régimen actual, y arrastrar 12 años de historia con
    workovers e intervenciones sesga el ajuste hacia condiciones ya superadas.
    """

    def __init__(self, ventana: int | None = 36):
        self.ventana = ventana
        self.nombre = f"Arps-{ventana}m" if ventana else "Arps-todo"

    def fit(self, t: np.ndarray, q: np.ndarray) -> "ArpsModel":
        from .arps import fit_mejor

        if self.ventana is not None and len(t) > self.ventana:
            t, q = t[-self.ventana :], q[-self.ventana :]

        self._t0 = float(t[0])
        self._fit = fit_mejor(t - self._t0, q)
        return self

    def predict(self, t_fut: np.ndarray) -> np.ndarray:
        pred = self._fit.predict(np.asarray(t_fut, float) - self._t0)
        return np.clip(np.nan_to_num(pred, nan=0.0, posinf=0.0), 0.0, None)
