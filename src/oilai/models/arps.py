"""Curvas de declinación de Arps (1945): la referencia de la industria.

Cualquier modelo de IA que se proponga para pronóstico de producción tiene que
batir a Arps, porque es lo que un ingeniero de yacimientos usa hoy. Este módulo
implementa el ajuste de las tres formas clásicas y sirve como línea base física
del benchmark.

    q(t) = qi / (1 + b·Di·t)^(1/b)          hiperbólica  (0 < b < 1)
    q(t) = qi · exp(-Di·t)                  exponencial  (b = 0)
    q(t) = qi / (1 + Di·t)                  armónica     (b = 1)

donde q es el caudal (bpd), t el tiempo en meses desde el inicio del ajuste,
qi el caudal inicial y Di la tasa de declinación nominal inicial (1/mes).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

# b fuera de [0, 2] no tiene sentido físico; b > 1 aparece en yacimientos no
# convencionales, pero en crudo convencional colombiano acotamos a 1.
B_BOUNDS = (0.0, 1.0)
DI_BOUNDS = (1e-6, 1.0)


def arps(t: np.ndarray, qi: float, di: float, b: float) -> np.ndarray:
    """Caudal de Arps. Maneja el límite exponencial b→0 de forma continua."""
    t = np.asarray(t, dtype=float)
    if b < 1e-6:
        return qi * np.exp(-di * t)
    return qi / np.power(1.0 + b * di * t, 1.0 / b)


@dataclass
class ArpsFit:
    """Resultado del ajuste, con lo necesario para pronosticar y reportar."""

    qi: float
    di: float
    b: float
    rmse_ajuste: float
    n_puntos: int
    exito: bool

    @property
    def tipo(self) -> str:
        if self.b < 0.05:
            return "exponencial"
        if self.b > 0.95:
            return "armónica"
        return "hiperbólica"

    def predict(self, t: np.ndarray) -> np.ndarray:
        return arps(t, self.qi, self.di, self.b)


def _initial_guess(t: np.ndarray, q: np.ndarray) -> list[float]:
    """Semilla razonable: qi del primer tramo y Di de la pendiente log-lineal."""
    qi0 = float(np.nanmax(q[: max(3, len(q) // 10)]))

    # Pendiente de log(q) sobre t da la declinación exponencial equivalente.
    mask = q > 0
    if mask.sum() >= 2:
        slope = np.polyfit(t[mask], np.log(q[mask]), 1)[0]
        di0 = float(np.clip(-slope, 1e-4, 0.5))
    else:
        di0 = 0.01

    return [qi0, di0, 0.5]


def fit(
    t: np.ndarray,
    q: np.ndarray,
    b_fijo: float | None = None,
) -> ArpsFit:
    """Ajusta Arps por mínimos cuadrados no lineales acotados.

    `b_fijo` permite forzar la forma exponencial (b=0) o armónica (b=1), útil
    para comparar las tres variantes en el benchmark.
    """
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)

    valid = np.isfinite(t) & np.isfinite(q) & (q > 0)
    t, q = t[valid], q[valid]

    if len(t) < 4:
        # Sin puntos suficientes para un ajuste no lineal de 3 parámetros.
        nivel = float(np.mean(q)) if len(q) else 0.0
        return ArpsFit(nivel, 0.0, 0.0, np.inf, len(t), exito=False)

    p0 = _initial_guess(t, q)
    qi_max = float(np.nanmax(q)) * 5

    if b_fijo is None:
        bounds = ([1e-3, DI_BOUNDS[0], B_BOUNDS[0]], [qi_max, DI_BOUNDS[1], B_BOUNDS[1]])
        func = arps
    else:
        b_val = float(b_fijo)
        bounds = ([1e-3, DI_BOUNDS[0]], [qi_max, DI_BOUNDS[1]])
        p0 = p0[:2]

        def func(tt, qi, di):  # noqa: ANN001 - firma exigida por curve_fit
            return arps(tt, qi, di, b_val)

    try:
        popt, _ = curve_fit(func, t, q, p0=p0, bounds=bounds, maxfev=10_000)
    except (RuntimeError, ValueError):
        nivel = float(np.mean(q))
        return ArpsFit(nivel, 0.0, 0.0, np.inf, len(t), exito=False)

    qi, di = popt[0], popt[1]
    b = float(b_fijo) if b_fijo is not None else popt[2]

    residuo = q - arps(t, qi, di, b)
    rmse = float(np.sqrt(np.mean(residuo**2)))

    return ArpsFit(float(qi), float(di), b, rmse, len(t), exito=True)


def fit_mejor(t: np.ndarray, q: np.ndarray) -> ArpsFit:
    """Elige entre exponencial, armónica e hiperbólica libre por RMSE de ajuste."""
    candidatos = [fit(t, q, b_fijo=0.0), fit(t, q, b_fijo=1.0), fit(t, q)]
    validos = [c for c in candidatos if c.exito and np.isfinite(c.rmse_ajuste)]
    if not validos:
        return candidatos[-1]
    return min(validos, key=lambda c: c.rmse_ajuste)
