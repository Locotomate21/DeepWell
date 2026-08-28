"""Intervalos de predicción para el pronóstico de producción.

Un pronóstico puntual no basta para decidir. Una operadora que planea inversión
necesita saber si la producción del año que viene estará entre 9 000 y 11 000 bpd
o entre 4 000 y 16 000: el punto medio es el mismo y las decisiones, opuestas.

Se comparan dos maneras de construir esos intervalos, elegidas porque responden a
necesidades distintas:

**Regresión cuantílica.** Se entrenan modelos adicionales que estiman
directamente los cuantiles del objetivo en lugar de su centro. Aprovecha las
variables —puede dar intervalos más anchos en campos volátiles y más estrechos en
los estables— pero no garantiza nada sobre la cobertura real.

**Predicción conformal por particiones.** Se mide la distribución de los errores
sobre un conjunto de calibración **que el modelo no vio**, y se usa su cuantil
empírico como margen. Es agnóstica al modelo —envuelve cualquier pronosticador,
incluida la combinación convexa de la Fase 4, que no tiene verosimilitud— y bajo
intercambiabilidad garantiza la cobertura nominal.

La garantía conformal supone intercambiabilidad, que una serie temporal **no
cumple**. Por eso aquí no se invoca como teorema sino que se mide: lo que decide
es la cobertura empírica observada sobre datos posteriores al corte.

Los intervalos se construyen en escala logarítmica y se transforman después a
barriles, lo que los vuelve asimétricos — correcto para una magnitud positiva:
el margen hacia arriba en bpd es mayor que el margen hacia abajo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import OBJETIVO_MAX, OBJETIVO_MIN, columnas_predictoras
from .models.global_ml import PARAMETROS

# Nivel nominal de los intervalos. 80 % es el habitual en planeación de
# producción: da un rango accionable sin la anchura desmedida de un 95 %.
NIVEL = 0.80


def _alfas(nivel: float = NIVEL) -> tuple[float, float]:
    """Cuantiles inferior y superior para un nivel central dado."""
    resto = (1.0 - nivel) / 2.0
    return resto, 1.0 - resto


# --- Regresión cuantílica --------------------------------------------------


class ModeloCuantil:
    """LightGBM entrenado sobre la pérdida pinball para un cuantil dado."""

    def __init__(self, alfa: float, **kwargs):
        self.alfa = alfa
        self.parametros = {
            **PARAMETROS,
            "objective": "quantile",
            "alpha": alfa,
            **kwargs,
        }
        self.modelo = None
        self.columnas: list[str] = []

    def fit(self, muestras: pd.DataFrame) -> "ModeloCuantil":
        import lightgbm as lgb

        self.columnas = columnas_predictoras(muestras)
        self.modelo = lgb.LGBMRegressor(**self.parametros)
        self.modelo.set_params(n_estimators=400)
        self.modelo.fit(muestras[self.columnas], muestras.y)
        return self

    def predict_log(self, muestras: pd.DataFrame) -> np.ndarray:
        return np.clip(
            self.modelo.predict(muestras[self.columnas]), OBJETIVO_MIN, OBJETIVO_MAX
        )


# --- Predicción conformal --------------------------------------------------


def calibrar_conformal(
    residuos: pd.DataFrame, nivel: float = NIVEL
) -> dict[int, tuple[float, float]]:
    """Márgenes por horizonte a partir de los errores de calibración.

    `residuos` debe traer las columnas `h` y `residuo`, este último en la escala
    logarítmica del objetivo: `y_real - y_predicho`.

    Los márgenes se estiman **por horizonte** porque la incertidumbre crece con
    él: un único margen daría intervalos demasiado anchos a un mes y demasiado
    estrechos a doce. Y son **asimétricos** —cuantiles con signo en vez del
    cuantil del valor absoluto— porque el error de estos modelos está sesgado:
    tienden a subestimar, así que el margen hacia arriba debe ser mayor.
    """
    baja, alta = _alfas(nivel)
    salida: dict[int, tuple[float, float]] = {}

    for h, g in residuos.groupby("h"):
        r = g.residuo.to_numpy(float)
        r = r[np.isfinite(r)]
        if len(r) < 20:
            continue
        salida[int(h)] = (float(np.quantile(r, baja)), float(np.quantile(r, alta)))

    return salida


def aplicar_conformal(
    pred_log: np.ndarray,
    ancla: np.ndarray,
    h: np.ndarray,
    margenes: dict[int, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Convierte una predicción puntual en un intervalo, en barriles por día."""
    if not margenes:
        raise ValueError("no hay márgenes de calibración")

    # Respaldo para un horizonte no calibrado: el margen más ancho disponible.
    respaldo = (
        min(m[0] for m in margenes.values()),
        max(m[1] for m in margenes.values()),
    )

    bajos = np.array([margenes.get(int(x), respaldo)[0] for x in h])
    altos = np.array([margenes.get(int(x), respaldo)[1] for x in h])

    lo = ancla * np.exp(np.clip(pred_log + bajos, OBJETIVO_MIN, OBJETIVO_MAX))
    hi = ancla * np.exp(np.clip(pred_log + altos, OBJETIVO_MIN, OBJETIVO_MAX))
    return np.clip(lo, 0.0, None), np.clip(hi, 0.0, None)


# --- Métricas de calidad de intervalos -------------------------------------


def cobertura(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Fracción de valores reales que caen dentro del intervalo, en %."""
    return float(np.mean((y >= lo) & (y <= hi)) * 100)


def anchura_relativa(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Anchura media del intervalo como múltiplo del valor real.

    En barriles la anchura no es comparable entre campos de escalas distintas;
    relativizarla la vuelve interpretable: 0.5 significa que el intervalo abarca
    medio caudal del campo.
    """
    denom = np.where(y > 0, y, np.nan)
    return float(np.nanmean((hi - lo) / denom))


def winkler(
    y: np.ndarray, lo: np.ndarray, hi: np.ndarray, nivel: float = NIVEL
) -> float:
    """Puntuación de Winkler: penaliza anchura y fallos de cobertura a la vez.

    Es una regla de puntuación propia, así que no se puede mejorar haciendo
    trampa: un intervalo infinitamente ancho cubre todo pero puntúa fatal, y uno
    degenerado es estrecho pero se penaliza cada vez que falla. Menor es mejor.
    """
    alfa = 1.0 - nivel
    ancho = hi - lo
    penalizacion = np.where(
        y < lo,
        (2.0 / alfa) * (lo - y),
        np.where(y > hi, (2.0 / alfa) * (y - hi), 0.0),
    )
    return float(np.mean(ancho + penalizacion))


def resumen_intervalos(df: pd.DataFrame, nivel: float = NIVEL) -> pd.DataFrame:
    """Tabla comparativa por método. `df` trae `metodo`, `y`, `lo`, `hi`."""
    filas = []
    for metodo, g in df.groupby("metodo"):
        y = g.y.to_numpy(float)
        lo = g.lo.to_numpy(float)
        hi = g.hi.to_numpy(float)
        filas.append(
            {
                "metodo": metodo,
                "cobertura_%": cobertura(y, lo, hi),
                "objetivo_%": nivel * 100,
                "desvio_pp": cobertura(y, lo, hi) - nivel * 100,
                "anchura_rel": anchura_relativa(y, lo, hi),
                "winkler_bpd": winkler(y, lo, hi, nivel),
                "n": len(g),
            }
        )
    return pd.DataFrame(filas).set_index("metodo").sort_values("winkler_bpd")


# --- Conformal condicionada por grupo (Mondrian) ---------------------------

# Residuos mínimos para estimar los márgenes de un grupo. Por debajo, el cuantil
# empírico sería demasiado ruidoso y se prefiere el margen del horizonte.
MIN_RESIDUOS_GRUPO = 100


def calibrar_conformal_grupo(
    residuos: pd.DataFrame, nivel: float = NIVEL
) -> tuple[dict[tuple, tuple[float, float]], dict[int, tuple[float, float]]]:
    """Márgenes por (horizonte, grupo), con repliegue a solo horizonte.

    La conformal marginal garantiza la cobertura **en promedio**, no dentro de
    cada subpoblación. Con un único margen por horizonte en escala logarítmica,
    los campos grandes —mucho menos volátiles— reciben un intervalo desmesurado y
    los pequeños uno insuficiente: la cobertura agregada sale correcta ocultando
    dos errores de signo contrario.

    Calibrar por grupo (conformal de Mondrian) corrige eso a cambio de dividir la
    muestra de calibración, así que un grupo escaso se repliega al margen del
    horizonte completo.

    `residuos` debe traer `h`, `grupo` y `residuo`.
    """
    baja, alta = _alfas(nivel)

    por_grupo: dict[tuple, tuple[float, float]] = {}
    for (h, grupo), g in residuos.groupby(["h", "grupo"], observed=True):
        r = g.residuo.to_numpy(float)
        r = r[np.isfinite(r)]
        if len(r) < MIN_RESIDUOS_GRUPO:
            continue
        por_grupo[(int(h), grupo)] = (
            float(np.quantile(r, baja)),
            float(np.quantile(r, alta)),
        )

    return por_grupo, calibrar_conformal(residuos, nivel)


def aplicar_conformal_grupo(
    pred_log: np.ndarray,
    ancla: np.ndarray,
    h: np.ndarray,
    grupo: np.ndarray,
    margenes_grupo: dict[tuple, tuple[float, float]],
    margenes_horizonte: dict[int, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Igual que `aplicar_conformal`, eligiendo el margen del grupo si existe."""
    if not margenes_horizonte:
        raise ValueError("no hay márgenes de calibración")

    respaldo = (
        min(m[0] for m in margenes_horizonte.values()),
        max(m[1] for m in margenes_horizonte.values()),
    )

    pares = [
        margenes_grupo.get(
            (int(hh), gg), margenes_horizonte.get(int(hh), respaldo)
        )
        for hh, gg in zip(h, grupo)
    ]
    bajos = np.array([p[0] for p in pares])
    altos = np.array([p[1] for p in pares])

    lo = ancla * np.exp(np.clip(pred_log + bajos, OBJETIVO_MIN, OBJETIVO_MAX))
    hi = ancla * np.exp(np.clip(pred_log + altos, OBJETIVO_MIN, OBJETIVO_MAX))
    return np.clip(lo, 0.0, None), np.clip(hi, 0.0, None)


def cobertura_condicional(
    df: pd.DataFrame, por: str, nivel: float = NIVEL
) -> pd.DataFrame:
    """Cobertura de cada método dentro de cada nivel de una variable.

    Es la comprobación que distingue un intervalo bien calibrado de uno que
    acierta el promedio compensando errores opuestos.
    """
    d = df.copy()
    d["dentro"] = (d.y >= d.lo) & (d.y <= d.hi)
    tabla = (
        d.pivot_table(index="metodo", columns=por, values="dentro",
                      aggfunc="mean", observed=True) * 100
    )
    tabla["desvio_max_pp"] = (tabla - nivel * 100).abs().max(axis=1)
    return tabla.sort_values("desvio_max_pp")
