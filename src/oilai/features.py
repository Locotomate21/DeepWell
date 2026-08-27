"""Construcción del conjunto supervisado para el modelo global.

Un modelo global aprende de los 608 campos a la vez en lugar de ajustar una
curva por campo. Eso plantea dos problemas que este módulo resuelve de forma
explícita:

1. **Escala.** La producción abarca cinco órdenes de magnitud (de 1 a 120 000
   bpd). Un modelo entrenado sobre barriles crudos dedicaría toda su capacidad a
   los campos grandes. La solución es predecir el **cambio logarítmico respecto
   a un ancla reciente**, `log(q_{t+h} / ancla_t)`, que es adimensional: un campo
   que cae un 10 % aporta la misma señal produzca 50 o 50 000 bpd.

2. **Causalidad.** Con un panel de muchos campos es fácil filtrar información
   del futuro sin darse cuenta. Aquí **toda** variable en el instante `t` se
   calcula con operaciones que solo miran hacia atrás (`shift`, `rolling`,
   `cummax`, `cumsum`), nunca con estadísticos de la serie completa. La prueba
   `test_las_variables_no_miran_al_futuro` lo verifica truncando la serie y
   comprobando que las variables no cambian.

El ancla es la mediana de los últimos tres meses observados, no el último valor:
en campos ruidosos un solo mes atípico desplazaría todas las predicciones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Piso para evitar log(0) en meses de campo cerrado. Es un caudal por debajo de
# cualquier producción real reportada, así que no distorsiona los campos vivos.
PISO_BPD = 0.1

# Recorte del objetivo en escala logarítmica. exp(-3) ~ 5 % y exp(1.5) ~ 4.5x:
# cubre cualquier transición plausible en doce meses y evita que un puñado de
# reactivaciones extremas domine la función de pérdida.
OBJETIVO_MIN, OBJETIVO_MAX = -3.0, 1.5

VENTANA_ANCLA = 3

COLUMNAS_CATEGORICAS = ["operadora", "departamento"]


def _serie_densa(g: pd.DataFrame) -> pd.Series:
    """Serie mensual continua del campo, con NaN en los meses sin reporte.

    Densificar es necesario para que `shift(12)` signifique «hace doce meses» y
    no «doce reportes atrás», que en un campo con huecos es otra cosa.
    """
    s = g.set_index("fecha").bpd.sort_index()
    idx = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    return s.reindex(idx)


def _variables_de_campo(s: pd.Series) -> pd.DataFrame:
    """Variables causales para cada instante de la serie de un campo."""
    observado = s.notna()
    q = s.clip(lower=PISO_BPD)
    log_q = np.log(q)

    # Ancla: mediana de los últimos meses observados (rolling ignora los NaN).
    ancla = q.rolling(VENTANA_ANCLA, min_periods=1).median()
    log_ancla = np.log(ancla.clip(lower=PISO_BPD))

    d_log = log_q.diff()

    X = pd.DataFrame(index=s.index)

    # Escala del campo: permite al modelo aprender que los campos pequeños son
    # más ruidosos, sin que la escala contamine el objetivo.
    X["log_ancla"] = log_ancla

    # Historia reciente, siempre relativa al ancla -> adimensional.
    # El rezago 0 es el mes del propio origen, que sí está disponible: se
    # pronostica desde un mes con reporte. Omitirlo dejaba al modelo sin el dato
    # más informativo a horizonte corto, donde repetir el último valor es difícil
    # de batir, y el ancla (mediana de tres meses) solo lo refleja difuminado.
    for k in (0, 1, 2, 3, 6, 12):
        X[f"rel_lag{k}"] = log_q.shift(k) - log_ancla

    # Tendencia: pendiente media de log q por mes en distintas ventanas.
    for k in (3, 6, 12):
        X[f"pendiente{k}"] = (log_q - log_q.shift(k)) / k

    # Volatilidad reciente: el predictor más directo del error alcanzable.
    for k in (6, 12):
        X[f"volatilidad{k}"] = d_log.rolling(k, min_periods=3).std()

    # Madurez causal: caudal actual contra el máximo visto HASTA AHORA. Usar el
    # máximo de toda la serie sería mirar al futuro.
    X["rel_maximo"] = log_ancla - np.log(q.cummax())

    # Edad y densidad de reporte.
    X["meses_observados"] = observado.cumsum()
    X["edad_meses"] = np.arange(len(s), dtype=float)
    X["frac_huecos12"] = 1.0 - observado.rolling(12, min_periods=1).mean()
    X["meses_sin_reporte"] = (
        observado.groupby(observado.cumsum()).cumcount()
    )

    X["mes"] = s.index.month
    return X


def construir_muestras(
    panel: pd.DataFrame,
    horizontes: range | list[int],
    origenes: pd.DatetimeIndex | list[pd.Timestamp] | None = None,
    min_historia: int = 24,
    con_objetivo: bool = True,
) -> pd.DataFrame:
    """Tabla (campo, origen, h) con variables causales y objetivo.

    `origenes=None` genera una muestra en cada mes posible, que es lo que se
    quiere para entrenar. Para evaluar se pasan los cortes de calendario.
    """
    horizontes = list(horizontes)
    origenes_set = set(pd.to_datetime(origenes)) if origenes is not None else None

    trozos: list[pd.DataFrame] = []

    for campo, g in panel.groupby("campo", sort=False):
        g = g.sort_values("fecha")
        if g.fecha.nunique() < min_historia:
            continue

        s = _serie_densa(g)
        X = _variables_de_campo(s)
        log_ancla = X["log_ancla"]

        # Solo son orígenes válidos los meses con reporte y con historia
        # suficiente: no se pronostica desde un mes en blanco.
        valido = s.notna() & (X["meses_observados"] >= min_historia)
        if origenes_set is not None:
            valido &= s.index.isin(origenes_set)
        if not valido.any():
            continue

        # Atributos categóricos vigentes en cada instante.
        meta = (
            g.set_index("fecha")[COLUMNAS_CATEGORICAS]
            .reindex(s.index)
            .ffill()
        )

        log_q = np.log(s.clip(lower=PISO_BPD))

        for h in horizontes:
            filas = X.loc[valido].copy()
            filas["h"] = h
            filas["campo"] = campo
            filas["origen"] = filas.index
            filas["fecha_objetivo"] = filas.index + pd.DateOffset(months=h)
            filas["ancla_bpd"] = np.exp(log_ancla.loc[valido])

            if con_objetivo:
                # shift(-h) sobre la serie densa: el valor h meses DESPUÉS.
                futuro = log_q.shift(-h)
                observado_futuro = s.shift(-h).notna()

                filas["y"] = (futuro - log_ancla).loc[valido]
                filas["bpd_real"] = s.shift(-h).loc[valido]
                filas = filas[observado_futuro.loc[valido].to_numpy()]
                filas["y"] = filas["y"].clip(OBJETIVO_MIN, OBJETIVO_MAX)

            for col in COLUMNAS_CATEGORICAS:
                filas[col] = meta[col].loc[filas.index].to_numpy()

            trozos.append(filas.reset_index(drop=True))

    if not trozos:
        return pd.DataFrame()

    out = pd.concat(trozos, ignore_index=True)
    for col in COLUMNAS_CATEGORICAS:
        out[col] = out[col].astype("category")
    return out


def columnas_predictoras(df: pd.DataFrame) -> list[str]:
    """Nombres de las variables que entran al modelo."""
    excluir = {
        "campo", "origen", "fecha_objetivo", "y", "bpd_real", "ancla_bpd",
    }
    return [c for c in df.columns if c not in excluir]


def reconstruir_bpd(ancla: np.ndarray, y_log: np.ndarray) -> np.ndarray:
    """Deshace la transformación logarítmica: de cambio relativo a bpd."""
    return np.clip(ancla * np.exp(np.clip(y_log, OBJETIVO_MIN, OBJETIVO_MAX)), 0.0, None)
