"""Modelos híbridos: física de yacimientos combinada con aprendizaje automático.

Las tres fases anteriores dejaron un hecho establecido: **ningún modelo gana en
todos los regímenes**. El pronóstico ingenuo domina a un mes, el modelo global a
partir del tercero, y Arps se defiende en los campos grandes. La Fase 4 explora
dos formas de combinarlos, deliberadamente distintas en filosofía:

**Vía A — Arps como variable (híbrido implícito).** Se calcula el pronóstico de
Arps para cada campo, origen y horizonte, y se entrega al modelo global como una
variable más. El modelo aprende por sí mismo cuándo la física es fiable y cuándo
conviene ignorarla. Es flexible pero opaco: la combinación queda dentro de los
árboles.

**Vía B — Combinación convexa por régimen (híbrido explícito).** Se estiman pesos
que suman uno sobre los pronósticos de los modelos base, por separado para cada
horizonte y clase de campo. Es menos flexible, pero cada peso es legible: dice
literalmente cuánto confiar en cada modelo en cada situación. Para una empresa
operadora eso vale tanto como el error.

Ambas se evalúan bajo el protocolo de calendario de la Fase 3, contra los mismos
campos y horizontes, de modo que la comparación con el modelo global es directa.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from ..config import DATA_PROCESSED
from ..features import OBJETIVO_MAX, OBJETIVO_MIN, PISO_BPD
from .arps import arps_multiple, fit_mejor

AJUSTES_PARQUET = DATA_PROCESSED / "ajustes_arps_por_origen.parquet"

# Ventana de ajuste: la que ganó en el benchmark de la Fase 1.
VENTANA_ARPS = 24

# Nombres de las variables que la vía A añade al modelo global.
VARIABLES_ARPS = ["arps_rel", "arps_di", "arps_b"]


# --- Vía A: Arps como variable --------------------------------------------


def ajustes_arps(
    panel: pd.DataFrame,
    ventana: int = VENTANA_ARPS,
    min_historia: int = 24,
    force: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """Ajuste de Arps en cada (campo, origen) posible.

    Se calcula una sola vez y se cachea: el ajuste en el origen `t` depende
    únicamente de datos anteriores o iguales a `t`, así que el mismo resultado
    sirve para los tres cortes de evaluación sin riesgo de fuga.

    El eje temporal es de **meses de calendario** desde el inicio de la ventana,
    no de posiciones: en un campo con huecos, doce reportes atrás no son doce
    meses atrás y la declinación estimada saldría mal.
    """
    if AJUSTES_PARQUET.exists() and not force:
        return pd.read_parquet(AJUSTES_PARQUET)

    filas = []
    campos = panel.campo.unique()

    for i, (campo, g) in enumerate(panel.groupby("campo", sort=False), 1):
        g = g.sort_values("fecha")
        fechas = g.fecha.to_numpy()
        q = g.bpd.to_numpy(float)

        if len(g) < min_historia:
            continue

        meses = (
            (g.fecha.dt.year - g.fecha.dt.year.iloc[0]) * 12
            + (g.fecha.dt.month - g.fecha.dt.month.iloc[0])
        ).to_numpy(float)

        for j in range(min_historia - 1, len(g)):
            desde = max(0, j - ventana + 1)
            t_ventana = meses[desde : j + 1]
            q_ventana = q[desde : j + 1]

            # El origen queda en el instante t_origen dentro del eje del ajuste.
            t_rel = t_ventana - t_ventana[0]
            ajuste = fit_mejor(t_rel, q_ventana)

            filas.append(
                {
                    "campo": campo,
                    "origen": fechas[j],
                    "arps_qi": ajuste.qi,
                    "arps_di": ajuste.di,
                    "arps_b": ajuste.b,
                    "arps_t_origen": float(t_rel[-1]),
                    "arps_ok": ajuste.exito,
                }
            )

        if verbose and i % 100 == 0:
            print(f"  {i}/{len(campos)} campos ajustados", flush=True)

    out = pd.DataFrame(filas)
    out.to_parquet(AJUSTES_PARQUET, index=False)
    return out


def prediccion_arps(ajustes: pd.DataFrame, h: np.ndarray) -> np.ndarray:
    """Caudal que Arps proyecta `h` meses después del origen."""
    pred = arps_multiple(
        ajustes.arps_t_origen.to_numpy(float) + np.asarray(h, float),
        ajustes.arps_qi.to_numpy(float),
        ajustes.arps_di.to_numpy(float),
        ajustes.arps_b.to_numpy(float),
    )
    return np.clip(np.nan_to_num(pred, nan=0.0, posinf=0.0), 0.0, None)


def agregar_variable_arps(
    muestras: pd.DataFrame, ajustes: pd.DataFrame
) -> pd.DataFrame:
    """Añade el pronóstico de Arps al conjunto supervisado, en escala relativa.

    Se entrega como `log(q_arps / ancla)`, la misma escala del objetivo, para que
    el modelo pueda usarla como punto de partida en lugar de tener que aprender
    la conversión. Se acompaña de `Di` y `b`, que informan del régimen de
    declinación estimado.
    """
    columnas = ["campo", "origen", "arps_qi", "arps_di", "arps_b", "arps_t_origen"]
    out = muestras.merge(ajustes[columnas], on=["campo", "origen"], how="left")

    # Arps se evalúa fila a fila: cada muestra tiene su propio horizonte.
    pred = np.full(len(out), np.nan)
    tiene = out.arps_qi.notna().to_numpy()
    if tiene.any():
        pred[tiene] = prediccion_arps(out[tiene], out.h.to_numpy()[tiene])

    ancla = out.ancla_bpd.to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.log(np.clip(pred, PISO_BPD, None) / np.clip(ancla, PISO_BPD, None))

    out["arps_rel"] = np.where(tiene, np.clip(rel, OBJETIVO_MIN, OBJETIVO_MAX), np.nan)
    return out.drop(columns=["arps_qi", "arps_t_origen"])


# --- Vía B: combinación convexa por régimen -------------------------------

# Malla de pesos sobre el símplex, en pasos de 0.1. Con tres modelos son 66
# combinaciones: barato y suficiente. Una optimización continua daría pesos
# con una precisión que los datos no respaldan.
PASO_PESOS = 0.1

# Bajo este número de observaciones un bucket no da para estimar tres pesos y se
# repliega al nivel más general.
MIN_OBS_BUCKET = 60


def _malla_simplex(n: int, paso: float = PASO_PESOS) -> list[tuple[float, ...]]:
    """Todas las combinaciones de pesos no negativos que suman uno."""
    pasos = int(round(1.0 / paso))
    salida = []
    for combo in itertools.product(range(pasos + 1), repeat=n):
        if sum(combo) == pasos:
            salida.append(tuple(c / pasos for c in combo))
    return salida


def _mejores_pesos(
    matriz: np.ndarray, real: np.ndarray, malla: list[tuple[float, ...]]
) -> np.ndarray:
    """Pesos de la malla que minimizan el error absoluto medio."""
    mejor, mejor_error = None, np.inf
    for pesos in malla:
        err = np.mean(np.abs(real - matriz @ np.array(pesos)))
        if err < mejor_error:
            mejor, mejor_error = pesos, err
    return np.array(mejor)


class CombinacionPorRegimen:
    """Pesos convexos por (horizonte, clase de campo) sobre modelos base.

    Los pesos se estiman en un periodo de validación **anterior** al corte de
    evaluación, nunca sobre los datos con los que se mide. Cuando un bucket no
    reúne observaciones suficientes se usan los pesos del horizonte completo, y
    si tampoco, los globales.
    """

    nombre = "Híbrido-regimen"

    def __init__(self, modelos: list[str]):
        self.modelos = list(modelos)
        self.pesos: dict[tuple, np.ndarray] = {}
        self.pesos_horizonte: dict[int, np.ndarray] = {}
        self.pesos_globales: np.ndarray | None = None

    def _matriz(self, ancho: pd.DataFrame) -> np.ndarray:
        return ancho[self.modelos].to_numpy(float)

    def fit(self, ancho: pd.DataFrame) -> "CombinacionPorRegimen":
        """`ancho` trae una columna por modelo, más `y`, `h` y `clase`."""
        malla = _malla_simplex(len(self.modelos))

        self.pesos_globales = _mejores_pesos(
            self._matriz(ancho), ancho.y.to_numpy(float), malla
        )

        for h, g in ancho.groupby("h"):
            if len(g) >= MIN_OBS_BUCKET:
                self.pesos_horizonte[int(h)] = _mejores_pesos(
                    self._matriz(g), g.y.to_numpy(float), malla
                )

        for (h, clase), g in ancho.groupby(["h", "clase"], observed=True):
            if len(g) >= MIN_OBS_BUCKET:
                self.pesos[(int(h), clase)] = _mejores_pesos(
                    self._matriz(g), g.y.to_numpy(float), malla
                )

        return self

    def pesos_de(self, h: int, clase) -> np.ndarray:
        if (h, clase) in self.pesos:
            return self.pesos[(h, clase)]
        if h in self.pesos_horizonte:
            return self.pesos_horizonte[h]
        return self.pesos_globales

    def predict(self, ancho: pd.DataFrame) -> np.ndarray:
        matriz = self._matriz(ancho)
        pesos = np.vstack(
            [
                self.pesos_de(int(h), clase)
                for h, clase in zip(ancho.h, ancho.clase)
            ]
        )
        return np.sum(matriz * pesos, axis=1)

    def tabla_pesos(self) -> pd.DataFrame:
        """Los pesos aprendidos, para poder leerlos e interpretarlos."""
        filas = []
        for (h, clase), pesos in sorted(self.pesos.items(), key=lambda x: (x[0][0], str(x[0][1]))):
            fila = {"h": h, "clase": clase}
            fila.update(dict(zip(self.modelos, pesos)))
            filas.append(fila)
        return pd.DataFrame(filas)
