"""Segmentación de campos por comportamiento productivo.

La motivación es práctica, no taxonómica: la Fase 1 mostró que el modelo ganador
cambia según el tipo de campo. Si esos tipos pueden identificarse **antes** de
pronosticar, el modelo híbrido de la Fase 4 puede elegir su estrategia por
segmento en lugar de aplicar la misma receta a los 608 campos.

Se usan cuatro descriptores, todos calculables con datos pasados únicamente:

* `log10(bpd_medio)` — escala del campo. En logaritmo porque abarca cinco
  órdenes de magnitud y en escala lineal los campos grandes dominarían la
  distancia euclídea.
* `declinacion_anual_pct` — velocidad de agotamiento.
* `volatilidad` — variabilidad relativa mes a mes, proxy del error irreducible.
* `madurez` — fracción del caudal pico que aún se produce.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .config import DATA_PROCESSED
from .eda import caracterizar_campos

SEGMENTOS_PARQUET = DATA_PROCESSED / "segmentos_campos.parquet"

VARIABLES = ["log_bpd", "declinacion_anual_pct", "volatilidad", "madurez"]

# Recortes para que unos pocos campos extremos no capturen los centroides.
# No se eliminan campos: se acotan sus valores, porque un campo con 200 % de
# declinación aparente sigue siendo un campo agotándose y debe clasificarse.
LIMITES = {
    "declinacion_anual_pct": (0.0, 60.0),
    "volatilidad": (0.0, 2.0),
    "madurez": (0.0, 1.0),
}

SEMILLA = 42


def _matriz(df: pd.DataFrame) -> np.ndarray:
    X = df[VARIABLES].to_numpy(float)
    return StandardScaler().fit_transform(X)


def preparar(caracterizacion: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tabla de descriptores lista para agrupar, con recortes aplicados."""
    df = caracterizar_campos() if caracterizacion is None else caracterizacion.copy()

    df = df[df.volatilidad.notna()].copy()
    df["log_bpd"] = np.log10(df.bpd_medio.clip(lower=1.0))
    for col, (lo, hi) in LIMITES.items():
        df[col] = df[col].clip(lo, hi)

    return df.dropna(subset=VARIABLES)


def elegir_k(df: pd.DataFrame, ks: range = range(2, 8)) -> pd.DataFrame:
    """Silueta media para cada número de grupos candidato."""
    X = _matriz(df)
    filas = []
    for k in ks:
        etiquetas = KMeans(k, n_init=10, random_state=SEMILLA).fit_predict(X)
        filas.append({"k": k, "silueta": float(silhouette_score(X, etiquetas))})
    return pd.DataFrame(filas)


def segmentar(k: int = 4, force: bool = False) -> pd.DataFrame:
    """Asigna un segmento a cada campo y lo etiqueta de forma interpretable."""
    if SEGMENTOS_PARQUET.exists() and not force:
        return pd.read_parquet(SEGMENTOS_PARQUET)

    df = preparar()
    X = _matriz(df)

    modelo = KMeans(k, n_init=10, random_state=SEMILLA)
    df["segmento"] = modelo.fit_predict(X)
    df["silueta_global"] = float(silhouette_score(X, df.segmento))

    df["segmento_nombre"] = _nombrar(df)
    df.to_parquet(SEGMENTOS_PARQUET, index=False)
    return df


def _nombrar(df: pd.DataFrame) -> pd.Series:
    """Traduce los índices de KMeans a etiquetas legibles.

    Los índices que devuelve KMeans son arbitrarios y cambian entre corridas, así
    que se nombran por el perfil del centroide en lugar de por su número. El
    nombre se decide con reglas sobre las medianas del grupo, de modo que sea
    reproducible y auditable.
    """
    perfil = df.groupby("segmento").agg(
        bpd=("bpd_medio", "median"),
        decl=("declinacion_anual_pct", "median"),
        vol=("volatilidad", "median"),
        mad=("madurez", "median"),
    )

    nombres = {}
    for seg, fila in perfil.iterrows():
        if fila.decl >= 25:
            nombres[seg] = "En agotamiento"
        elif fila.vol >= 0.45:
            nombres[seg] = "Marginal errático"
        elif fila.mad >= 0.45 and fila.decl < 5:
            nombres[seg] = "Núcleo estable"
        else:
            nombres[seg] = "Maduro en declinación"

    # Si dos grupos recibieran el mismo nombre, se desempata por producción.
    vistos: dict[str, int] = {}
    for seg in perfil.sort_values("bpd", ascending=False).index:
        nombre = nombres[seg]
        if nombre in vistos:
            vistos[nombre] += 1
            nombres[seg] = f"{nombre} {vistos[nombre] + 1}"
        else:
            vistos[nombre] = 1

    return df.segmento.map(nombres)


def representantes(df: pd.DataFrame | None = None) -> dict[str, str]:
    """Campo más cercano al centroide de cada segmento, en el espacio estandarizado.

    Se elige por distancia al centroide y no por producción mediana: el campo
    mediano en tamaño puede ser atípico en declinación o volatilidad, y como
    ilustración del segmento resultaría engañoso.
    """
    df = segmentar() if df is None else df
    X = _matriz(df)

    elegidos: dict[str, str] = {}
    for nombre in df.segmento_nombre.unique():
        en_segmento = (df.segmento_nombre == nombre).to_numpy()
        centro = X[en_segmento].mean(axis=0)

        # El centroide se calcula con todo el segmento, pero el representante se
        # busca solo entre campos activos: ilustrar un segmento con un campo que
        # dejó de reportar hace años induce a error.
        candidatos = en_segmento & df.activo.to_numpy()
        if not candidatos.any():
            candidatos = en_segmento

        distancias = np.linalg.norm(X[candidatos] - centro, axis=1)
        elegidos[nombre] = df.loc[candidatos, "campo"].iloc[int(distancias.argmin())]

    return elegidos


def perfil_segmentos(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Resumen por segmento: tamaño, descriptores medianos y aporte al país."""
    df = segmentar() if df is None else df

    out = df.groupby("segmento_nombre").agg(
        campos=("campo", "size"),
        campos_activos=("activo", "sum"),
        bpd_mediano=("bpd_medio", "median"),
        declinacion_pct=("declinacion_anual_pct", "median"),
        volatilidad=("volatilidad", "median"),
        madurez=("madurez", "median"),
        historia_meses=("meses_historia", "median"),
    )

    # La cuota se calcula sobre producción ACTUAL de campos ACTIVOS. Usar la
    # media histórica de todos los campos sobrerrepresenta a los ya cerrados:
    # el núcleo estable pasa de un 75 % aparente a un 88 % real de lo que hoy
    # produce el país.
    activos = df[df.activo]
    bpd_actual = activos.groupby("segmento_nombre").bpd_ultimo.sum()
    out["bpd_actual"] = bpd_actual.reindex(out.index).fillna(0.0)
    out["pct_produccion"] = out.bpd_actual / out.bpd_actual.sum() * 100

    return out.sort_values("bpd_actual", ascending=False)
