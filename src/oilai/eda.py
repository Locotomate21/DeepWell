"""Caracterización de los campos y agregados nacionales.

El objetivo de esta etapa no es describir por describir, sino **explicar los
resultados del benchmark de la Fase 1**. Allí quedó abierta una pregunta: por
qué el pronóstico ingenuo gana en el agregado pero Arps gana en los campos
grandes. La caracterización responde midiendo, campo por campo, tres cosas que
determinan qué tan pronosticable es una serie: su tasa de declinación, su
volatilidad y su madurez.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_PROCESSED
from .clean import build_panel
from .models.arps import fit_mejor

CARACTERIZACION_PARQUET = DATA_PROCESSED / "caracterizacion_campos.parquet"

# Meses mínimos de historia para calificar un campo. Por debajo de dos años no
# se puede estimar una declinación con sentido.
MIN_MESES = 24

# Ventana de ajuste de Arps: la misma que ganó en el benchmark de la Fase 1.
VENTANA_ARPS = 36

# Meses sin reportar tras los cuales un campo se considera inactivo. Se toma 3
# para absorber el rezago habitual de publicación de la ANH sin dar por vivo un
# campo que lleva años cerrado.
MESES_INACTIVIDAD = 3


def declinacion_anual_efectiva(di: float) -> float:
    """Convierte la declinación nominal mensual de Arps a % anual efectivo.

    `Di` de Arps es una tasa nominal instantánea y no se lee de forma intuitiva.
    La declinación efectiva anual, 1 - exp(-Di·12), es el porcentaje de caudal
    que se pierde en un año y es como la industria reporta la declinación.
    """
    return float((1.0 - np.exp(-di * 12.0)) * 100.0)


def volatilidad_log(q: np.ndarray) -> float:
    """Desviación estándar de los cambios logarítmicos mes a mes.

    Se mide en logaritmos para que sea adimensional y comparable entre un campo
    de 120 000 bpd y uno de 50 bpd: expresa la variabilidad *relativa* típica de
    un mes al siguiente. Es el indicador más directo de irreducibilidad del
    error de pronóstico.
    """
    positivos = q[q > 0]
    if len(positivos) < 3:
        return np.nan
    return float(np.std(np.diff(np.log(positivos))))


def caracterizar_campos(force: bool = False) -> pd.DataFrame:
    """Una fila por campo con los descriptores que explican su pronosticabilidad."""
    if CARACTERIZACION_PARQUET.exists() and not force:
        return pd.read_parquet(CARACTERIZACION_PARQUET)

    panel = build_panel()
    ultima_fecha = panel.fecha.max()
    filas = []

    for campo, g in panel.groupby("campo"):
        g = g.sort_values("fecha")
        if len(g) < MIN_MESES:
            continue

        q = g.bpd.to_numpy(float)
        t = g.meses_desde_inicio.to_numpy(float)

        t_v, q_v = t[-VENTANA_ARPS:], q[-VENTANA_ARPS:]
        ajuste = fit_mejor(t_v - t_v[0], q_v)

        pico = float(q.max())

        # Un tercio de los campos del histórico ya no reporta. Mezclarlos con
        # los activos distorsiona cualquier cuota de producción: un campo muerto
        # en 2018 sigue aportando su producción histórica a la suma del grupo.
        meses_inactivo = int(
            round((ultima_fecha - g.fecha.iloc[-1]).days / 30.44)
        )

        filas.append(
            {
                "campo": campo,
                "activo": meses_inactivo <= MESES_INACTIVIDAD,
                "meses_inactivo": meses_inactivo,
                "operadora": g.operadora.iloc[-1],
                "departamento": g.departamento.iloc[-1],
                "latitud": g.latitud.iloc[-1],
                "longitud": g.longitud.iloc[-1],
                "meses_historia": len(g),
                "fecha_inicio": g.fecha.iloc[0],
                "fecha_fin": g.fecha.iloc[-1],
                "bpd_medio": float(q.mean()),
                "bpd_ultimo": float(q[-1]),
                "bpd_pico": pico,
                # Madurez: fracción del caudal pico que aún se produce. Cerca de
                # 1 = campo en plateau; cerca de 0 = campo casi agotado.
                "madurez": float(q[-1] / pico) if pico > 0 else np.nan,
                "declinacion_anual_pct": declinacion_anual_efectiva(ajuste.di),
                "arps_b": ajuste.b,
                "arps_tipo": ajuste.tipo,
                "arps_exito": ajuste.exito,
                "volatilidad": volatilidad_log(q),
                "meses_sin_produccion": int((q <= 0).sum()),
            }
        )

    df = pd.DataFrame(filas)
    df.to_parquet(CARACTERIZACION_PARQUET, index=False)
    return df


# Un mes cuya cobertura cae por debajo de esta fracción de la mediana local se
# considera de publicación incompleta, no una caída real de producción.
UMBRAL_COBERTURA = 0.70


def cobertura_mensual(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Detecta meses con publicación incompleta de la ANH.

    Motivación: noviembre de 2025 aparece con 93 campos reportando frente a una
    mediana de ~300, y el agregado nacional cae a 184 000 bpd. No es una caída
    real: los campos que sí reportaron ese mes traen valores normales (Chichimene
    35 477 bpd, contra 37 463 el mes anterior), simplemente **faltan las filas**
    de los demás. Incluir esos meses en cualquier serie agregada produciría un
    desplome ficticio del 75 %.

    La detección compara el número de campos que reportan cada mes contra una
    mediana móvil centrada, que absorbe la tendencia de largo plazo (el número de
    campos activos baja con los años) sin dejarse arrastrar por el mes anómalo.
    """
    panel = build_panel() if panel is None else panel

    out = (
        panel.groupby("fecha")
        .agg(bpd=("bpd", "sum"), campos_activos=("campo", "nunique"))
        .reset_index()
    )

    referencia = (
        out.campos_activos.rolling(13, center=True, min_periods=3).median()
    )
    out["campos_esperados"] = referencia
    out["cobertura"] = out.campos_activos / referencia
    out["reporte_completo"] = out.cobertura >= UMBRAL_COBERTURA

    return out


def serie_nacional(
    panel: pd.DataFrame | None = None,
    solo_completos: bool = True,
) -> pd.DataFrame:
    """Producción agregada del país por mes, en bpd.

    Por defecto excluye los meses de publicación incompleta: una serie nacional
    que los incluya es simplemente incorrecta. Con `solo_completos=False` se
    devuelven todos los meses junto con la bandera, para poder auditarlos.
    """
    out = cobertura_mensual(panel)
    if solo_completos:
        out = out[out.reporte_completo].reset_index(drop=True)
    return out


def concentracion(panel: pd.DataFrame | None = None, anio: int = 2025) -> pd.DataFrame:
    """Qué fracción de la producción aportan los N campos más grandes.

    Es el hecho estructural que condiciona todo el proyecto: si un puñado de
    campos concentra la mayor parte de la producción, el esfuerzo de modelado
    debe concentrarse ahí, y el promedio simple entre campos es una métrica
    engañosa para la relevancia práctica.
    """
    panel = build_panel() if panel is None else panel
    anual = panel[panel.fecha.dt.year == anio]
    por_campo = anual.groupby("campo").bpd.mean().sort_values(ascending=False)
    total = por_campo.sum()

    filas = []
    for n in (1, 5, 10, 20, 50, 100):
        if n > len(por_campo):
            break
        filas.append(
            {
                "top_n": n,
                "bpd": float(por_campo.head(n).sum()),
                "pct_produccion": float(por_campo.head(n).sum() / total * 100),
            }
        )

    out = pd.DataFrame(filas)
    out.attrs["anio"] = anio
    out.attrs["campos_activos"] = len(por_campo)
    out.attrs["bpd_total"] = float(total)
    return out


def indice_hhi(panel: pd.DataFrame | None = None, anio: int = 2025) -> float:
    """Índice Herfindahl-Hirschman de concentración por operadora (0-10 000).

    Por encima de 2 500 se considera un mercado altamente concentrado.
    """
    panel = build_panel() if panel is None else panel
    anual = panel[panel.fecha.dt.year == anio]
    por_op = anual.groupby("operadora").bpd.sum()
    cuotas = por_op / por_op.sum() * 100
    return float((cuotas**2).sum())


def ranking(
    columna: str,
    panel: pd.DataFrame | None = None,
    anio: int = 2025,
    top: int = 10,
) -> pd.DataFrame:
    """Participación en la producción por operadora, departamento o campo."""
    panel = build_panel() if panel is None else panel
    anual = panel[panel.fecha.dt.year == anio]

    # Media mensual del agregado, no suma: evita que un grupo con más meses
    # reportados parezca mayor de lo que es.
    por_grupo = (
        anual.groupby([columna, "fecha"]).bpd.sum().groupby(columna).mean()
    ).sort_values(ascending=False)

    out = por_grupo.head(top).to_frame("bpd")
    out["pct"] = out.bpd / por_grupo.sum() * 100
    return out.reset_index()
