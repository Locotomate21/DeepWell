"""Capa de datos del tablero.

Deliberadamente **sin dependencia de Streamlit**: la interfaz vive en
`app/tablero.py` y aquí solo hay funciones puras sobre los artefactos que
generan las fases anteriores. Así la lógica se puede probar sin levantar la
aplicación, que es donde suelen esconderse los errores de un tablero.

El tablero no reentrena nada: consume los pronósticos fuera de muestra que ya
produjo la Fase 5. Lo que se muestra es exactamente lo que el modelo predijo sin
haber visto esos meses, no un ajuste sobre datos conocidos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .clean import build_panel
from .config import REPORTS
from .eda import caracterizar_campos

# Método de intervalos que se muestra: el mejor calibrado dentro de cada clase
# de campo, que es la condición que importa al mirar un campo concreto.
METODO = "Conformal-clase"

ARTEFACTOS = {
    "intervalos": REPORTS / "intervalos.parquet",
    "alertas": REPORTS / "alertas.parquet",
    "segmentos": REPORTS / "perfil_segmentos.csv",
    "comparativa": REPORTS / "backtest_hibrido.parquet",
}


class ArtefactoFaltante(FileNotFoundError):
    """Se pide algo que aún no ha generado el pipeline."""


def _exigir(nombre: str) -> Path:
    ruta = ARTEFACTOS[nombre]
    if not ruta.exists():
        raise ArtefactoFaltante(
            f"falta {ruta.name}: ejecuta `oilai all` para generar los artefactos"
        )
    return ruta


def cargar_intervalos(metodo: str = METODO) -> pd.DataFrame:
    """Pronósticos fuera de muestra con su intervalo de predicción."""
    df = pd.read_parquet(_exigir("intervalos"))
    return df[df.metodo == metodo].copy()


def cargar_alertas() -> pd.DataFrame:
    return pd.read_parquet(_exigir("alertas"))


def campos_disponibles() -> list[str]:
    """Campos con pronóstico en el tablero, ordenados por producción actual."""
    intervalos = cargar_intervalos()
    caracterizacion = caracterizar_campos().set_index("campo")

    campos = sorted(intervalos.campo.unique())
    orden = caracterizacion.bpd_ultimo.reindex(campos).fillna(0.0)
    return list(orden.sort_values(ascending=False).index)


def historia(campo: str, panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Serie mensual observada del campo."""
    panel = build_panel() if panel is None else panel
    g = panel[panel.campo == campo].sort_values("fecha")
    return g[["fecha", "bpd", "operadora", "departamento", "municipio"]].reset_index(
        drop=True
    )


def origenes_disponibles(campo: str, intervalos: pd.DataFrame | None = None) -> list:
    """Meses desde los que existe un pronóstico para este campo."""
    iv = cargar_intervalos() if intervalos is None else intervalos
    return sorted(iv[iv.campo == campo].origen.unique())


def origen_por_defecto(g: pd.DataFrame) -> pd.Timestamp:
    """Origen más reciente que aún tiene el horizonte completo.

    Los últimos orígenes solo traen uno o dos horizontes, porque los meses
    siguientes todavía no han ocurrido y no hay valor real con el que
    comparar. Mostrar uno de esos por defecto daría un tablero con una sola
    barra. Se elige el origen más reciente con la trayectoria completa, que es
    el que permite ver el pronóstico junto a lo que realmente pasó.
    """
    por_origen = g.groupby("origen").h.count()
    completos = por_origen[por_origen == por_origen.max()]
    return completos.index.max()


def pronostico(
    campo: str,
    origen: pd.Timestamp | None = None,
    intervalos: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Pronóstico a 12 meses desde un origen, con banda de incertidumbre."""
    iv = cargar_intervalos() if intervalos is None else intervalos
    g = iv[iv.campo == campo]
    if g.empty:
        return pd.DataFrame()

    origen = origen_por_defecto(g) if origen is None else pd.Timestamp(origen)
    g = g[g.origen == origen].sort_values("h")

    salida = g[["fecha_objetivo", "h", "punto", "lo", "hi", "y"]].reset_index(drop=True)
    salida.attrs["origen"] = origen
    return salida


def ficha(campo: str) -> dict:
    """Resumen del campo: identidad, estado y comportamiento."""
    caracterizacion = caracterizar_campos()
    fila = caracterizacion[caracterizacion.campo == campo]
    if fila.empty:
        return {"campo": campo}

    fila = fila.iloc[0]
    ficha = {
        "campo": campo,
        "operadora": fila.operadora,
        "departamento": fila.departamento,
        "activo": bool(fila.activo),
        "bpd_ultimo": float(fila.bpd_ultimo),
        "bpd_pico": float(fila.bpd_pico),
        "madurez": float(fila.madurez),
        "declinacion_anual_pct": float(fila.declinacion_anual_pct),
        "volatilidad": float(fila.volatilidad),
        "meses_historia": int(fila.meses_historia),
    }

    # El segmento solo existe si la Fase 2 llegó a clasificarlo.
    ruta = REPORTS.parent / "data" / "processed" / "segmentos_campos.parquet"
    if ruta.exists():
        seg = pd.read_parquet(ruta)
        coincide = seg[seg.campo == campo]
        if not coincide.empty:
            ficha["segmento"] = coincide.segmento_nombre.iloc[0]

    return ficha


def alertas_campo(campo: str, alertas: pd.DataFrame | None = None) -> pd.DataFrame:
    """Alertas de caída del campo, de la más grave a la más leve."""
    al = cargar_alertas() if alertas is None else alertas
    g = al[(al.campo == campo) & al.anomalia_baja]
    return (
        g[["fecha_objetivo", "y", "lo", "punto", "severidad"]]
        .sort_values("severidad", ascending=False)
        .reset_index(drop=True)
    )


def alertas_recientes(meses: int = 6, alertas: pd.DataFrame | None = None) -> pd.DataFrame:
    """Campos en alerta en los últimos meses, priorizados por severidad.

    Es la vista que un ingeniero de producción usaría primero: qué revisar hoy.
    """
    al = cargar_alertas() if alertas is None else alertas
    caidas = al[al.anomalia_baja]
    if caidas.empty:
        return pd.DataFrame()

    corte = caidas.fecha_objetivo.max() - pd.DateOffset(months=meses)
    recientes = caidas[caidas.fecha_objetivo > corte].copy()

    # La severidad ordena, pero la pérdida en barriles es lo que decide dónde
    # mirar primero: una caída del doble de ancho en un campo de 50 bpd importa
    # menos que una leve en uno de 20 000.
    recientes["deficit_bpd"] = recientes.lo - recientes.y

    return (
        recientes[["campo", "fecha_objetivo", "clase", "y", "lo", "punto",
                   "severidad", "deficit_bpd"]]
        .sort_values("deficit_bpd", ascending=False)
        .reset_index(drop=True)
    )


def resumen_nacional() -> dict:
    """Indicadores de cabecera del tablero."""
    from .eda import cobertura_mensual

    cobertura = cobertura_mensual()
    completos = cobertura[cobertura.reporte_completo]
    ultimo = completos.iloc[-1]

    caracterizacion = caracterizar_campos()
    alertas = cargar_alertas()
    recientes = alertas_recientes(alertas=alertas)

    return {
        "fecha": ultimo.fecha,
        "bpd_nacional": float(ultimo.bpd),
        "campos_activos": int(caracterizacion.activo.sum()),
        "campos_totales": int(len(caracterizacion)),
        "alertas_recientes": int(len(recientes)),
    }


# --- Mapa ------------------------------------------------------------------

# Solo dos colores, y a propósito. El mapa dibuja todos los campos en el mismo
# plano, así que cualquier par de colores debe ser distinguible entre sí; con
# cuatro categorías —los segmentos de la Fase 2— la paleta no supera el umbral
# de discriminación. Codificar el estado, que es binario, sí funciona y además
# responde a la pregunta operativa: dónde hay que mirar hoy.
COLOR_NORMAL = "#2a78d6"
COLOR_ALERTA = "#e34948"

# Radio en metros. La raíz cuadrada hace que el ÁREA del círculo sea
# proporcional a la producción; usar el radio directamente exageraría los campos
# grandes hasta tapar el mapa.
ESCALA_RADIO = 90
RADIO_MIN = 1_500
RADIO_MAX = 45_000


# Ventana por defecto para marcar un campo en alerta dentro del mapa. Un mes,
# no seis: con una cola del 10 %, la probabilidad de que un campo dispare al
# menos una alerta en seis meses roza el 50 %, y medio mapa sale en rojo. Medido
# sobre los datos: 157 de 294 campos con ventana de seis meses frente a 36 con
# ventana de uno. El mapa responde a "qué revisar hoy", no a "qué falló alguna
# vez este semestre"; para lo segundo está la pestaña de alertas.
MESES_ALERTA_MAPA = 1


def mapa_campos(
    solo_activos: bool = True,
    meses_alerta: int = MESES_ALERTA_MAPA,
    caracterizacion: pd.DataFrame | None = None,
    alertas: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Campos georreferenciados, con su estado de alerta y tamaño de dibujo."""
    d = caracterizar_campos() if caracterizacion is None else caracterizacion
    d = d[d.latitud.notna() & d.longitud.notna()].copy()

    if solo_activos:
        d = d[d.activo]

    recientes = alertas_recientes(meses=meses_alerta, alertas=alertas)
    en_alerta = set(recientes.campo) if not recientes.empty else set()

    d["en_alerta"] = d.campo.isin(en_alerta)
    d["color"] = np.where(d.en_alerta, COLOR_ALERTA, COLOR_NORMAL)
    d["radio"] = np.clip(
        np.sqrt(d.bpd_ultimo.clip(lower=0)) * ESCALA_RADIO, RADIO_MIN, RADIO_MAX
    )

    columnas = [
        "campo", "latitud", "longitud", "bpd_ultimo", "operadora",
        "departamento", "en_alerta", "color", "radio",
    ]
    return d[columnas].sort_values("bpd_ultimo", ascending=False).reset_index(drop=True)


def resumen_mapa(mapa: pd.DataFrame) -> dict:
    """Cifras de cabecera de la vista de mapa."""
    if mapa.empty:
        return {"campos": 0, "en_alerta": 0, "bpd_total": 0.0, "bpd_en_alerta": 0.0}

    alerta = mapa[mapa.en_alerta]
    return {
        "campos": int(len(mapa)),
        "en_alerta": int(len(alerta)),
        "bpd_total": float(mapa.bpd_ultimo.sum()),
        "bpd_en_alerta": float(alerta.bpd_ultimo.sum()),
    }


# --- Comparador de modelos -------------------------------------------------

# Máximo de modelos superpuestos en una misma gráfica. No es una preferencia
# estética: en un solo plano todos los pares de colores deben distinguirse
# entre sí, y la paleta validada solo garantiza esa separación hasta tres.
# Con más, la identidad pasaría a depender de adivinar el tono.
MAX_MODELOS_GRAFICA = 3

# Orden de preferencia al proponer una selección inicial: el híbrido ganador,
# el mejor modelo puro y la referencia que hay que batir.
MODELOS_SUGERIDOS = ["Híbrido-regimen", "ML-global", "Naive"]


def cargar_comparativa() -> pd.DataFrame:
    """Predicciones de todos los modelos bajo el protocolo de calendario."""
    return pd.read_parquet(_exigir("comparativa"))


def modelos_disponibles(datos: pd.DataFrame | None = None) -> list[str]:
    """Modelos del benchmark, con los sugeridos al principio."""
    df = cargar_comparativa() if datos is None else datos
    presentes = set(df.modelo.unique())

    orden = [m for m in MODELOS_SUGERIDOS if m in presentes]
    orden += sorted(presentes - set(orden))
    return orden


def ranking_modelos(datos: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tabla comparativa global, con las mismas métricas del benchmark."""
    from .evaluate import resumen

    df = cargar_comparativa() if datos is None else datos
    return resumen(df)


def mase_por_horizonte(datos: pd.DataFrame | None = None) -> pd.DataFrame:
    """MASE medio de cada modelo en cada horizonte."""
    df = cargar_comparativa() if datos is None else datos
    d = df.assign(mase=lambda x: (x.y - x.yhat).abs() / x.escala)
    return d.pivot_table(index="modelo", columns="h", values="mase", aggfunc="mean")


def campos_comparables(datos: pd.DataFrame | None = None) -> list[str]:
    """Campos con predicciones de todos los modelos."""
    df = cargar_comparativa() if datos is None else datos
    return sorted(df.campo.unique())


def trayectorias(
    campo: str,
    origen: pd.Timestamp | None = None,
    datos: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Lo que cada modelo predijo para un campo, junto al valor real.

    Formato ancho: una fila por horizonte, una columna por modelo, más `y`.
    """
    df = cargar_comparativa() if datos is None else datos
    g = df[df.campo == campo]
    if g.empty:
        return pd.DataFrame()

    origen = g.origen.max() if origen is None else pd.Timestamp(origen)
    g = g[g.origen == origen]

    ancho = g.pivot_table(index="h", columns="modelo", values="yhat")
    ancho.insert(0, "y", g.groupby("h").y.first())

    # Se reindexa al horizonte completo para que un mes sin dato quede como
    # hueco explícito. Sin esto la gráfica une los puntos vecinos con una recta
    # y sugiere un valor que no existe: es justo lo que pasa con noviembre de
    # 2025, el mes de publicación incompleta de la ANH, que deja sin objetivo a
    # los campos que no reportaron.
    completo = range(1, int(df.h.max()) + 1)
    salida = ancho.reindex(completo).rename_axis("h").reset_index()

    salida.attrs["origen"] = origen
    return salida


def error_por_modelo(
    campo: str,
    origen: pd.Timestamp | None = None,
    datos: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Error de cada modelo sobre un campo concreto, del mejor al peor."""
    df = cargar_comparativa() if datos is None else datos
    g = df[df.campo == campo]
    if g.empty:
        return pd.DataFrame()

    if origen is not None:
        g = g[g.origen == pd.Timestamp(origen)]

    g = g.assign(
        ae=lambda x: (x.y - x.yhat).abs(),
        se=lambda x: (x.y - x.yhat) ** 2,
    )

    tabla = g.groupby("modelo").apply(
        lambda x: pd.Series(
            {
                "MAE_bpd": x.ae.mean(),
                "RMSE_bpd": np.sqrt(x.se.mean()),
                "MASE": (x.ae / x.escala).mean(),
                "sesgo_bpd": (x.yhat - x.y).mean(),
            }
        ),
        include_groups=False,
    )
    return tabla.sort_values("MASE")
