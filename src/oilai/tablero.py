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
