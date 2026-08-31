"""Tablero DeepWell — vista de campo y lista de alertas.

Interfaz mínima sobre `oilai.tablero`, que es donde vive toda la lógica. Aquí
solo hay presentación, de modo que la aplicación se pueda cambiar sin tocar nada
de lo que las pruebas cubren.

Para ejecutarlo:

    streamlit run app/tablero.py

Requiere que el pipeline haya generado los artefactos (`oilai all`).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from oilai import tablero
from oilai.clean import build_panel
from oilai.figuras import SERIE, SUPERFICIE, TINTA_2, TINTA_3, _estilo, _limpiar, _miles
from oilai.incertidumbre import NIVEL
from matplotlib.ticker import FuncFormatter

st.set_page_config(page_title="DeepWell", page_icon="🛢️", layout="wide")


@st.cache_data(show_spinner=False)
def _panel() -> pd.DataFrame:
    return build_panel()


@st.cache_data(show_spinner=False)
def _intervalos() -> pd.DataFrame:
    return tablero.cargar_intervalos()


@st.cache_data(show_spinner=False)
def _alertas() -> pd.DataFrame:
    return tablero.cargar_alertas()


@st.cache_data(show_spinner=False)
def _campos() -> list[str]:
    return tablero.campos_disponibles()


@st.cache_data(show_spinner=False)
def _resumen() -> dict:
    return tablero.resumen_nacional()


@st.cache_data(show_spinner=False)
def _comparativa() -> pd.DataFrame:
    return tablero.cargar_comparativa()


@st.cache_data(show_spinner=False)
def _mapa(meses: int, solo_activos: bool) -> pd.DataFrame:
    return tablero.mapa_campos(
        solo_activos=solo_activos, meses_alerta=meses, alertas=_alertas()
    )


def grafica_campo(historia: pd.DataFrame, pron: pd.DataFrame, campo: str):
    """Historia observada y pronóstico fuera de muestra con su banda."""
    _estilo()
    fig, ax = plt.subplots(figsize=(11, 4.2))

    # Solo los últimos años: doce años de historia aplastan el detalle reciente.
    corte = pron.fecha_objetivo.min() - pd.DateOffset(months=36)
    reciente = historia[historia.fecha >= corte]

    ax.plot(reciente.fecha, reciente.bpd, color=TINTA_2, lw=1.8, zorder=4,
            label="producción observada")

    ax.fill_between(pron.fecha_objetivo, pron.lo, pron.hi, color=SERIE[0],
                    alpha=0.16, zorder=2,
                    label=f"intervalo del {NIVEL * 100:.0f} %")
    ax.plot(pron.fecha_objetivo, pron.punto, color=SERIE[0], lw=2.2, zorder=5,
            marker="o", markersize=4, markeredgecolor=SUPERFICIE,
            markeredgewidth=1.2, label="pronóstico del modelo")

    origen = pron.attrs.get("origen")
    if origen is not None:
        ax.axvline(origen, color=TINTA_3, lw=1, ls="--", zorder=3)
        ax.annotate("origen del pronóstico", xy=(origen, ax.get_ylim()[1]),
                    xytext=(6, -14), textcoords="offset points",
                    fontsize=8, color=TINTA_3)

    _limpiar(ax)
    ax.yaxis.set_major_formatter(FuncFormatter(_miles))
    ax.set_ylim(bottom=0)
    ax.set_ylabel("barriles por día")
    ax.set_title(f"{campo.title()} — pronóstico fuera de muestra")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=3)
    fig.tight_layout()
    return fig


def vista_campo() -> None:
    campos = _campos()
    campo = st.sidebar.selectbox("Campo", campos, index=0)

    ficha = tablero.ficha(campo)
    pron = tablero.pronostico(campo, intervalos=_intervalos())
    historia = tablero.historia(campo, panel=_panel())

    st.subheader(campo.title())

    cols = st.columns(5)
    cols[0].metric("Producción actual", f"{ficha.get('bpd_ultimo', 0):,.0f} bpd")
    cols[1].metric("Declinación anual", f"{ficha.get('declinacion_anual_pct', 0):.1f} %")
    cols[2].metric("Madurez", f"{ficha.get('madurez', 0):.0%}",
                   help="caudal actual sobre el caudal pico histórico")
    cols[3].metric("Volatilidad", f"{ficha.get('volatilidad', 0):.2f}",
                   help="desviación del cambio logarítmico mensual")
    cols[4].metric("Historia", f"{ficha.get('meses_historia', 0)} meses")

    st.caption(
        f"Operadora: {ficha.get('operadora', '—')} · "
        f"Departamento: {ficha.get('departamento', '—')} · "
        f"Segmento: {ficha.get('segmento', 'sin clasificar')}"
    )

    if pron.empty:
        st.info("Este campo no tiene pronóstico almacenado.")
        return

    st.pyplot(grafica_campo(historia, pron, campo))
    st.caption(
        "El pronóstico se emitió sin haber visto ninguno de los meses que "
        "aparecen a su derecha: el modelo se entrenó únicamente con datos "
        "anteriores al origen marcado."
    )

    alertas = tablero.alertas_campo(campo, alertas=_alertas())
    if alertas.empty:
        st.success("Sin alertas de caída registradas para este campo.")
    else:
        st.markdown(f"**{len(alertas)} alertas de caída**")
        st.dataframe(
            alertas.assign(
                fecha=lambda d: d.fecha_objetivo.dt.strftime("%Y-%m")
            )[["fecha", "y", "lo", "punto", "severidad"]].rename(
                columns={
                    "y": "real (bpd)",
                    "lo": "límite inferior",
                    "punto": "esperado",
                    "severidad": "anchuras por debajo",
                }
            ),
            hide_index=True,
            width="stretch",
        )


def vista_alertas() -> None:
    st.subheader("Campos que requieren revisión")
    st.caption(
        "Caídas por debajo del intervalo de predicción en los últimos seis "
        "meses, ordenadas por barriles perdidos frente al mínimo esperado."
    )

    recientes = tablero.alertas_recientes(alertas=_alertas())
    if recientes.empty:
        st.success("Ninguna alerta reciente.")
        return

    st.dataframe(
        recientes.assign(
            fecha=lambda d: d.fecha_objetivo.dt.strftime("%Y-%m")
        )[["campo", "fecha", "clase", "y", "lo", "deficit_bpd", "severidad"]].rename(
            columns={
                "y": "real (bpd)",
                "lo": "mínimo esperado",
                "deficit_bpd": "déficit (bpd)",
                "severidad": "anchuras por debajo",
            }
        ),
        hide_index=True,
        width="stretch",
        height=460,
    )


def grafica_mase(piv: pd.DataFrame, modelos: list[str]):
    """MASE por horizonte de los modelos seleccionados."""
    _estilo()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    hs = list(piv.columns)

    for modelo, color in zip(modelos, SERIE):
        if modelo not in piv.index:
            continue
        ax.plot(hs, piv.loc[modelo], color=color, label=modelo, zorder=3,
                marker="o", markersize=4.5,
                markeredgecolor=SUPERFICIE, markeredgewidth=1.2)

    ax.axhline(1.0, color=TINTA_3, lw=1, ls=":", zorder=2)
    _limpiar(ax)
    ax.set_xticks(hs)
    ax.set_xlabel("horizonte (meses)")
    ax.set_ylabel("MASE")
    ax.set_title("Error según el horizonte")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    return fig


def grafica_trayectorias(tray: pd.DataFrame, modelos: list[str], campo: str):
    """Lo que cada modelo predijo frente a lo que realmente ocurrió."""
    _estilo()
    fig, ax = plt.subplots(figsize=(9, 4.0))

    # El valor real va en tinta neutra, no en un color de serie: no es un
    # modelo más, es la referencia contra la que se miden todos.
    ax.plot(tray.h, tray.y, color=TINTA_2, lw=2.4, zorder=6,
            marker="o", markersize=5, markeredgecolor=SUPERFICIE,
            markeredgewidth=1.2, label="producción real")

    for modelo, color in zip(modelos, SERIE):
        if modelo not in tray.columns:
            continue
        ax.plot(tray.h, tray[modelo], color=color, lw=1.9, ls="--", zorder=4,
                label=modelo)

    _limpiar(ax)
    ax.yaxis.set_major_formatter(FuncFormatter(_miles))
    ax.set_xticks(tray.h)
    ax.set_xlabel("meses desde el origen del pronóstico")
    ax.set_ylabel("barriles por día")
    ax.set_title(f"{campo.title()} — pronóstico de cada modelo")
    ax.legend(frameon=False, fontsize=8.5, loc="best", ncol=2)
    fig.tight_layout()
    return fig


def vista_comparador() -> None:
    st.subheader("Qué modelo conviene y cuándo")

    datos = _comparativa()
    disponibles = tablero.modelos_disponibles(datos)

    st.markdown("**Todos los modelos, sobre los mismos campos y horizontes**")
    st.dataframe(
        tablero.ranking_modelos(datos).round(3).reset_index(),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Protocolo de calendario de la Fase 3: mismos campos, mismos orígenes y "
        "mismos horizontes para todos. MASE menor es mejor."
    )

    seleccion = st.multiselect(
        "Modelos a graficar",
        disponibles,
        default=disponibles[: tablero.MAX_MODELOS_GRAFICA],
        max_selections=tablero.MAX_MODELOS_GRAFICA,
        help="Hasta tres: en un mismo plano la paleta solo garantiza que los "
             "colores se distingan entre sí hasta ese número.",
    )
    if not seleccion:
        st.info("Selecciona al menos un modelo.")
        return

    st.pyplot(grafica_mase(tablero.mase_por_horizonte(datos), seleccion))

    st.divider()
    st.markdown("**Comparación sobre un campo concreto**")

    campos = tablero.campos_comparables(datos)
    campo = st.selectbox("Campo a comparar", campos,
                         index=campos.index("RUBIALES") if "RUBIALES" in campos else 0)

    tray = tablero.trayectorias(campo, datos=datos)
    if tray.empty:
        st.info("Este campo no tiene comparación almacenada.")
        return

    st.caption(f"Origen del pronóstico: {tray.attrs['origen']:%Y-%m}")
    st.pyplot(grafica_trayectorias(tray, seleccion, campo))

    st.dataframe(
        tablero.error_por_modelo(campo, datos=datos).round(2).reset_index(),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "El orden dentro de un campo no tiene por qué coincidir con el global: "
        "el ranking agregado es un promedio sobre 336 campos."
    )


def vista_mapa() -> None:
    st.subheader("Dónde están los campos y cuáles fallan")

    controles = st.columns([1, 1, 2])
    meses = controles[0].slider(
        "Ventana de alerta (meses)", 1, 6, tablero.MESES_ALERTA_MAPA,
        help="Con seis meses más de la mitad de los campos ha disparado alguna "
             "alerta y el mapa deja de servir para priorizar.",
    )
    solo_activos = controles[1].toggle("Solo campos activos", value=True)

    mapa = _mapa(meses, solo_activos)
    if mapa.empty:
        st.info("No hay campos con coordenadas para mostrar.")
        return

    resumen = tablero.resumen_mapa(mapa)
    cols = st.columns(4)
    cols[0].metric("Campos en el mapa", f"{resumen['campos']}")
    cols[1].metric("En alerta", f"{resumen['en_alerta']}")
    cols[2].metric("Producción representada", f"{resumen['bpd_total']:,.0f} bpd")
    cols[3].metric("Producción en alerta", f"{resumen['bpd_en_alerta']:,.0f} bpd")

    st.map(
        mapa,
        latitude="latitud",
        longitude="longitud",
        color="color",
        size="radio",
        height=520,
    )
    st.caption(
        "El área del círculo es proporcional a la producción actual. "
        "En rojo, los campos con una caída por debajo de su intervalo de "
        "predicción dentro de la ventana elegida."
    )

    en_alerta = mapa[mapa.en_alerta]
    if en_alerta.empty:
        st.success("Ningún campo en alerta en la ventana seleccionada.")
        return

    st.markdown(f"**{len(en_alerta)} campos en alerta**")
    st.dataframe(
        en_alerta[["campo", "operadora", "departamento", "bpd_ultimo"]].rename(
            columns={"bpd_ultimo": "producción actual (bpd)"}
        ),
        hide_index=True,
        width="stretch",
    )


def main() -> None:
    st.title("DeepWell · Pronóstico de producción petrolera")

    try:
        resumen = _resumen()
    except tablero.ArtefactoFaltante as error:
        st.error(str(error))
        return

    cols = st.columns(4)
    cols[0].metric("Producción nacional",
                   f"{resumen['bpd_nacional']:,.0f} bpd",
                   help=f"último mes con reporte completo: "
                        f"{resumen['fecha']:%Y-%m}")
    cols[1].metric("Campos activos", f"{resumen['campos_activos']}")
    cols[2].metric("Campos en el histórico", f"{resumen['campos_totales']}")
    cols[3].metric("Alertas recientes", f"{resumen['alertas_recientes']}")

    st.divider()

    mapa, campo, alertas, comparador = st.tabs(
        ["Mapa", "Vista de campo", "Alertas", "Comparador de modelos"]
    )
    with mapa:
        vista_mapa()
    with campo:
        vista_campo()
    with alertas:
        vista_alertas()
    with comparador:
        vista_comparador()

    st.sidebar.divider()
    st.sidebar.caption(
        "Datos: Producción Fiscalizada de Crudo Consolidada (ANH), "
        "vía Datos Abiertos Colombia."
    )


if __name__ == "__main__":
    main()
