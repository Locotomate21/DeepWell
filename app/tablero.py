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
            use_container_width=True,
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
        use_container_width=True,
        height=460,
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

    campo, alertas = st.tabs(["Vista de campo", "Alertas"])
    with campo:
        vista_campo()
    with alertas:
        vista_alertas()

    st.sidebar.divider()
    st.sidebar.caption(
        "Datos: Producción Fiscalizada de Crudo Consolidada (ANH), "
        "vía Datos Abiertos Colombia."
    )


if __name__ == "__main__":
    main()
