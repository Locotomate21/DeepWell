"""Figuras del análisis exploratorio.

Todas las figuras se generan con el mismo sistema visual y se guardan en
`reports/figures/` en PNG a 200 dpi, tamaño apto para insertar en el documento
de tesis.

Notas de diseño que condicionan el código:

* La paleta categórica es la de referencia validada. Con cuatro segmentos, los
  cuatro colores **no** superan el umbral de discriminación cuando todos los
  pares coexisten en un mismo plano (amarillo y naranja quedan en ΔE 13.7, bajo
  el piso de 15). Por eso las figuras que comparan los cuatro segmentos usan
  *small multiples*: un segmento por panel, un solo color por panel, y el resto
  de campos en gris de contexto.
* Dos de los colores quedan por debajo de 3:1 de contraste contra el fondo
  claro, así que cada panel lleva su nombre como rótulo directo: la identidad
  nunca depende solo del color.
* Sin ejes dobles en ninguna figura.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from .config import FIGURES
from .clean import build_panel
from .eda import caracterizar_campos, concentracion, serie_nacional
from .segmentacion import perfil_segmentos, representantes, segmentar

# --- Sistema visual -------------------------------------------------------

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TINTA_3 = "#8a8880"
REJILLA = "#e6e5e1"

SERIE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # slots 1-4 validados

DPI = 200


def _estilo() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SUPERFICIE,
            "axes.facecolor": SUPERFICIE,
            "savefig.facecolor": SUPERFICIE,
            "axes.edgecolor": REJILLA,
            "axes.labelcolor": TINTA_2,
            "axes.titlecolor": TINTA,
            "text.color": TINTA,
            "xtick.color": TINTA_2,
            "ytick.color": TINTA_2,
            "grid.color": REJILLA,
            "grid.linewidth": 0.8,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 2.0,
            "figure.dpi": DPI,
        }
    )


def _limpiar(ax, eje_y: bool = True) -> None:
    """Rejilla recesiva en un solo eje; los ejes no compiten con los datos."""
    ax.grid(axis="y" if eje_y else "x", alpha=0.9, zorder=0)
    ax.set_axisbelow(True)


def _miles(x, _pos) -> str:
    """Abrevia miles y millones sin colapsar marcas distintas en la misma etiqueta.

    Redondear 1 250 y 1 750 a "1k" y "2k" produce ejes con etiquetas repetidas.
    Por debajo de 10k se conserva un decimal, y se elimina el ".0" sobrante.
    """
    for corte, sufijo in ((1_000_000, "M"), (1_000, "k")):
        if abs(x) >= corte:
            valor = x / corte
            texto = f"{valor:.1f}" if abs(valor) < 10 else f"{valor:.0f}"
            return texto.removesuffix(".0") + sufijo
    return f"{x:.0f}"


def _guardar(fig, nombre: str) -> str:
    ruta = FIGURES / nombre
    fig.savefig(ruta, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    return str(ruta)


# --- Figuras --------------------------------------------------------------


def fig_produccion_nacional() -> str:
    """Serie única: la producción del país mes a mes."""
    _estilo()
    nac = serie_nacional()

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(nac.fecha, nac.bpd, color=SERIE[0], zorder=3)
    ax.fill_between(nac.fecha, 0, nac.bpd, color=SERIE[0], alpha=0.08, zorder=2)

    _limpiar(ax)
    ax.yaxis.set_major_formatter(FuncFormatter(_miles))
    ax.set_ylim(0, nac.bpd.max() * 1.12)
    ax.set_ylabel("barriles por día")
    ax.set_title("Producción fiscalizada de crudo en Colombia")

    # El desplome de 2020 no es geológico: es precio y recortes de la OPEP+.
    caida = nac.loc[nac[nac.fecha.dt.year == 2020].bpd.idxmin()]
    ax.annotate(
        f"mínimo COVID\n{caida.bpd:,.0f} bpd",
        xy=(caida.fecha, caida.bpd),
        xytext=(20, -34),
        textcoords="offset points",
        fontsize=8,
        color=TINTA_2,
        arrowprops=dict(arrowstyle="-", color=TINTA_3, lw=1),
    )

    pico = nac.loc[nac.bpd.idxmax()]
    ax.annotate(
        f"máximo {pico.bpd:,.0f} bpd",
        xy=(pico.fecha, pico.bpd),
        xytext=(6, 10),
        textcoords="offset points",
        fontsize=8,
        color=TINTA_2,
    )

    inicio, fin = nac.bpd.iloc[0], nac.bpd.iloc[-1]
    ax.text(
        0.99,
        0.10,
        f"{(fin / inicio - 1) * 100:+.0f}% entre {nac.fecha.iloc[0]:%Y} y {nac.fecha.iloc[-1]:%Y}",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color=TINTA_3,
    )

    # Transparencia sobre los datos: se dice qué se excluyó y por qué.
    todos = serie_nacional(solo_completos=False)
    excluidos = todos[~todos.reporte_completo]
    if len(excluidos):
        etiquetas = ", ".join(f"{f:%Y-%m}" for f in excluidos.fecha)
        ax.text(
            0.99,
            0.02,
            f"excluidos por publicación incompleta de la ANH: {etiquetas}",
            transform=ax.transAxes,
            ha="right",
            fontsize=7.5,
            color=TINTA_3,
            style="italic",
        )
    return _guardar(fig, "01_produccion_nacional.png")


def fig_concentracion() -> str:
    """Curva acumulada: cuántos campos hacen falta para explicar la producción."""
    _estilo()
    panel = build_panel()
    anual = panel[panel.fecha.dt.year == 2025].groupby("campo").bpd.mean()
    orden = anual.sort_values(ascending=False)
    acum = orden.cumsum() / orden.sum() * 100
    x = np.arange(1, len(acum) + 1)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(x, acum.to_numpy(), color=SERIE[0], zorder=3)
    ax.fill_between(x, 0, acum.to_numpy(), color=SERIE[0], alpha=0.08, zorder=2)

    tabla = concentracion(panel)
    for _, fila in tabla[tabla.top_n.isin([10, 50])].iterrows():
        n = int(fila.top_n)
        ax.scatter([n], [fila.pct_produccion], s=42, color=SERIE[1], zorder=5,
                   edgecolor=SUPERFICIE, linewidth=2)
        ax.annotate(
            f"  {n} campos = {fila.pct_produccion:.0f}%",
            xy=(n, fila.pct_produccion),
            xytext=(8, -4),
            textcoords="offset points",
            fontsize=8.5,
            color=TINTA,
            fontweight="bold",
        )

    _limpiar(ax)
    ax.set_xlim(0, len(acum))
    ax.set_ylim(0, 103)
    ax.set_xlabel("campos ordenados de mayor a menor producción")
    ax.set_ylabel("% acumulado de la producción")
    ax.set_title(f"La producción se concentra en pocos campos ({len(acum)} activos en 2025)")
    return _guardar(fig, "02_concentracion.png")


def fig_volatilidad_vs_tamano() -> str:
    """Por qué el benchmark de la Fase 1 dio lo que dio.

    Los campos pequeños son intrínsecamente más ruidosos. Esa es la explicación
    de que ningún modelo alcance MASE < 1 en el agregado: buena parte del error
    es irreducible.
    """
    _estilo()
    d = caracterizar_campos()
    d = d[d.volatilidad.notna() & (d.bpd_medio > 0)]

    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.scatter(
        d.bpd_medio, d.volatilidad,
        s=16, color=SERIE[0], alpha=0.35, linewidth=0, zorder=3,
    )

    # Mediana por decil de tamaño: la tendencia sin suponer forma funcional.
    d = d.copy()
    d["bin"] = pd.qcut(np.log10(d.bpd_medio.clip(lower=1)), 10, duplicates="drop")
    med = d.groupby("bin", observed=True).agg(
        x=("bpd_medio", "median"), y=("volatilidad", "median")
    )
    ax.plot(med.x, med.y, color=SERIE[1], zorder=4,
            marker="o", markersize=5, markeredgecolor=SUPERFICIE, markeredgewidth=1.5)
    ax.annotate(
        "mediana por decil de tamaño",
        xy=(med.x.iloc[0], med.y.iloc[0]),
        xytext=(4, 18),
        textcoords="offset points",
        fontsize=8.5,
        color=SERIE[1],
        fontweight="bold",
        ha="left",
    )

    ax.set_xscale("log")
    _limpiar(ax)
    ax.xaxis.set_major_formatter(FuncFormatter(_miles))
    ax.set_xlabel("producción media del campo (bpd, escala log)")
    ax.set_ylabel("volatilidad mensual (desv. est. de Δlog)")
    ax.set_title("Los campos pequeños son intrínsecamente menos pronosticables")
    ax.text(
        0.99, 0.95,
        f"{len(d)} campos",
        transform=ax.transAxes, ha="right", fontsize=8, color=TINTA_3,
    )
    return _guardar(fig, "03_volatilidad_vs_tamano.png")


def fig_segmentos() -> str:
    """Small multiples: un segmento por panel, un color por panel."""
    _estilo()
    seg = segmentar()
    perfil = perfil_segmentos(seg)
    nombres = list(perfil.index)

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7), sharex=True, sharey=True)

    for ax, nombre, color in zip(axes.ravel(), nombres, SERIE):
        resto = seg[seg.segmento_nombre != nombre]
        foco = seg[seg.segmento_nombre == nombre]

        ax.scatter(resto.madurez, resto.declinacion_anual_pct,
                   s=10, color=TINTA_3, alpha=0.18, linewidth=0, zorder=2)
        ax.scatter(foco.madurez, foco.declinacion_anual_pct,
                   s=np.clip(foco.bpd_medio / 40, 8, 220),
                   color=color, alpha=0.75, linewidth=0.5,
                   edgecolor=SUPERFICIE, zorder=3)

        fila = perfil.loc[nombre]
        ax.set_title(
            f"{nombre}\n{int(fila.campos)} campos ({int(fila.campos_activos)} activos) · "
            f"{fila.pct_produccion:.0f}% de la producción actual",
            fontsize=9.5,
        )
        _limpiar(ax)

    for ax in axes[1]:
        ax.set_xlabel("madurez (caudal actual / caudal pico)")
    for ax in axes[:, 0]:
        ax.set_ylabel("declinación anual (%)")

    fig.suptitle(
        "Cuatro comportamientos de campo, y dónde está la producción",
        fontsize=12, fontweight="bold", y=0.98,
    )
    fig.text(
        0.5, 0.005,
        "El tamaño del punto es proporcional a la producción media del campo. "
        "La declinación se recorta al 60 % anual, de ahí la acumulación en el borde superior.",
        ha="center", fontsize=8, color=TINTA_3,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    return _guardar(fig, "04_segmentos.png")


def fig_mapa() -> str:
    """Distribución geográfica; el tamaño codifica producción, sin doble código."""
    _estilo()
    d = caracterizar_campos()
    d = d[d.latitud.notna() & d.longitud.notna() & (d.bpd_ultimo > 0)]

    fig, ax = plt.subplots(figsize=(7, 8))
    ax.scatter(
        d.longitud, d.latitud,
        s=np.clip(np.sqrt(d.bpd_ultimo) * 1.6, 6, 420),
        color=SERIE[0], alpha=0.45, linewidth=0.6, edgecolor=SUPERFICIE, zorder=3,
    )

    # Los campos grandes se apiñan en los Llanos, así que rotularlos in situ
    # produce una maraña ilegible. Se apilan a la izquierda, en zona vacía del
    # mapa, y se conectan con líneas guía finas.
    top = d.nlargest(5, "bpd_ultimo").sort_values("latitud", ascending=False)
    x_etiqueta = d.longitud.min() - 1.6
    y_etiquetas = np.linspace(
        d.latitud.max() * 0.86, d.latitud.max() * 0.34, len(top)
    )

    for (_, fila), y in zip(top.iterrows(), y_etiquetas):
        ax.annotate(
            f"{fila.campo.title()} · {fila.bpd_ultimo:,.0f} bpd",
            xy=(fila.longitud, fila.latitud),
            xytext=(x_etiqueta, y),
            textcoords="data",
            fontsize=8.5,
            color=TINTA,
            fontweight="bold",
            ha="left",
            va="center",
            arrowprops=dict(
                arrowstyle="-", color=TINTA_3, lw=0.8, alpha=0.7,
                shrinkA=2, shrinkB=4,
            ),
        )

    ax.grid(alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlim(x_etiqueta - 0.4, d.longitud.max() + 1.0)
    ax.set_xlabel("longitud")
    ax.set_ylabel("latitud")
    ax.set_title(f"Campos productores de Colombia ({len(d)} campos)")
    # La esquina superior derecha del mapa está vacía: la nota no compite con
    # los datos ni con el rótulo del eje.
    ax.text(
        0.98, 0.98,
        "El área del círculo es proporcional\na la producción actual del campo.",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8, color=TINTA_3, linespacing=1.4,
    )
    return _guardar(fig, "05_mapa_campos.png")


def fig_curvas_ejemplo() -> str:
    """Un campo representativo por segmento: la mediana de producción del grupo."""
    _estilo()
    panel = build_panel()
    seg = segmentar()
    perfil = perfil_segmentos(seg)
    nombres = list(perfil.index)

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6), sharex=True)

    elegidos = representantes(seg)

    for ax, nombre, color in zip(axes.ravel(), nombres, SERIE):
        campo = elegidos[nombre]
        g = panel[panel.campo == campo].sort_values("fecha")
        ax.plot(g.fecha, g.bpd, color=color, zorder=3)
        ax.fill_between(g.fecha, 0, g.bpd, color=color, alpha=0.10, zorder=2)

        _limpiar(ax)
        ax.set_ylim(0, max(g.bpd.max() * 1.15, 1))
        ax.yaxis.set_major_formatter(FuncFormatter(_miles))
        ax.set_title(f"{nombre}\n{campo.title()}", fontsize=9.5)

    for ax in axes[:, 0]:
        ax.set_ylabel("bpd")

    fig.suptitle(
        "Un campo representativo de cada segmento",
        fontsize=12, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _guardar(fig, "06_curvas_ejemplo.png")


TODAS = [
    fig_produccion_nacional,
    fig_concentracion,
    fig_volatilidad_vs_tamano,
    fig_segmentos,
    fig_mapa,
    fig_curvas_ejemplo,
]


def generar_todas() -> list[str]:
    return [f() for f in TODAS]
