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


# --- Figuras de la Fase 3 --------------------------------------------------

# Par divergente azul<->rojo con gris neutro, para magnitudes con signo.
POSITIVO = "#2a78d6"
NEGATIVO = "#e34948"
NEUTRO = "#f0efec"

MODELOS_DESTACADOS = ["ML-global", "Naive", "Arps-24m"]


def _mase_por_horizonte() -> "pd.DataFrame":
    from .config import REPORTS

    df = pd.read_parquet(REPORTS / "backtest_global.parquet")
    df["mase"] = (df.y - df.yhat).abs() / df.escala
    return df.pivot_table(index="modelo", columns="h", values="mase", aggfunc="mean")


def fig_ml_vs_referencias() -> str:
    """Dónde gana el modelo global y dónde no.

    Dos paneles porque son dos preguntas distintas: el nivel de error por
    horizonte, y la diferencia respecto a la referencia que hay que batir. Un
    solo panel con dos escalas sería un eje doble, que no se usa.
    """
    _estilo()
    piv = _mase_por_horizonte()
    hs = list(piv.columns)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.5, 7), height_ratios=[1.35, 1], sharex=True
    )

    # Panel superior: nivel de error. Tres series, con leyenda y rótulo directo.
    presentes = [(m, c) for m, c in zip(MODELOS_DESTACADOS, SERIE) if m in piv.index]

    for modelo, color in presentes:
        ax1.plot(hs, piv.loc[modelo], color=color, zorder=3, label=modelo,
                 marker="o", markersize=4.5,
                 markeredgecolor=SUPERFICIE, markeredgewidth=1.2)

    # Naive y Arps terminan casi en el mismo valor: sin separación mínima los
    # rótulos directos se superponen y dejan de identificar nada.
    finales = sorted(
        ((piv.loc[m].iloc[-1], m, c) for m, c in presentes), key=lambda x: x[0]
    )
    rango = float(np.ptp(piv.loc[[m for m, _ in presentes]].to_numpy()))
    separacion = rango * 0.06

    posiciones: list[float] = []
    for valor, _, _ in finales:
        y = valor if not posiciones else max(valor, posiciones[-1] + separacion)
        posiciones.append(y)

    for (valor, modelo, color), y in zip(finales, posiciones):
        ax1.annotate(
            f" {modelo}",
            xy=(hs[-1], valor),
            xytext=(hs[-1] + 0.35, y),
            textcoords="data",
            fontsize=8.5, color=color, fontweight="bold", va="center",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.7, alpha=0.5,
                            shrinkA=0, shrinkB=2),
        )

    _limpiar(ax1)
    ax1.set_ylabel("MASE (menor es mejor)")
    ax1.set_title("El modelo global gana solo a horizontes largos")
    ax1.set_xlim(0.5, hs[-1] + 2.6)
    ax1.legend(frameon=False, loc="upper left", fontsize=8.5)
    ax1.axhline(1.0, color=TINTA_3, lw=1, ls=":", zorder=1)
    ax1.annotate(
        "MASE = 1: igual que el pronóstico ingenuo de un paso",
        xy=(hs[-1] + 2.4, 1.0), xytext=(0, 6), textcoords="offset points",
        fontsize=7.5, color=TINTA_3, ha="right",
    )

    # Panel inferior: diferencia con signo -> par divergente.
    ganancia = (piv.loc["Naive"] - piv.loc["ML-global"]) / piv.loc["Naive"] * 100
    colores = [POSITIVO if v >= 0 else NEGATIVO for v in ganancia]
    ax2.bar(hs, ganancia, color=colores, zorder=3, width=0.62)
    ax2.axhline(0, color=TINTA_2, lw=1, zorder=4)

    for h, v in zip(hs, ganancia):
        ax2.annotate(
            f"{v:+.0f}",
            xy=(h, v),
            xytext=(0, 4 if v >= 0 else -12),
            textcoords="offset points",
            ha="center", fontsize=7.5, color=TINTA_2,
        )

    _limpiar(ax2)
    ax2.set_xlabel("horizonte de pronóstico (meses)")
    ax2.set_ylabel("ventaja sobre Naive (%)")
    ax2.set_xticks(hs)
    margen = max(abs(ganancia.min()), abs(ganancia.max())) * 0.35
    ax2.set_ylim(ganancia.min() - margen, ganancia.max() + margen)
    ax2.text(
        0.99, 0.06,
        "azul: el modelo global es mejor · rojo: peor",
        transform=ax2.transAxes, ha="right", fontsize=8, color=TINTA_3,
    )

    fig.tight_layout()
    return _guardar(fig, "07_ml_vs_referencias.png")


def fig_importancia_variables() -> str:
    """Qué mira el modelo. Serie única: no hace falta codificar por color."""
    _estilo()
    from .config import REPORTS

    imp = pd.read_csv(REPORTS / "importancia_variables.csv", index_col=0)
    imp = imp.iloc[:, 0].head(14).sort_values()

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.barh(imp.index, imp.to_numpy(), color=SERIE[0], zorder=3, height=0.68)

    for nombre, valor in imp.items():
        ax.annotate(
            f" {valor:.1f}%",
            xy=(valor, nombre), xytext=(4, 0), textcoords="offset points",
            va="center", fontsize=8, color=TINTA_2,
        )

    _limpiar(ax, eje_y=False)
    ax.set_xlim(0, imp.max() * 1.18)
    ax.set_xlabel("ganancia aportada al modelo (%)")
    ax.set_title("Qué variables usa el modelo global")
    fig.tight_layout()
    return _guardar(fig, "08_importancia_variables.png")


TODAS_FASE3 = [fig_ml_vs_referencias, fig_importancia_variables]


def generar_fase3() -> list[str]:
    return [f() for f in TODAS_FASE3]


# --- Figuras de la Fase 4 --------------------------------------------------

MODELOS_FASE4 = ["Híbrido-regimen", "ML-global", "Naive"]
BASES_PESOS = ["Naive", "Arps-24m", "ML-global"]


def _mase_hibrido() -> "pd.DataFrame":
    from .config import REPORTS

    df = pd.read_parquet(REPORTS / "backtest_hibrido.parquet")
    df["mase"] = (df.y - df.yhat).abs() / df.escala
    return df


def fig_hibrido_vs_todos() -> str:
    """El híbrido frente al mejor modelo puro y a la referencia."""
    _estilo()
    df = _mase_hibrido()
    piv = df.pivot_table(index="modelo", columns="h", values="mase", aggfunc="mean")
    hs = list(piv.columns)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    for modelo, color in zip(MODELOS_FASE4, SERIE):
        if modelo not in piv.index:
            continue
        ancho = 2.6 if modelo == "Híbrido-regimen" else 1.8
        ax.plot(hs, piv.loc[modelo], color=color, zorder=3, label=modelo,
                linewidth=ancho, marker="o", markersize=4.5,
                markeredgecolor=SUPERFICIE, markeredgewidth=1.2)

    # Separación mínima entre rótulos: Naive y ML terminan muy cerca.
    finales = sorted(
        ((piv.loc[m].iloc[-1], m, c) for m, c in zip(MODELOS_FASE4, SERIE)
         if m in piv.index),
        key=lambda x: x[0],
    )
    rango = float(np.ptp(piv.loc[[m for m in MODELOS_FASE4 if m in piv.index]].to_numpy()))
    posiciones: list[float] = []
    for valor, _, _ in finales:
        posiciones.append(valor if not posiciones else max(valor, posiciones[-1] + rango * 0.07))

    for (valor, modelo, color), y in zip(finales, posiciones):
        ax.annotate(
            f" {modelo}", xy=(hs[-1], valor), xytext=(hs[-1] + 0.35, y),
            textcoords="data", fontsize=8.5, color=color, fontweight="bold",
            va="center",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.7, alpha=0.5,
                            shrinkA=0, shrinkB=2),
        )

    _limpiar(ax)
    ax.set_xlim(0.5, hs[-1] + 2.8)
    ax.set_xticks(hs)
    ax.set_xlabel("horizonte de pronóstico (meses)")
    ax.set_ylabel("MASE (menor es mejor)")
    ax.set_title("El híbrido gana en todos los horizontes, incluido el primero")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.tight_layout()
    return _guardar(fig, "09_hibrido_vs_modelos.png")


def fig_pesos_hibrido() -> str:
    """Los pesos aprendidos: cuánto confiar en cada modelo según el horizonte.

    Es el entregable interpretable de la fase. Barras apiladas porque los pesos
    suman uno por construcción, con separación entre segmentos para que el
    apilado se lea como partes de un todo y no como un bloque continuo.
    """
    _estilo()
    from .config import REPORTS

    pesos = pd.read_csv(REPORTS / "pesos_hibrido.csv")
    medios = pesos.groupby("h")[BASES_PESOS].mean()
    hs = medios.index.to_numpy()

    fig, ax = plt.subplots(figsize=(8.5, 4.6))

    base = np.zeros(len(hs))
    for modelo, color in zip(BASES_PESOS, SERIE):
        valores = medios[modelo].to_numpy()
        ax.bar(hs, valores, bottom=base, color=color, label=modelo,
               width=0.68, zorder=3,
               edgecolor=SUPERFICIE, linewidth=1.4)
        base += valores

    _limpiar(ax)
    ax.set_ylim(0, 1)
    ax.set_xticks(hs)
    ax.set_xlabel("horizonte de pronóstico (meses)")
    ax.set_ylabel("peso asignado")
    ax.set_title("De la persistencia al aprendizaje: los pesos que el híbrido aprende")
    ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=8.5,
              bbox_to_anchor=(0.5, -0.16))

    # Se rotulan solo los extremos: la lectura es la tendencia, no cada valor.
    for h in (hs[0], hs[-1]):
        fila = medios.loc[h]
        acumulado = 0.0
        for modelo in BASES_PESOS:
            valor = fila[modelo]
            if valor >= 0.12:
                ax.text(h, acumulado + valor / 2, f"{valor:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold", zorder=5)
            acumulado += valor

    fig.text(
        0.5, -0.02,
        "Promedio de los tres cortes. A un mes manda la persistencia; "
        "a doce, el modelo global y la física.",
        ha="center", fontsize=8, color=TINTA_3,
    )
    fig.tight_layout()
    return _guardar(fig, "10_pesos_hibrido.png")


def fig_sintesis() -> str:
    """La figura de cierre: qué se ganó y frente a qué.

    Serie única, así que no hace falta codificar nada por color. Se destaca el
    modelo ganador y el resto queda en un tono recesivo: el contraste dirige la
    lectura sin introducir una escala categórica que no aporta.
    """
    _estilo()
    from .config import REPORTS
    from .evaluate import resumen

    df = pd.read_parquet(REPORTS / "backtest_hibrido.parquet")
    tabla = resumen(df).sort_values("MASE", ascending=False)

    ganador = tabla.MASE.idxmin()
    referencia = tabla.loc["Naive", "MASE"] if "Naive" in tabla.index else None

    colores = [SERIE[0] if m == ganador else "#c9c8c2" for m in tabla.index]

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    barras = ax.barh(tabla.index, tabla.MASE, color=colores, height=0.66,
                     zorder=3, edgecolor=SUPERFICIE, linewidth=1.2)

    for barra, (modelo, fila) in zip(barras, tabla.iterrows()):
        texto = f" {fila.MASE:.3f}"
        if referencia is not None and modelo != "Naive":
            texto += f"  ({(1 - fila.MASE / referencia) * 100:+.1f}% vs Naive)"
        ax.annotate(
            texto, xy=(fila.MASE, barra.get_y() + barra.get_height() / 2),
            xytext=(4, 0), textcoords="offset points", va="center",
            fontsize=8.5, color=TINTA if modelo == ganador else TINTA_2,
            fontweight="bold" if modelo == ganador else "normal",
        )

    if referencia is not None:
        ax.axvline(referencia, color=TINTA_3, lw=1, ls="--", zorder=4)

    _limpiar(ax, eje_y=False)
    ax.set_xlim(0, tabla.MASE.max() * 1.42)
    ax.set_xlabel("MASE (menor es mejor)")
    ax.set_title("Resultado final: el híbrido por régimen es el mejor modelo")

    # El pie va fuera del área de datos: dentro pisaba la última barra.
    fig.text(
        0.5, -0.02,
        f"{len(df):,} predicciones · {df.campo.nunique()} campos · "
        f"{df.origen.nunique()} cortes de calendario. "
        "La línea discontinua marca el pronóstico ingenuo.",
        ha="center", fontsize=8, color=TINTA_3,
    )
    fig.tight_layout()
    return _guardar(fig, "13_sintesis_modelos.png")


TODAS_FASE4 = [fig_hibrido_vs_todos, fig_pesos_hibrido, fig_sintesis]


def generar_fase4() -> list[str]:
    return [f() for f in TODAS_FASE4]


# --- Figuras de la Fase 5 --------------------------------------------------

METODOS_INTERVALO = ["Conformal-clase", "Conformal", "Cuantílica"]


def fig_calibracion() -> str:
    """Si un intervalo dice 80 %, debe cubrir el 80 %.

    Dos paneles porque la calibración se puede fallar de dos formas distintas:
    empeorando con el horizonte, o compensando entre subpoblaciones un promedio
    que parece correcto.
    """
    _estilo()
    from .config import REPORTS
    from .incertidumbre import NIVEL

    iv = pd.read_parquet(REPORTS / "intervalos.parquet")
    iv["dentro"] = (iv.y >= iv.lo) & (iv.y <= iv.hi)
    objetivo = NIVEL * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    # Panel A: cobertura por horizonte.
    por_h = iv.pivot_table(index="metodo", columns="h", values="dentro",
                           aggfunc="mean") * 100
    hs = list(por_h.columns)

    for metodo, color in zip(METODOS_INTERVALO, SERIE):
        if metodo in por_h.index:
            ax1.plot(hs, por_h.loc[metodo], color=color, label=metodo, zorder=3,
                     marker="o", markersize=4.5,
                     markeredgecolor=SUPERFICIE, markeredgewidth=1.2)

    ax1.axhline(objetivo, color=TINTA_2, lw=1.2, ls="--", zorder=4)
    ax1.annotate(f"objetivo {objetivo:.0f} %", xy=(hs[0], objetivo),
                 xytext=(2, 6), textcoords="offset points",
                 ha="left", fontsize=8, color=TINTA_2, fontweight="bold")
    _limpiar(ax1)
    ax1.set_xticks(hs)
    ax1.set_xlabel("horizonte (meses)")
    ax1.set_ylabel("cobertura observada (%)")
    ax1.set_title("Calibración según el horizonte", fontsize=10.5)

    # Panel B: cobertura por clase de campo.
    orden = ["<0.5k", "0.5-5k", "5-50k", ">50k"]
    por_clase = iv.pivot_table(index="metodo", columns="clase", values="dentro",
                               aggfunc="mean", observed=True) * 100
    presentes = [c for c in orden if c in por_clase.columns]
    x = np.arange(len(presentes))
    ancho = 0.26

    for i, (metodo, color) in enumerate(zip(METODOS_INTERVALO, SERIE)):
        if metodo not in por_clase.index:
            continue
        ax2.bar(x + (i - 1) * ancho, por_clase.loc[metodo, presentes],
                width=ancho * 0.9, color=color, label=metodo, zorder=3,
                edgecolor=SUPERFICIE, linewidth=1.2)

    ax2.axhline(objetivo, color=TINTA_2, lw=1.2, ls="--", zorder=4)
    _limpiar(ax2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(presentes)
    ax2.set_ylim(0, 105)
    ax2.set_xlabel("tamaño del campo (bpd)")
    ax2.set_ylabel("cobertura observada (%)")
    ax2.set_title("Calibración dentro de cada tamaño", fontsize=10.5)

    # Una sola leyenda para los dos paneles: los mismos métodos con los mismos
    # colores, así la identidad no depende de recordar el panel de al lado.
    asas, etiquetas = ax1.get_legend_handles_labels()
    fig.legend(asas, etiquetas, frameon=False, fontsize=9, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Un intervalo del 80 % debe cubrir el 80 %, y en todas partes",
                 fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    return _guardar(fig, "11_calibracion_intervalos.png")


def fig_validacion_alertas() -> str:
    """¿Una alerta anticipa algo? Comparación contra los meses sin alerta."""
    _estilo()
    from .config import REPORTS
    from .anomalias import validar

    ev = pd.read_parquet(REPORTS / "evolucion_alertas.parquet")
    tabla = validar(ev)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    # Panel A: distribución del cociente posterior/previo.
    grupos = [
        ("alerta de caída", ev[ev.anomalia_baja], SERIE[1]),
        ("sin alerta", ev[~ev.anomalia], SERIE[0]),
    ]
    bins = np.linspace(0, 2, 41)

    for nombre, g, color in grupos:
        if g.empty:
            continue
        ax1.hist(g.cociente.clip(0, 2), bins=bins, density=True, color=color,
                 alpha=0.55, label=f"{nombre} (n={len(g):,})", zorder=3)
        ax1.axvline(g.cociente.median(), color=color, lw=2, ls="--", zorder=4)

    ax1.axvline(1.0, color=TINTA_2, lw=1, zorder=2)
    _limpiar(ax1)
    ax1.set_xlabel("producción posterior / producción previa (6 meses)")
    ax1.set_ylabel("densidad")
    ax1.set_title("Lo que ocurre después", fontsize=10.5)
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    ax1.text(0.99, 0.55, "las líneas punteadas\nson las medianas",
             transform=ax1.transAxes, ha="right", fontsize=7.5, color=TINTA_3)

    # Panel B: probabilidad de caída sostenida.
    etiquetas = ["cae más\ndel 20 %", "cae más\ndel 50 %"]
    x = np.arange(len(etiquetas))
    ancho = 0.34

    for i, (nombre, color) in enumerate(
        (("alerta de caída", SERIE[1]), ("sin alerta", SERIE[0]))
    ):
        if nombre not in tabla.index:
            continue
        valores = [tabla.loc[nombre, "pct_cae_mas_20"], tabla.loc[nombre, "pct_cae_mas_50"]]
        barras = ax2.bar(x + (i - 0.5) * ancho, valores, width=ancho * 0.9,
                         color=color, label=nombre, zorder=3,
                         edgecolor=SUPERFICIE, linewidth=1.2)
        for barra, v in zip(barras, valores):
            ax2.annotate(f"{v:.0f} %", xy=(barra.get_x() + barra.get_width() / 2, v),
                         xytext=(0, 4), textcoords="offset points",
                         ha="center", fontsize=9, fontweight="bold", color=TINTA)

    _limpiar(ax2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(etiquetas)
    ax2.set_ylabel("% de casos")
    ax2.set_title("Probabilidad de caída sostenida", fontsize=10.5)
    ax2.legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle("Una alerta anticipa una caída sostenida",
                 fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _guardar(fig, "12_validacion_alertas.png")


TODAS_FASE5 = [fig_calibracion, fig_validacion_alertas]


def generar_fase5() -> list[str]:
    return [f() for f in TODAS_FASE5]
