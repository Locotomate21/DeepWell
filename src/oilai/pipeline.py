"""Punto de entrada del proyecto: reproduce la Fase 1 completa.

    oilai ingest      descarga el histórico de la ANH
    oilai panel       construye el panel mensual limpio
    oilai benchmark   corre el backtesting walk-forward de las líneas base
    oilai eda         caracteriza los campos, los segmenta y genera las figuras
    oilai ml          entrena y evalúa el modelo global de aprendizaje
    oilai all         las cinco, en orden

Cada etapa cachea su salida, así que volver a correr `all` es barato: solo
recalcula lo que falte. Con `--force` se rehace todo desde cero.
"""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from .clean import build_panel
from .config import FIGURES, PANEL_PARQUET, RAW_PARQUET, REPORTS
from .evaluate import resumen
from .ingest import download


def _titulo(texto: str) -> None:
    print(f"\n{'=' * 62}\n{texto}\n{'=' * 62}", flush=True)


def etapa_ingest(force: bool) -> pd.DataFrame:
    _titulo("ETAPA 1/5 · Descarga del histórico ANH")
    t0 = time.perf_counter()
    df = download(force=force)
    origen = "descargado" if force or not RAW_PARQUET.exists() else "caché"
    print(f"{len(df):,} registros ({origen}) en {time.perf_counter() - t0:.1f}s")
    print(f"-> {RAW_PARQUET}")
    return df


def etapa_panel(force: bool) -> pd.DataFrame:
    _titulo("ETAPA 2/5 · Construcción del panel mensual")
    t0 = time.perf_counter()
    panel = build_panel(force=force)
    print(f"{len(panel):,} filas · {panel.campo.nunique()} campos · "
          f"{panel.operadora.nunique()} operadoras")
    print(f"cobertura: {panel.fecha.min():%Y-%m} a {panel.fecha.max():%Y-%m}")
    print(f"completado en {time.perf_counter() - t0:.1f}s")
    print(f"-> {PANEL_PARQUET}")
    return panel


def etapa_benchmark(force: bool) -> pd.DataFrame:
    _titulo("ETAPA 3/5 · Backtesting walk-forward de líneas base")
    destino = REPORTS / "backtest_base.parquet"

    if destino.exists() and not force:
        df = pd.read_parquet(destino)
        print(f"{len(df):,} predicciones (caché)")
    else:
        from .run_benchmark import main as correr

        t0 = time.perf_counter()
        df = correr()
        print(f"completado en {time.perf_counter() - t0:.1f}s")

    print(f"{len(df):,} predicciones · {df.campo.nunique()} campos evaluados\n")

    tabla = resumen(df).round(3)
    print("Ranking global (menor MASE es mejor):")
    print(tabla.to_string())

    tabla.to_csv(REPORTS / "ranking_base.csv")
    print(f"\n-> {REPORTS / 'ranking_base.csv'}")
    return df


def etapa_eda(force: bool) -> pd.DataFrame:
    _titulo("ETAPA 4/5 · Caracterización, segmentación y figuras")

    from .eda import caracterizar_campos, cobertura_mensual, concentracion, indice_hhi
    from .figuras import generar_todas
    from .segmentacion import perfil_segmentos, representantes, segmentar

    t0 = time.perf_counter()

    campos = caracterizar_campos(force=force)
    print(f"{len(campos)} campos caracterizados "
          f"({int(campos.activo.sum())} activos, {int((~campos.activo).sum())} inactivos)")

    cobertura = cobertura_mensual()
    incompletos = cobertura[~cobertura.reporte_completo]
    if len(incompletos):
        meses = ", ".join(f"{f:%Y-%m}" for f in incompletos.fecha)
        print(f"meses excluidos por publicación incompleta de la ANH: {meses}")

    conc = concentracion()
    top10 = conc[conc.top_n == 10].pct_produccion.iloc[0]
    print(f"concentración {conc.attrs['anio']}: "
          f"{conc.attrs['campos_activos']} campos activos, "
          f"top 10 = {top10:.1f}% de la producción")
    print(f"HHI por operadora: {indice_hhi():,.0f}")

    seg = segmentar(force=force)
    perfil = perfil_segmentos(seg)
    print()
    print(f"Segmentos (silueta {seg.silueta_global.iloc[0]:.3f}):")
    print(perfil[
        ["campos", "campos_activos", "declinacion_pct", "volatilidad",
         "madurez", "pct_produccion"]
    ].round(2).to_string())

    print()
    print("Campo representativo de cada segmento:")
    for nombre, campo in representantes(seg).items():
        print(f"  {nombre:<24} {campo}")

    rutas = generar_todas()
    print()
    print(f"{len(rutas)} figuras generadas en {FIGURES}")

    perfil.to_csv(REPORTS / "perfil_segmentos.csv")
    campos.drop(columns=["fecha_inicio", "fecha_fin"]).to_csv(
        REPORTS / "caracterizacion_campos.csv", index=False
    )
    print(f"completado en {time.perf_counter() - t0:.1f}s")
    print(f"-> {REPORTS / 'perfil_segmentos.csv'}")
    return campos


def etapa_ml(force: bool) -> pd.DataFrame:
    _titulo("ETAPA 5/5 · Modelo global de aprendizaje automático")

    from .backtest_global import CORTES, main as correr_global
    from .evaluate import resumen
    from .figuras import generar_fase3

    destino = REPORTS / "backtest_global.parquet"
    t0 = time.perf_counter()

    if destino.exists() and not force:
        df = pd.read_parquet(destino)
        print(f"{len(df):,} predicciones (caché)")
    else:
        cortes = ", ".join(f"{c:%Y-%m}" for c in CORTES)
        print(f"cortes de calendario: {cortes}")
        df = correr_global()
        print(f"entrenamiento y evaluación en {time.perf_counter() - t0:.1f}s")

    print(f"{len(df):,} predicciones · {df.campo.nunique()} campos · "
          f"{df.modelo.nunique()} modelos")
    print()

    tabla = resumen(df).round(3)
    print("Ranking bajo orígenes de calendario (menor MASE es mejor):")
    print(tabla.to_string())

    df = df.copy()
    df["mase"] = (df.y - df.yhat).abs() / df.escala
    piv = df.pivot_table(index="modelo", columns="h", values="mase", aggfunc="mean")
    print()
    print("MASE por horizonte:")
    print(piv[[1, 3, 6, 9, 12]].round(3).sort_values(12).to_string())

    ventaja = (piv.loc["Naive"] - piv.loc["ML-global"]) / piv.loc["Naive"] * 100
    cruce = next((h for h in piv.columns if ventaja[h] > 0), None)
    print()
    if cruce is not None:
        print(f"El modelo global supera al Naive a partir del horizonte {cruce} "
              f"(ventaja de {ventaja.iloc[-1]:+.1f}% a 12 meses).")
    else:
        print("El modelo global no supera al Naive en ningún horizonte.")

    tabla.to_csv(REPORTS / "ranking_global.csv")
    piv.to_csv(REPORTS / "mase_por_horizonte.csv")

    rutas = generar_fase3()
    print(f"{len(rutas)} figuras generadas en {FIGURES}")
    print(f"-> {REPORTS / 'ranking_global.csv'}")
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="oilai",
        description="Pipeline de pronóstico de producción petrolera (Fase 1).",
    )
    parser.add_argument(
        "etapa",
        choices=["ingest", "panel", "benchmark", "eda", "ml", "all"],
        help="etapa a ejecutar",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="ignora la caché y recalcula desde cero",
    )
    args = parser.parse_args(argv)

    if args.etapa in ("ingest", "all"):
        etapa_ingest(args.force)
    if args.etapa in ("panel", "all"):
        etapa_panel(args.force)
    if args.etapa in ("benchmark", "all"):
        etapa_benchmark(args.force)
    if args.etapa in ("eda", "all"):
        etapa_eda(args.force)
    if args.etapa in ("ml", "all"):
        etapa_ml(args.force)

    print("\nListo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
