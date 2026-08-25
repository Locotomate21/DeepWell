"""Punto de entrada del proyecto: reproduce la Fase 1 completa.

    oilai ingest      descarga el histórico de la ANH
    oilai panel       construye el panel mensual limpio
    oilai benchmark   corre el backtesting walk-forward de las líneas base
    oilai all         las tres, en orden

Cada etapa cachea su salida, así que volver a correr `all` es barato: solo
recalcula lo que falte. Con `--force` se rehace todo desde cero.
"""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from .clean import build_panel
from .config import PANEL_PARQUET, RAW_PARQUET, REPORTS
from .evaluate import resumen
from .ingest import download


def _titulo(texto: str) -> None:
    print(f"\n{'=' * 62}\n{texto}\n{'=' * 62}", flush=True)


def etapa_ingest(force: bool) -> pd.DataFrame:
    _titulo("ETAPA 1/3 · Descarga del histórico ANH")
    t0 = time.perf_counter()
    df = download(force=force)
    origen = "descargado" if force or not RAW_PARQUET.exists() else "caché"
    print(f"{len(df):,} registros ({origen}) en {time.perf_counter() - t0:.1f}s")
    print(f"-> {RAW_PARQUET}")
    return df


def etapa_panel(force: bool) -> pd.DataFrame:
    _titulo("ETAPA 2/3 · Construcción del panel mensual")
    t0 = time.perf_counter()
    panel = build_panel(force=force)
    print(f"{len(panel):,} filas · {panel.campo.nunique()} campos · "
          f"{panel.operadora.nunique()} operadoras")
    print(f"cobertura: {panel.fecha.min():%Y-%m} a {panel.fecha.max():%Y-%m}")
    print(f"completado en {time.perf_counter() - t0:.1f}s")
    print(f"-> {PANEL_PARQUET}")
    return panel


def etapa_benchmark(force: bool) -> pd.DataFrame:
    _titulo("ETAPA 3/3 · Backtesting walk-forward de líneas base")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="oilai",
        description="Pipeline de pronóstico de producción petrolera (Fase 1).",
    )
    parser.add_argument(
        "etapa",
        choices=["ingest", "panel", "benchmark", "all"],
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

    print("\nListo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
