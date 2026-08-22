"""Ejecuta el benchmark walk-forward sobre todos los campos elegibles."""

from __future__ import annotations

import pandas as pd

from .clean import build_panel
from .config import REPORTS
from .evaluate import backtest_campo, resumen
from .models.baselines import ArpsModel, Drift, MediaMovil, Naive

HORIZONTE = 12
MIN_TRAIN = 24
N_ORIGENES = 3


def modelos_base() -> list:
    return [
        Naive(),
        MediaMovil(3),
        MediaMovil(6),
        Drift(),
        ArpsModel(ventana=24),
        ArpsModel(ventana=36),
        ArpsModel(ventana=None),
    ]


def main() -> pd.DataFrame:
    panel = build_panel()

    filas = []
    campos = panel.campo.unique()
    for i, campo in enumerate(campos, 1):
        g = panel[panel.campo == campo].sort_values("fecha")
        res = backtest_campo(
            t=g.meses_desde_inicio.to_numpy(float),
            q=g.bpd.to_numpy(float),
            fechas=g.fecha.reset_index(drop=True),
            modelos=modelos_base(),
            horizonte=HORIZONTE,
            n_origenes=N_ORIGENES,
            min_train=MIN_TRAIN,
        )
        for v in res:
            v.campo = campo
        filas.extend(res)
        if i % 100 == 0:
            print(f"  {i}/{len(campos)} campos procesados", flush=True)

    df = pd.DataFrame([v.__dict__ for v in filas])
    df.to_parquet(REPORTS / "backtest_base.parquet", index=False)
    return df


if __name__ == "__main__":
    df = main()
    print(f"\n{len(df):,} predicciones evaluadas sobre {df.campo.nunique()} campos\n")
    print("=== RANKING GLOBAL (todos los horizontes) ===")
    print(resumen(df).round(3).to_string())
