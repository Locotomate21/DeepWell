"""Descarga de la producción fiscalizada de crudo publicada por la ANH.

El recurso se expone en Datos Abiertos Colombia mediante la API Socrata, que
pagina los resultados. Descargamos el histórico completo una sola vez y lo
guardamos en Parquet para que el resto del pipeline sea reproducible y no
dependa de la disponibilidad del servicio.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

from .config import RAW_PARQUET, SOCRATA_URL

PAGE_SIZE = 10_000
TIMEOUT = 120


def _fetch_page(offset: int, page_size: int = PAGE_SIZE) -> list[dict]:
    params = {
        "$limit": page_size,
        "$offset": offset,
        "$order": ":id",  # orden estable: sin esto la paginación puede repetir filas
    }
    response = requests.get(SOCRATA_URL, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def download(force: bool = False) -> pd.DataFrame:
    """Devuelve el histórico completo, usando el Parquet cacheado si existe."""
    if RAW_PARQUET.exists() and not force:
        return pd.read_parquet(RAW_PARQUET)

    pages: list[dict] = []
    offset = 0
    while True:
        page = _fetch_page(offset)
        if not page:
            break
        pages.extend(page)
        print(f"  descargados {len(pages):,} registros...", flush=True)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(0.3)  # cortesía con el servicio público

    frame = pd.DataFrame(pages)
    frame.to_parquet(RAW_PARQUET, index=False)
    return frame


if __name__ == "__main__":
    df = download(force=True)
    print(f"\n{len(df):,} registros guardados en {RAW_PARQUET}")
    print(f"columnas: {list(df.columns)}")
