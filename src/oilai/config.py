"""Rutas y constantes compartidas del proyecto."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

for _d in (DATA_RAW, DATA_PROCESSED, REPORTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# Fuente: Agencia Nacional de Hidrocarburos (ANH) vía Datos Abiertos Colombia.
# "Produccion Fiscalizada Crudo Consolidada" — producción mensual por campo.
SOCRATA_DOMAIN = "www.datos.gov.co"
SOCRATA_DATASET = "fdvb-hsrf"
SOCRATA_URL = f"https://{SOCRATA_DOMAIN}/resource/{SOCRATA_DATASET}.json"

RAW_PARQUET = DATA_RAW / "anh_produccion_cruda.parquet"
PANEL_PARQUET = DATA_PROCESSED / "panel_campos_mensual.parquet"
