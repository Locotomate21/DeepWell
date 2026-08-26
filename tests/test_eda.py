"""Pruebas de la caracterización de campos y los agregados nacionales."""

import numpy as np
import pandas as pd
import pytest

from oilai.eda import (
    UMBRAL_COBERTURA,
    cobertura_mensual,
    concentracion,
    declinacion_anual_efectiva,
    indice_hhi,
    ranking,
    serie_nacional,
    volatilidad_log,
)


def test_declinacion_anual_de_di_cero_es_cero():
    assert declinacion_anual_efectiva(0.0) == pytest.approx(0.0)


def test_declinacion_anual_coincide_con_la_formula():
    # Di = 0.05/mes durante 12 meses -> 1 - exp(-0.6) = 45.1 %
    assert declinacion_anual_efectiva(0.05) == pytest.approx(45.12, abs=0.01)


def test_declinacion_anual_crece_con_di():
    valores = [declinacion_anual_efectiva(di) for di in (0.01, 0.03, 0.10)]
    assert valores == sorted(valores)


def test_volatilidad_de_serie_constante_es_cero():
    """Una serie plana no tiene variabilidad relativa."""
    assert volatilidad_log(np.full(12, 500.0)) == pytest.approx(0.0)


def test_volatilidad_es_invariante_a_la_escala():
    """Es la propiedad que la hace comparable entre campos de tamaños distintos."""
    base = np.array([100.0, 120.0, 90.0, 110.0, 105.0])

    assert volatilidad_log(base) == pytest.approx(volatilidad_log(base * 1000))


def test_volatilidad_ignora_meses_sin_produccion():
    """log(0) es indefinido: los ceros se descartan en vez de propagar -inf."""
    v = volatilidad_log(np.array([100.0, 0.0, 110.0, 105.0, 98.0]))

    assert np.isfinite(v)


def test_volatilidad_de_serie_muy_corta_es_nan():
    assert np.isnan(volatilidad_log(np.array([100.0, 110.0])))


def _panel_sintetico(meses: int = 24, campos: int = 10) -> pd.DataFrame:
    fechas = pd.date_range("2020-01-01", periods=meses, freq="MS")
    filas = [
        {"campo": f"C{i}", "fecha": f, "bpd": 100.0 * (i + 1), "operadora": f"OP{i % 3}",
         "departamento": "META"}
        for f in fechas
        for i in range(campos)
    ]
    return pd.DataFrame(filas)


def test_cobertura_detecta_un_mes_con_publicacion_incompleta():
    """Reproduce el caso real de noviembre de 2025 en los datos de la ANH."""
    panel = _panel_sintetico(meses=24, campos=10)
    # Un solo mes en el que únicamente reporta un campo de los diez.
    mes_malo = pd.Timestamp("2021-06-01")
    panel = panel[~((panel.fecha == mes_malo) & (panel.campo != "C0"))]

    cob = cobertura_mensual(panel)
    fila = cob[cob.fecha == mes_malo].iloc[0]

    assert fila.campos_activos == 1
    assert fila.cobertura < UMBRAL_COBERTURA
    assert not fila.reporte_completo
    # Ningún otro mes debe marcarse.
    assert (~cob.reporte_completo).sum() == 1


def test_serie_nacional_excluye_los_meses_incompletos():
    panel = _panel_sintetico(meses=24, campos=10)
    mes_malo = pd.Timestamp("2021-06-01")
    panel = panel[~((panel.fecha == mes_malo) & (panel.campo != "C0"))]

    limpia = serie_nacional(panel)
    completa = serie_nacional(panel, solo_completos=False)

    assert mes_malo not in set(limpia.fecha)
    assert mes_malo in set(completa.fecha)
    assert len(completa) == len(limpia) + 1


def test_serie_nacional_de_panel_sano_no_pierde_meses():
    panel = _panel_sintetico(meses=24, campos=10)

    assert len(serie_nacional(panel)) == 24


def test_concentracion_llega_al_cien_por_ciento():
    panel = _panel_sintetico(meses=12, campos=10)
    panel["fecha"] = pd.date_range("2025-01-01", periods=12, freq="MS").repeat(10)[
        : len(panel)
    ]

    conc = concentracion(panel, anio=2025)

    assert conc.pct_produccion.is_monotonic_increasing
    assert conc.pct_produccion.max() <= 100.0 + 1e-6
    assert conc.attrs["campos_activos"] == 10


def test_hhi_de_monopolio_es_diez_mil():
    """Un solo operador con el 100 % da el máximo del índice."""
    fechas = pd.date_range("2025-01-01", periods=6, freq="MS")
    panel = pd.DataFrame(
        {"campo": "A", "fecha": fechas, "bpd": 100.0, "operadora": "UNICA",
         "departamento": "META"}
    )

    assert indice_hhi(panel, anio=2025) == pytest.approx(10_000.0)


def test_hhi_de_competencia_perfecta_es_bajo():
    fechas = pd.date_range("2025-01-01", periods=6, freq="MS")
    filas = [
        {"campo": f"C{i}", "fecha": f, "bpd": 100.0, "operadora": f"OP{i}",
         "departamento": "META"}
        for f in fechas
        for i in range(10)
    ]

    # Diez operadores iguales -> 10 * 10^2 = 1000
    assert indice_hhi(pd.DataFrame(filas), anio=2025) == pytest.approx(1000.0)


def test_ranking_ordena_y_suma_cien():
    fechas = pd.date_range("2025-01-01", periods=6, freq="MS")
    filas = [
        {"campo": f"C{i}", "fecha": f, "bpd": float(10 ** (i + 1)),
         "operadora": f"OP{i}", "departamento": "META"}
        for f in fechas
        for i in range(3)
    ]

    r = ranking("operadora", pd.DataFrame(filas), anio=2025, top=3)

    assert list(r.operadora) == ["OP2", "OP1", "OP0"]
    assert r.pct.sum() == pytest.approx(100.0)
