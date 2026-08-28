"""Pruebas de la detección y validación de anomalías."""

import numpy as np
import pandas as pd
import pytest

from oilai.anomalias import (
    detectar,
    evolucion_posterior,
    tasa_por_mes,
    validar,
)


def _predicciones() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "campo": ["A", "B", "C", "D"],
            "fecha_objetivo": pd.Timestamp("2024-06-01"),
            "y": [50.0, 100.0, 250.0, 100.0],
            "lo": [80.0, 90.0, 90.0, 100.0],
            "hi": [120.0, 110.0, 110.0, 100.0],
        }
    )


def test_detecta_caidas_y_repuntes_por_separado():
    d = detectar(_predicciones())

    # A cae por debajo, C sube por encima, B queda dentro.
    assert d.set_index("campo").anomalia_baja.to_dict() == {
        "A": True, "B": False, "C": False, "D": False
    }
    assert d.set_index("campo").anomalia_alta.to_dict() == {
        "A": False, "B": False, "C": True, "D": False
    }


def test_la_anomalia_es_la_union_de_ambas_direcciones():
    d = detectar(_predicciones())

    assert (d.anomalia == (d.anomalia_baja | d.anomalia_alta)).all()
    assert int(d.anomalia.sum()) == 2


def test_la_severidad_mide_anchuras_de_intervalo():
    """A queda 30 bpd por debajo de un intervalo de 40 de ancho -> 0.75."""
    d = detectar(_predicciones()).set_index("campo")

    assert d.loc["A", "severidad"] == pytest.approx(0.75)
    assert d.loc["B", "severidad"] == pytest.approx(0.0)


def test_un_intervalo_degenerado_no_produce_severidad_infinita():
    """El campo D tiene lo == hi: dividir por el ancho daría infinito."""
    d = detectar(_predicciones()).set_index("campo")

    assert not np.isfinite(d.loc["D", "severidad"]) or d.loc["D", "severidad"] == 0


def test_una_prediccion_perfecta_no_genera_alertas():
    df = pd.DataFrame(
        {
            "campo": ["A"] * 5,
            "fecha_objetivo": pd.date_range("2024-01-01", periods=5, freq="MS"),
            "y": [100.0] * 5,
            "lo": [90.0] * 5,
            "hi": [110.0] * 5,
        }
    )

    assert not detectar(df).anomalia.any()


def test_la_tasa_mensual_resume_por_fecha():
    df = pd.DataFrame(
        {
            "campo": ["A", "B", "A", "B"],
            "fecha_objetivo": [pd.Timestamp("2024-01-01")] * 2
            + [pd.Timestamp("2024-02-01")] * 2,
            "y": [50.0, 100.0, 100.0, 100.0],
            "lo": [80.0, 90.0, 90.0, 90.0],
            "hi": [120.0, 110.0, 110.0, 110.0],
        }
    )

    tasa = tasa_por_mes(detectar(df)).set_index("fecha_objetivo")

    assert tasa.loc[pd.Timestamp("2024-01-01"), "pct_bajas"] == pytest.approx(50.0)
    assert tasa.loc[pd.Timestamp("2024-02-01"), "pct_bajas"] == pytest.approx(0.0)


# --- Validación ------------------------------------------------------------


def _panel_con_caida(caida_en: str = "2024-06-01", factor: float = 0.5) -> pd.DataFrame:
    fechas = pd.date_range("2023-01-01", periods=36, freq="MS")
    bpd = np.full(36, 1000.0)
    corte = pd.Timestamp(caida_en)
    bpd[fechas > corte] *= factor
    return pd.DataFrame({"campo": "A", "fecha": fechas, "bpd": bpd})


def test_la_evolucion_detecta_la_caida_posterior():
    panel = _panel_con_caida(factor=0.5)
    alertas = pd.DataFrame(
        {
            "campo": ["A"],
            "fecha_objetivo": [pd.Timestamp("2024-06-01")],
            "anomalia_baja": [True],
            "anomalia": [True],
        }
    )

    ev = evolucion_posterior(panel, alertas)

    assert len(ev) == 1
    assert ev.cociente.iloc[0] == pytest.approx(0.5, abs=0.01)


def test_sin_caida_el_cociente_es_uno():
    panel = _panel_con_caida(factor=1.0)
    alertas = pd.DataFrame(
        {
            "campo": ["A"],
            "fecha_objetivo": [pd.Timestamp("2024-06-01")],
            "anomalia_baja": [False],
            "anomalia": [False],
        }
    )

    ev = evolucion_posterior(panel, alertas)

    assert ev.cociente.iloc[0] == pytest.approx(1.0, abs=0.01)


def test_se_exige_historia_a_ambos_lados():
    """Sin meses suficientes antes o después, el cociente no significa nada."""
    panel = _panel_con_caida()
    alertas = pd.DataFrame(
        {
            "campo": ["A", "A"],
            # El primer y el último mes no tienen ventana completa.
            "fecha_objetivo": [pd.Timestamp("2023-01-01"), pd.Timestamp("2025-12-01")],
            "anomalia_baja": [True, True],
            "anomalia": [True, True],
        }
    )

    assert evolucion_posterior(panel, alertas).empty


def test_un_campo_desconocido_se_ignora():
    panel = _panel_con_caida()
    alertas = pd.DataFrame(
        {
            "campo": ["INEXISTENTE"],
            "fecha_objetivo": [pd.Timestamp("2024-06-01")],
            "anomalia_baja": [True],
            "anomalia": [True],
        }
    )

    assert evolucion_posterior(panel, alertas).empty


def test_la_validacion_compara_alertas_contra_meses_normales():
    rng = np.random.default_rng(5)
    evolucion = pd.DataFrame(
        {
            "campo": "A",
            "fecha": pd.Timestamp("2024-01-01"),
            "anomalia_baja": [True] * 100 + [False] * 100,
            "anomalia": [True] * 100 + [False] * 100,
            "cociente": np.r_[
                rng.normal(0.6, 0.05, 100),  # tras alerta, cae
                rng.normal(1.0, 0.05, 100),  # sin alerta, estable
            ],
        }
    )

    tabla = validar(evolucion)

    assert set(tabla.index) == {"alerta de caída", "sin alerta"}
    assert tabla.loc["alerta de caída", "cociente_medio"] < tabla.loc[
        "sin alerta", "cociente_medio"
    ]
    assert tabla.loc["alerta de caída", "pct_cae_mas_20"] > 90
    assert tabla.loc["sin alerta", "pct_cae_mas_20"] < 10


def test_la_validacion_no_falla_si_no_hay_alertas():
    evolucion = pd.DataFrame(
        {
            "campo": ["A"],
            "fecha": [pd.Timestamp("2024-01-01")],
            "anomalia_baja": [False],
            "anomalia": [False],
            "cociente": [1.0],
        }
    )

    tabla = validar(evolucion)

    assert list(tabla.index) == ["sin alerta"]
