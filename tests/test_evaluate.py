"""Pruebas del protocolo de evaluación.

La prueba crítica es `test_el_backtest_no_ve_el_futuro`: si el walk-forward
filtrara datos posteriores al origen, todas las métricas del proyecto serían
inválidas.
"""

import numpy as np
import pandas as pd
import pytest

from oilai.evaluate import (
    backtest_campo,
    escala_naive,
    mae,
    resumen,
    rmse,
    sesgo,
    smape,
)


def test_metricas_basicas():
    y = np.array([10.0, 20.0, 30.0])
    yhat = np.array([12.0, 18.0, 33.0])

    assert mae(y, yhat) == pytest.approx(7 / 3)
    assert rmse(y, yhat) == pytest.approx(np.sqrt((4 + 4 + 9) / 3))
    # errores con signo: +2, -2, +3 -> media 1.0 (sobreestima)
    assert sesgo(y, yhat) == pytest.approx(1.0)


def test_prediccion_perfecta_da_error_cero():
    y = np.array([5.0, 7.0, 9.0])
    assert mae(y, y) == 0.0
    assert smape(y, y) == 0.0


def test_smape_no_explota_cuando_el_valor_real_es_cero():
    """Un campo cerrado (q=0) haría MAPE infinito; sMAPE debe seguir acotado."""
    y = np.array([0.0, 0.0, 100.0])
    yhat = np.array([0.0, 50.0, 100.0])

    valor = smape(y, yhat)

    assert np.isfinite(valor)
    assert 0 <= valor <= 200


def test_escala_naive_es_el_mae_de_un_paso():
    q = np.array([100.0, 110.0, 105.0])
    # diferencias absolutas: 10, 5 -> media 7.5
    assert escala_naive(q) == pytest.approx(7.5)


def test_escala_naive_de_serie_constante_es_nan():
    """Serie plana => denominador 0; debe ser NaN y no dividir por cero."""
    assert np.isnan(escala_naive(np.array([50.0, 50.0, 50.0])))


class _EspiaDeEntrenamiento:
    """Modelo falso que registra cuántos puntos vio en cada entrenamiento."""

    nombre = "Espia"

    def __init__(self):
        self.vistos: list[np.ndarray] = []

    def fit(self, t, q):
        self.vistos.append(np.asarray(t).copy())
        self._ultimo = q[-1]
        return self

    def predict(self, t_fut):
        return np.full(len(t_fut), self._ultimo)


def test_el_backtest_no_ve_el_futuro():
    """El entrenamiento nunca debe incluir instantes >= al primero evaluado."""
    n = 80
    t = np.arange(float(n))
    q = 1000 * np.exp(-0.02 * t)
    fechas = pd.Series(pd.date_range("2015-01-01", periods=n, freq="MS"))

    espia = _EspiaDeEntrenamiento()
    ventanas = backtest_campo(
        t, q, fechas, [espia], horizonte=12, n_origenes=3, min_train=24
    )

    assert ventanas, "el backtest no produjo evaluaciones"

    # Para cada entrenamiento, el t máximo visto debe ser anterior al horizonte.
    origenes = sorted({len(v) for v in espia.vistos})
    for corte in origenes:
        vistos = [v for v in espia.vistos if len(v) == corte][0]
        assert vistos.max() == corte - 1
        assert vistos.min() == 0


def test_horizontes_van_de_uno_a_h():
    n = 60
    t = np.arange(float(n))
    q = np.linspace(500, 200, n)
    fechas = pd.Series(pd.date_range("2016-01-01", periods=n, freq="MS"))

    from oilai.models.baselines import Naive

    ventanas = backtest_campo(
        t, q, fechas, [Naive()], horizonte=6, n_origenes=2, min_train=24
    )

    hs = {v.h for v in ventanas}
    assert hs == {1, 2, 3, 4, 5, 6}


def test_serie_corta_no_genera_evaluaciones():
    """Sin historia suficiente para train + horizonte, se omite el campo."""
    t = np.arange(10.0)
    q = np.full(10, 100.0)
    fechas = pd.Series(pd.date_range("2020-01-01", periods=10, freq="MS"))

    from oilai.models.baselines import Naive

    assert backtest_campo(t, q, fechas, [Naive()], min_train=24) == []


def test_resumen_ordena_por_mase_ascendente():
    df = pd.DataFrame(
        {
            "modelo": ["bueno"] * 3 + ["malo"] * 3,
            "y": [100.0, 100.0, 100.0] * 2,
            "yhat": [101.0, 99.0, 100.0, 150.0, 50.0, 130.0],
            "escala": [10.0] * 6,
        }
    )

    tabla = resumen(df)

    assert list(tabla.index) == ["bueno", "malo"]
    assert tabla.loc["bueno", "MASE"] < tabla.loc["malo", "MASE"]
