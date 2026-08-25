"""Pruebas de las líneas base del benchmark."""

import numpy as np
import pytest

from oilai.models.arps import arps
from oilai.models.baselines import ArpsModel, Drift, MediaMovil, Naive


@pytest.fixture
def serie():
    t = np.arange(48.0)
    q = arps(t, 8000, 0.03, 0.5)
    return t, q


def test_naive_repite_el_ultimo_valor(serie):
    t, q = serie
    pred = Naive().fit(t, q).predict(np.arange(48.0, 54.0))

    assert np.allclose(pred, q[-1])
    assert len(pred) == 6


def test_media_movil_promedia_solo_la_ventana():
    t = np.arange(10.0)
    q = np.array([100.0] * 7 + [10.0, 20.0, 30.0])

    pred = MediaMovil(3).fit(t, q).predict(np.arange(10.0, 13.0))

    assert np.allclose(pred, 20.0)  # media de 10, 20, 30


def test_drift_extrapola_la_tendencia():
    t = np.arange(5.0)
    q = np.array([100.0, 90.0, 80.0, 70.0, 60.0])  # pendiente -10/mes

    pred = Drift().fit(t, q).predict(np.array([5.0, 6.0]))

    assert np.allclose(pred, [50.0, 40.0])


def test_drift_nunca_predice_produccion_negativa():
    """Extrapolar una caída fuerte no debe dar barriles negativos."""
    t = np.arange(5.0)
    q = np.array([100.0, 80.0, 60.0, 40.0, 20.0])

    pred = Drift().fit(t, q).predict(np.arange(5.0, 20.0))

    assert np.all(pred >= 0)


def test_arps_model_solo_usa_la_ventana_indicada():
    """Con ventana=12 el ajuste debe ignorar la historia anterior."""
    t = np.arange(60.0)
    # Régimen viejo muy distinto del reciente.
    q = np.concatenate([np.full(48, 5000.0), arps(np.arange(12.0), 900, 0.05, 0.5)])

    modelo = ArpsModel(ventana=12).fit(t, q)
    pred = modelo.predict(np.array([60.0]))

    # Debe seguir el régimen reciente (~900 y bajando), no la meseta de 5000.
    assert pred[0] < 1500


def test_arps_model_devuelve_valores_finitos_y_no_negativos(serie):
    t, q = serie
    pred = ArpsModel(ventana=36).fit(t, q).predict(np.arange(48.0, 72.0))

    assert np.all(np.isfinite(pred))
    assert np.all(pred >= 0)


def test_todas_las_lineas_base_comparten_la_interfaz(serie):
    t, q = serie
    t_fut = np.arange(48.0, 54.0)

    for modelo in (Naive(), MediaMovil(6), Drift(), ArpsModel(24)):
        pred = modelo.fit(t, q).predict(t_fut)
        assert len(pred) == len(t_fut), f"{modelo.nombre} devolvió otra longitud"
        assert np.all(np.isfinite(pred)), f"{modelo.nombre} devolvió no finitos"
        assert isinstance(modelo.nombre, str)
