"""Pruebas del ajuste de curvas de declinación."""

import numpy as np
import pytest

from oilai.models.arps import arps, arps_multiple, fit, fit_mejor


def test_exponencial_es_el_limite_continuo_de_b_cero():
    """b→0 debe converger a qi·exp(-Di·t) sin discontinuidad numérica."""
    t = np.arange(24.0)
    exacta = 1000 * np.exp(-0.05 * t)

    assert np.allclose(arps(t, 1000, 0.05, 0.0), exacta)
    # Un b diminuto debe dar prácticamente lo mismo que el caso exacto.
    assert np.allclose(arps(t, 1000, 0.05, 1e-7), exacta, rtol=1e-3)


def test_armonica_coincide_con_la_formula_cerrada():
    t = np.arange(36.0)
    esperado = 5000 / (1 + 0.02 * t)
    assert np.allclose(arps(t, 5000, 0.02, 1.0), esperado)


def test_la_produccion_siempre_declina():
    t = np.arange(120.0)
    for b in (0.0, 0.3, 0.7, 1.0):
        q = arps(t, 10_000, 0.04, b)
        assert np.all(np.diff(q) <= 0), f"b={b} produce un tramo creciente"
        assert np.all(q > 0)


@pytest.mark.parametrize("b_real", [0.0, 0.5, 1.0])
def test_recupera_el_caudal_inicial_con_datos_limpios(b_real):
    """Sin ruido, el ajuste debe clavar qi; Di y b están correlacionados."""
    t = np.arange(60.0)
    q = arps(t, 12_000, 0.03, b_real)

    ajuste = fit_mejor(t, q)

    assert ajuste.exito
    assert ajuste.qi == pytest.approx(12_000, rel=0.02)
    # El ajuste debe reproducir la curva aunque (Di, b) difieran del par real.
    assert np.allclose(ajuste.predict(t), q, rtol=0.05)


def test_es_robusto_a_ruido_multiplicativo():
    rng = np.random.default_rng(42)
    t = np.arange(72.0)
    q = arps(t, 20_000, 0.025, 0.6) * (1 + rng.normal(0, 0.05, 72))

    ajuste = fit_mejor(t, q)

    assert ajuste.exito
    assert ajuste.qi == pytest.approx(20_000, rel=0.15)
    assert 0.0 <= ajuste.b <= 1.0


def test_series_demasiado_corta_no_lanza_excepcion():
    """Con <4 puntos no hay grados de libertad: debe degradarse, no romperse."""
    ajuste = fit(np.array([0.0, 1.0]), np.array([100.0, 90.0]))

    assert not ajuste.exito
    assert np.isfinite(ajuste.qi)


def test_ignora_los_ceros_y_negativos():
    """Meses de campo cerrado (q=0) no deben contaminar el ajuste."""
    t = np.arange(20.0)
    q = arps(t, 1000, 0.03, 0.5)
    q_con_paros = q.copy()
    q_con_paros[[5, 6]] = 0.0

    ajuste = fit(t, q_con_paros)

    assert ajuste.n_puntos == 18
    assert ajuste.qi == pytest.approx(1000, rel=0.1)


def test_la_etiqueta_de_tipo_corresponde_al_valor_de_b():
    t = np.arange(40.0)
    assert fit(t, arps(t, 900, 0.03, 0.0), b_fijo=0.0).tipo == "exponencial"
    assert fit(t, arps(t, 900, 0.03, 1.0), b_fijo=1.0).tipo == "armónica"


@pytest.mark.parametrize("b", [0.0, 1e-9, 0.25, 0.6, 1.0])
def test_la_version_vectorizada_coincide_con_la_escalar(b):
    """`arps_multiple` evalua miles de ajustes; debe dar exactamente lo mismo."""
    t = np.arange(0.0, 48.0)
    n = len(t)

    escalar = arps(t, 1000.0, 0.03, b)
    vectorial = arps_multiple(t, np.full(n, 1000.0), np.full(n, 0.03), np.full(n, b))

    assert np.allclose(escalar, vectorial)


def test_la_version_vectorizada_mezcla_distintos_regimenes():
    """Cada fila puede tener su propio b, que es el caso real de uso."""
    t = np.array([12.0, 12.0, 12.0])
    qi = np.array([1000.0, 1000.0, 1000.0])
    di = np.array([0.03, 0.03, 0.03])
    b = np.array([0.0, 0.5, 1.0])

    salida = arps_multiple(t, qi, di, b)

    for j, b_j in enumerate(b):
        assert salida[j] == pytest.approx(arps(12.0, 1000.0, 0.03, b_j))
