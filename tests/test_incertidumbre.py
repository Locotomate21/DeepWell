"""Pruebas de los intervalos de predicción."""

import numpy as np
import pandas as pd
import pytest

from oilai.incertidumbre import (
    NIVEL,
    _alfas,
    anchura_relativa,
    aplicar_conformal,
    calibrar_conformal,
    cobertura,
    resumen_intervalos,
    winkler,
)


def test_los_alfas_reparten_la_cola_por_igual():
    baja, alta = _alfas(0.80)

    assert baja == pytest.approx(0.10)
    assert alta == pytest.approx(0.90)


def test_un_nivel_mas_exigente_deja_colas_mas_pequenas():
    baja_80, _ = _alfas(0.80)
    baja_95, _ = _alfas(0.95)

    assert baja_95 < baja_80


# --- Calibración conformal -------------------------------------------------


def _residuos(n: int = 500, h_valores=(1, 6, 12), escala=0.1) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    filas = []
    for h in h_valores:
        # La dispersión crece con el horizonte, como en los datos reales.
        filas.append(
            pd.DataFrame(
                {"h": h, "residuo": rng.normal(0, escala * h, n)}
            )
        )
    return pd.concat(filas, ignore_index=True)


def test_los_margenes_se_estiman_por_horizonte():
    margenes = calibrar_conformal(_residuos(), NIVEL)

    assert set(margenes) == {1, 6, 12}
    # El intervalo debe ensancharse con el horizonte.
    ancho = {h: alto - bajo for h, (bajo, alto) in margenes.items()}
    assert ancho[1] < ancho[6] < ancho[12]


def test_los_margenes_recuperan_los_cuantiles_conocidos():
    """Con residuos normales, el margen debe acercarse a los cuantiles teóricos."""
    rng = np.random.default_rng(1)
    residuos = pd.DataFrame({"h": 1, "residuo": rng.normal(0, 1.0, 20_000)})

    bajo, alto = calibrar_conformal(residuos, 0.80)[1]

    assert bajo == pytest.approx(-1.2816, abs=0.05)
    assert alto == pytest.approx(1.2816, abs=0.05)


def test_los_margenes_son_asimetricos_si_el_error_lo_es():
    """Un modelo que subestima necesita más margen hacia arriba."""
    rng = np.random.default_rng(2)
    # Residuos sesgados a positivo: el real supera al pronóstico.
    residuos = pd.DataFrame({"h": 1, "residuo": rng.normal(0.5, 0.2, 5000)})

    bajo, alto = calibrar_conformal(residuos, NIVEL)[1]

    assert alto > abs(bajo)


def test_un_horizonte_con_pocos_residuos_se_omite():
    residuos = pd.concat(
        [
            pd.DataFrame({"h": 1, "residuo": np.zeros(200)}),
            pd.DataFrame({"h": 9, "residuo": np.zeros(5)}),
        ]
    )

    margenes = calibrar_conformal(residuos)

    assert 1 in margenes and 9 not in margenes


def test_los_residuos_no_finitos_se_descartan():
    residuos = pd.DataFrame(
        {"h": 1, "residuo": np.r_[np.random.default_rng(0).normal(0, 1, 500),
                                  [np.inf, -np.inf, np.nan]]}
    )

    bajo, alto = calibrar_conformal(residuos)[1]

    assert np.isfinite(bajo) and np.isfinite(alto)


# --- Aplicación ------------------------------------------------------------


def test_el_intervalo_contiene_a_la_prediccion_puntual():
    margenes = {1: (-0.2, 0.3)}
    pred_log = np.array([0.0, -0.1])
    ancla = np.array([1000.0, 500.0])

    lo, hi = aplicar_conformal(pred_log, ancla, np.array([1, 1]), margenes)

    punto = ancla * np.exp(pred_log)
    assert np.all(lo <= punto) and np.all(punto <= hi)


def test_el_intervalo_es_asimetrico_en_barriles():
    """En escala logarítmica es simétrico; al volver a bpd no debe serlo."""
    margenes = {1: (-0.3, 0.3)}
    lo, hi = aplicar_conformal(np.array([0.0]), np.array([1000.0]),
                               np.array([1]), margenes)

    punto = 1000.0
    assert (hi[0] - punto) > (punto - lo[0])


def test_los_limites_nunca_son_negativos():
    margenes = {1: (-3.0, 1.5)}
    lo, hi = aplicar_conformal(np.array([-2.5]), np.array([10.0]),
                               np.array([1]), margenes)

    assert lo[0] >= 0 and hi[0] >= 0


def test_un_horizonte_no_calibrado_usa_el_margen_mas_ancho():
    """Ante la duda, el intervalo debe ampliarse, nunca estrecharse."""
    margenes = {1: (-0.1, 0.1), 12: (-0.5, 0.6)}

    lo_conocido, hi_conocido = aplicar_conformal(
        np.array([0.0]), np.array([1000.0]), np.array([1]), margenes
    )
    lo_desconocido, hi_desconocido = aplicar_conformal(
        np.array([0.0]), np.array([1000.0]), np.array([7]), margenes
    )

    assert (hi_desconocido - lo_desconocido) > (hi_conocido - lo_conocido)


def test_sin_margenes_se_falla_de_forma_explicita():
    with pytest.raises(ValueError):
        aplicar_conformal(np.array([0.0]), np.array([1.0]), np.array([1]), {})


def test_la_calibracion_conformal_alcanza_la_cobertura_nominal():
    """Prueba de extremo a extremo sobre datos intercambiables.

    Es la propiedad que justifica el método: calibrando sobre una muestra y
    aplicando a otra de la misma distribución, la cobertura debe salir cerca del
    nivel nominal sin haberla ajustado.
    """
    rng = np.random.default_rng(11)
    ancla = 1000.0

    calibracion = pd.DataFrame({"h": 1, "residuo": rng.normal(0, 0.25, 4000)})
    margenes = calibrar_conformal(calibracion, 0.80)

    n = 4000
    reales = ancla * np.exp(rng.normal(0, 0.25, n))
    lo, hi = aplicar_conformal(
        np.zeros(n), np.full(n, ancla), np.ones(n, dtype=int), margenes
    )

    assert cobertura(reales, lo, hi) == pytest.approx(80.0, abs=2.0)


# --- Métricas --------------------------------------------------------------


def test_cobertura_cuenta_los_valores_dentro():
    y = np.array([5.0, 15.0, 25.0, 35.0])
    lo = np.array([0.0, 10.0, 30.0, 30.0])
    hi = np.array([10.0, 20.0, 40.0, 32.0])

    assert cobertura(y, lo, hi) == pytest.approx(50.0)


def test_cobertura_incluye_los_bordes():
    y = np.array([10.0, 20.0])
    assert cobertura(y, np.array([10.0, 15.0]), np.array([15.0, 20.0])) == 100.0


def test_anchura_relativa_se_expresa_en_multiplos_del_valor():
    y = np.array([100.0, 200.0])
    lo = np.array([50.0, 100.0])
    hi = np.array([150.0, 300.0])

    # anchuras 100 y 200 sobre valores 100 y 200 -> 1.0 en ambos casos
    assert anchura_relativa(y, lo, hi) == pytest.approx(1.0)


def test_winkler_penaliza_quedarse_fuera():
    y = np.array([100.0])
    dentro = winkler(y, np.array([90.0]), np.array([110.0]), 0.80)
    fuera = winkler(y, np.array([120.0]), np.array([140.0]), 0.80)

    assert fuera > dentro


def test_winkler_penaliza_la_anchura_excesiva():
    """Un intervalo enorme cubre todo pero no debe salir bien puntuado."""
    y = np.array([100.0] * 50)
    ajustado = winkler(y, np.full(50, 90.0), np.full(50, 110.0), 0.80)
    enorme = winkler(y, np.zeros(50), np.full(50, 10_000.0), 0.80)

    assert enorme > ajustado


def test_winkler_es_minimo_para_el_intervalo_justo():
    """Regla de puntuación propia: no se puede mejorar haciendo trampa."""
    rng = np.random.default_rng(3)
    y = rng.normal(100, 10, 20_000)

    # El intervalo central del 80 % de la propia distribución.
    justo = winkler(y, np.full_like(y, 100 - 1.2816 * 10),
                    np.full_like(y, 100 + 1.2816 * 10), 0.80)
    estrecho = winkler(y, np.full_like(y, 95.0), np.full_like(y, 105.0), 0.80)
    ancho = winkler(y, np.full_like(y, 50.0), np.full_like(y, 150.0), 0.80)

    assert justo < estrecho
    assert justo < ancho


def test_el_resumen_ordena_por_winkler():
    df = pd.DataFrame(
        {
            "metodo": ["bueno"] * 3 + ["malo"] * 3,
            "y": [100.0] * 6,
            "lo": [90.0, 90.0, 90.0, 0.0, 0.0, 0.0],
            "hi": [110.0, 110.0, 110.0, 9000.0, 9000.0, 9000.0],
        }
    )

    tabla = resumen_intervalos(df)

    assert list(tabla.index) == ["bueno", "malo"]
    assert "desvio_pp" in tabla.columns
    assert tabla.loc["bueno", "cobertura_%"] == 100.0
