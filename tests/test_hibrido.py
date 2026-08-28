"""Pruebas de los modelos híbridos de la Fase 4."""

import numpy as np
import pandas as pd
import pytest

from oilai.backtest_hibrido import (
    BASES_COMBINACION,
    ETIQUETAS_TAMANO,
    clase_tamano,
)
from oilai.features import OBJETIVO_MAX, OBJETIVO_MIN, construir_muestras
from oilai.models.arps import arps
from oilai.models.hibrido import (
    MIN_OBS_BUCKET,
    CombinacionPorRegimen,
    _malla_simplex,
    _mejores_pesos,
    agregar_variable_arps,
    ajustes_arps,
    prediccion_arps,
)


def _panel(campos: int = 4, meses: int = 72, inicio: str = "2016-01-01") -> pd.DataFrame:
    fechas = pd.date_range(inicio, periods=meses, freq="MS")
    trozos = []
    for i in range(campos):
        trozos.append(
            pd.DataFrame(
                {
                    "campo": f"C{i}",
                    "fecha": fechas,
                    "bpd": 500.0 * (i + 1) * 0.99 ** np.arange(meses),
                    "operadora": f"OP{i % 2}",
                    "departamento": "META",
                }
            )
        )
    return pd.concat(trozos, ignore_index=True)


# --- Malla de pesos --------------------------------------------------------


def test_la_malla_cubre_el_simplex():
    malla = _malla_simplex(3, paso=0.1)

    assert len(malla) == 66  # combinaciones de 3 pesos en pasos de 0.1
    for pesos in malla:
        assert sum(pesos) == pytest.approx(1.0)
        assert all(p >= 0 for p in pesos)


def test_la_malla_incluye_los_vertices():
    """Debe poder elegir un solo modelo si es el mejor."""
    malla = _malla_simplex(3)

    assert (1.0, 0.0, 0.0) in malla
    assert (0.0, 0.0, 1.0) in malla


def test_los_pesos_eligen_el_modelo_perfecto():
    real = np.array([10.0, 20.0, 30.0, 40.0])
    # Columna 1 exacta, las otras erróneas.
    matriz = np.c_[real + 50, real, real - 30]

    pesos = _mejores_pesos(matriz, real, _malla_simplex(3))

    assert pesos[1] == pytest.approx(1.0)


def test_la_mezcla_cancela_sesgos_opuestos():
    """Con un modelo que sobreestima y otro que subestima, la mezcla acierta.

    No se comprueban pesos concretos: bajo error absoluto el óptimo no es único
    —varias combinaciones anulan el sesgo por igual— y fijar unos pesos sería
    exigir algo que los datos no determinan. Lo que sí está determinado es que el
    error de la combinación baje a cero.
    """
    real = np.array([100.0] * 20)
    matriz = np.c_[real + 10, real - 10, real + 40]

    pesos = _mejores_pesos(matriz, real, _malla_simplex(3))

    assert np.mean(np.abs(real - matriz @ pesos)) == pytest.approx(0.0, abs=1e-9)
    # Y mejora a cualquiera de los modelos por separado.
    for j in range(matriz.shape[1]):
        assert np.mean(np.abs(real - matriz[:, j])) > 0


# --- Combinación por régimen ----------------------------------------------


def _ancho_sintetico(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    h = rng.integers(1, 13, n)
    y = rng.uniform(100, 1000, n)
    return pd.DataFrame(
        {
            "h": h,
            "clase": rng.choice(ETIQUETAS_TAMANO[:2], n),
            "y": y,
            "Naive": y + rng.normal(0, 20, n),
            "Arps-24m": y + rng.normal(0, 60, n),
            "ML-global": y + rng.normal(0, 30, n),
        }
    )


def test_las_predicciones_son_combinaciones_convexas():
    ancho = _ancho_sintetico()
    c = CombinacionPorRegimen(BASES_COMBINACION).fit(ancho)

    pred = c.predict(ancho)
    minimo = ancho[BASES_COMBINACION].min(axis=1).to_numpy()
    maximo = ancho[BASES_COMBINACION].max(axis=1).to_numpy()

    assert np.all(pred >= minimo - 1e-9)
    assert np.all(pred <= maximo + 1e-9)


def test_los_pesos_de_cada_bucket_suman_uno():
    ancho = _ancho_sintetico(1200)
    c = CombinacionPorRegimen(BASES_COMBINACION).fit(ancho)

    for pesos in list(c.pesos.values()) + list(c.pesos_horizonte.values()):
        assert pesos.sum() == pytest.approx(1.0)
    assert c.pesos_globales.sum() == pytest.approx(1.0)


def test_un_bucket_escaso_se_repliega_al_horizonte():
    ancho = _ancho_sintetico(1200)
    c = CombinacionPorRegimen(BASES_COMBINACION).fit(ancho)

    # Una clase nunca vista debe resolverse con los pesos del horizonte.
    pesos = c.pesos_de(3, "clase-inexistente")

    assert pesos is c.pesos_horizonte.get(3, c.pesos_globales)


def test_un_horizonte_no_visto_usa_los_pesos_globales():
    ancho = _ancho_sintetico(1200)
    c = CombinacionPorRegimen(BASES_COMBINACION).fit(ancho)

    assert np.array_equal(c.pesos_de(99, "otra"), c.pesos_globales)


def test_no_se_estiman_pesos_con_pocas_observaciones():
    """Con pocos datos por bucket los pesos serían ruido."""
    ancho = _ancho_sintetico(MIN_OBS_BUCKET // 2)
    c = CombinacionPorRegimen(BASES_COMBINACION).fit(ancho)

    assert c.pesos == {}
    assert c.pesos_globales is not None


def test_la_tabla_de_pesos_es_legible():
    ancho = _ancho_sintetico(2000)
    c = CombinacionPorRegimen(BASES_COMBINACION).fit(ancho)

    tabla = c.tabla_pesos()

    assert set(BASES_COMBINACION).issubset(tabla.columns)
    assert {"h", "clase"}.issubset(tabla.columns)
    assert np.allclose(tabla[BASES_COMBINACION].sum(axis=1), 1.0)


# --- Arps como variable ----------------------------------------------------


def test_los_ajustes_cubren_cada_origen_valido(tmp_path, monkeypatch):
    import oilai.models.hibrido as mod

    monkeypatch.setattr(mod, "AJUSTES_PARQUET", tmp_path / "a.parquet")
    panel = _panel(campos=2, meses=40)

    a = ajustes_arps(panel, min_historia=24, force=True)

    # 40 meses, 24 de historia mínima -> 17 orígenes por campo.
    assert len(a) == 2 * 17
    assert set(a.campo) == {"C0", "C1"}


def test_la_prediccion_de_arps_reproduce_la_curva():
    ajustes = pd.DataFrame(
        {
            "arps_qi": [1000.0],
            "arps_di": [0.02],
            "arps_b": [0.5],
            "arps_t_origen": [23.0],
        }
    )

    pred = prediccion_arps(ajustes, np.array([6]))

    assert pred[0] == pytest.approx(arps(29.0, 1000.0, 0.02, 0.5))


def test_la_variable_de_arps_esta_en_la_escala_del_objetivo(tmp_path, monkeypatch):
    import oilai.models.hibrido as mod

    monkeypatch.setattr(mod, "AJUSTES_PARQUET", tmp_path / "a.parquet")
    panel = _panel(campos=3, meses=60)

    muestras = construir_muestras(panel, range(1, 13), min_historia=24)
    con_arps = agregar_variable_arps(muestras, ajustes_arps(panel, force=True))

    assert "arps_rel" in con_arps.columns
    validos = con_arps.arps_rel.dropna()
    assert len(validos) > 0
    assert validos.between(OBJETIVO_MIN, OBJETIVO_MAX).all()


def test_la_variable_de_arps_no_pierde_filas(tmp_path, monkeypatch):
    import oilai.models.hibrido as mod

    monkeypatch.setattr(mod, "AJUSTES_PARQUET", tmp_path / "a.parquet")
    panel = _panel(campos=3, meses=60)
    muestras = construir_muestras(panel, range(1, 13), min_historia=24)

    con_arps = agregar_variable_arps(muestras, ajustes_arps(panel, force=True))

    assert len(con_arps) == len(muestras)


def test_arps_predice_bien_una_declinacion_exponencial_pura(tmp_path, monkeypatch):
    """Control de cordura: sobre la curva que Arps modela, debe acertar."""
    import oilai.models.hibrido as mod

    monkeypatch.setattr(mod, "AJUSTES_PARQUET", tmp_path / "a.parquet")
    panel = _panel(campos=1, meses=60)

    muestras = construir_muestras(panel, [12], min_historia=24)
    con_arps = agregar_variable_arps(muestras, ajustes_arps(panel, force=True))
    fila = con_arps.dropna(subset=["arps_rel"]).iloc[0]

    esperado = np.log(fila.bpd_real / fila.ancla_bpd)
    assert fila.arps_rel == pytest.approx(esperado, abs=0.05)


# --- Protocolo -------------------------------------------------------------


def test_filtrar_equivale_a_reconstruir():
    """Construir las muestras una vez y filtrar da lo mismo que rehacerlas.

    Es la equivalencia que permite calcular el conjunto supervisado una sola vez
    en lugar de una por corte. Solo se sostiene porque las variables son causales.
    """
    panel = _panel(campos=3, meses=72)
    corte = pd.Timestamp("2020-01-01")

    completo = construir_muestras(panel, range(1, 13), min_historia=24)
    filtrado = completo[completo.origen == corte].reset_index(drop=True)

    truncado = construir_muestras(
        panel[panel.fecha <= corte + pd.DateOffset(months=12)],
        range(1, 13),
        min_historia=24,
    )
    reconstruido = truncado[truncado.origen == corte].reset_index(drop=True)

    assert len(filtrado) == len(reconstruido)
    numericas = filtrado.select_dtypes("number").columns
    pd.testing.assert_frame_equal(
        filtrado[numericas].sort_index(axis=1),
        reconstruido[numericas].sort_index(axis=1),
    )


def test_la_clase_de_tamano_es_causal():
    """Debe calcularse con los doce meses previos, no con toda la serie."""
    panel = _panel(campos=1, meses=72)
    origen = pd.Timestamp("2019-01-01")

    # Un salto enorme DESPUÉS del origen no debe cambiar la clase.
    alterado = panel.copy()
    alterado.loc[alterado.fecha > origen, "bpd"] = 500_000.0

    assert clase_tamano(panel, origen).equals(clase_tamano(alterado, origen))


def test_la_clase_de_tamano_separa_campos_grandes_de_pequenos():
    panel = pd.concat(
        [
            _panel(campos=1).assign(campo="CHICO", bpd=100.0),
            _panel(campos=1).assign(campo="GRANDE", bpd=80_000.0),
        ]
    )

    clases = clase_tamano(panel, pd.Timestamp("2020-01-01"))

    assert clases["CHICO"] == "<0.5k"
    assert clases["GRANDE"] == ">50k"
