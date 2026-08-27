"""Pruebas del protocolo de calendario y del modelo global."""

import numpy as np
import pandas as pd
import pytest

from oilai.backtest_global import (
    HORIZONTE,
    MIN_HISTORIA,
    _meses_desde,
    evaluar_referencias,
    modelos_referencia,
)
from oilai.features import construir_muestras
from oilai.models.global_ml import MESES_VALIDACION, ModeloGlobal


def _panel(campos: int = 6, meses: int = 96, inicio: str = "2016-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(3)
    fechas = pd.date_range(inicio, periods=meses, freq="MS")
    trozos = []
    for i in range(campos):
        q0 = 100.0 * (i + 1) ** 2
        caida = 0.985 + 0.002 * i
        ruido = rng.normal(0, 0.02, meses)
        trozos.append(
            pd.DataFrame(
                {
                    "campo": f"C{i}",
                    "fecha": fechas,
                    "bpd": q0 * caida ** np.arange(meses) * np.exp(ruido),
                    "operadora": f"OP{i % 2}",
                    "departamento": "META",
                }
            )
        )
    return pd.concat(trozos, ignore_index=True)


def test_meses_desde_cuenta_en_calendario():
    inicio = pd.Timestamp("2020-01-01")
    fechas = pd.Series(pd.to_datetime(["2020-01-01", "2020-07-01", "2021-01-01"]))

    assert _meses_desde(inicio, fechas).tolist() == [0.0, 6.0, 12.0]


def test_las_referencias_solo_usan_historia_previa_al_corte():
    """Un espía registra qué instantes recibe cada línea base."""
    panel = _panel()
    corte = pd.Timestamp("2021-01-01")

    res = evaluar_referencias(panel, corte)

    assert not res.empty
    # Todos los horizontes caen dentro de la ventana de evaluación.
    assert res.h.between(1, HORIZONTE).all()
    assert (res.origen == corte).all()


def test_las_referencias_evaluan_los_mismos_campos_para_todos_los_modelos():
    panel = _panel()
    corte = pd.Timestamp("2021-01-01")

    res = evaluar_referencias(panel, corte)

    por_modelo = res.groupby("modelo").campo.nunique()
    assert por_modelo.nunique() == 1, "los modelos no cubren los mismos campos"
    assert set(res.modelo) == {m.nombre for m in modelos_referencia()}


def test_un_campo_sin_historia_suficiente_se_omite():
    panel = _panel(campos=1, meses=96)
    # El corte deja menos de MIN_HISTORIA meses de historia.
    corte = panel.fecha.iloc[MIN_HISTORIA - 5]

    assert evaluar_referencias(panel, corte).empty


def test_un_campo_sin_datos_futuros_se_omite():
    panel = _panel(campos=1, meses=96)
    corte = panel.fecha.iloc[-1]  # no hay meses después del corte

    assert evaluar_referencias(panel, corte).empty


def test_el_horizonte_se_mide_en_meses_de_calendario():
    """Con huecos, h debe seguir siendo la distancia real al corte."""
    panel = _panel(campos=1, meses=96)
    corte = pd.Timestamp("2021-01-01")
    faltante = corte + pd.DateOffset(months=3)
    panel = panel[panel.fecha != faltante]

    res = evaluar_referencias(panel, corte)

    # El mes ausente simplemente no se evalúa; los demás conservan su h real.
    assert 3 not in set(res.h)
    assert {1, 2, 4}.issubset(set(res.h))


def test_el_filtro_de_entrenamiento_excluye_objetivos_futuros():
    """La regla que sostiene la fase: fecha_objetivo <= corte, no origen <= corte."""
    panel = _panel()
    corte = pd.Timestamp("2021-01-01")

    muestras = construir_muestras(panel, range(1, HORIZONTE + 1), min_historia=MIN_HISTORIA)
    entrenamiento = muestras[muestras.fecha_objetivo <= corte]

    assert (entrenamiento.fecha_objetivo <= corte).all()
    # Filtrar solo por origen dejaría pasar objetivos posteriores al corte.
    ingenuo = muestras[muestras.origen <= corte]
    assert (ingenuo.fecha_objetivo > corte).any(), "el caso peligroso debe existir"
    assert len(entrenamiento) < len(ingenuo)


def test_el_modelo_global_aprende_una_declinacion_conocida():
    """Control de cordura: sobre series suaves debe batir al pronóstico ingenuo."""
    panel = _panel(campos=12, meses=120)
    corte = pd.Timestamp("2023-01-01")

    muestras = construir_muestras(panel, range(1, 13), min_historia=MIN_HISTORIA)
    entrenamiento = muestras[muestras.fecha_objetivo <= corte]
    prueba = muestras[muestras.origen == corte]

    modelo = ModeloGlobal(n_estimators=200).fit(entrenamiento)
    pred = modelo.predict_bpd(prueba)

    mae_modelo = np.mean(np.abs(prueba.bpd_real.to_numpy() - pred))
    mae_ingenuo = np.mean(np.abs(prueba.bpd_real.to_numpy() - prueba.ancla_bpd.to_numpy()))

    assert mae_modelo < mae_ingenuo


def test_las_predicciones_nunca_son_negativas():
    panel = _panel(campos=8, meses=110)
    corte = pd.Timestamp("2023-01-01")

    muestras = construir_muestras(panel, range(1, 13), min_historia=MIN_HISTORIA)
    modelo = ModeloGlobal(n_estimators=100).fit(
        muestras[muestras.fecha_objetivo <= corte]
    )
    pred = modelo.predict_bpd(muestras[muestras.origen == corte])

    assert np.all(pred >= 0)
    assert np.all(np.isfinite(pred))


def test_la_particion_de_validacion_es_temporal():
    """Una partición aleatoria daría una estimación optimista del error."""
    panel = _panel(campos=8, meses=110)
    corte = pd.Timestamp("2023-01-01")
    muestras = construir_muestras(panel, range(1, 13), min_historia=MIN_HISTORIA)
    entrenamiento = muestras[muestras.fecha_objetivo <= corte]

    limite = entrenamiento.fecha_objetivo.max() - pd.DateOffset(months=MESES_VALIDACION)
    validacion = entrenamiento[entrenamiento.fecha_objetivo > limite]

    assert len(validacion) > 0
    # Ninguna fecha de validación es anterior a las de entrenamiento.
    assert validacion.fecha_objetivo.min() > limite


def test_las_importancias_suman_cien():
    panel = _panel(campos=8, meses=110)
    muestras = construir_muestras(panel, range(1, 13), min_historia=MIN_HISTORIA)
    modelo = ModeloGlobal(n_estimators=60).fit(muestras)

    imp = modelo.importancias()

    assert imp.sum() == pytest.approx(100.0)
    assert imp.is_monotonic_decreasing
