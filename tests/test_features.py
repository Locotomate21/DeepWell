"""Pruebas del conjunto supervisado del modelo global.

La prueba central es `test_las_variables_no_miran_al_futuro`. Un modelo global
mezcla todos los campos, así que una fuga temporal es fácil de introducir y
difícil de notar: produciría métricas excelentes y un modelo inservible.
"""

import numpy as np
import pandas as pd
import pytest

from oilai.features import (
    OBJETIVO_MAX,
    OBJETIVO_MIN,
    PISO_BPD,
    columnas_predictoras,
    construir_muestras,
    reconstruir_bpd,
)


def _panel(
    campo: str = "A",
    meses: int = 60,
    inicio: str = "2015-01-01",
    q0: float = 1000.0,
    decaimiento: float = 0.98,
) -> pd.DataFrame:
    fechas = pd.date_range(inicio, periods=meses, freq="MS")
    return pd.DataFrame(
        {
            "campo": campo,
            "fecha": fechas,
            "bpd": q0 * decaimiento ** np.arange(meses),
            "operadora": "OP",
            "departamento": "META",
        }
    )


def test_las_variables_no_miran_al_futuro():
    """Truncar la serie después del origen no debe alterar ninguna variable."""
    panel = _panel(meses=80)
    corte = pd.Timestamp("2018-06-01")

    completo = construir_muestras(
        panel, [6], origenes=[corte], con_objetivo=False
    )
    truncado = construir_muestras(
        panel[panel.fecha <= corte], [6], origenes=[corte], con_objetivo=False
    )

    cols = columnas_predictoras(completo)
    assert len(completo) == 1 and len(truncado) == 1

    for c in cols:
        a, b = completo[c].iloc[0], truncado[c].iloc[0]
        if pd.isna(a) and pd.isna(b):
            continue
        assert a == b, f"la variable {c} cambia al truncar el futuro"


def test_el_objetivo_es_el_cambio_logaritmico_respecto_al_ancla():
    panel = _panel(meses=60)
    corte = pd.Timestamp("2017-01-01")

    m = construir_muestras(panel, [3], origenes=[corte])
    fila = m.iloc[0]

    real = panel.loc[panel.fecha == corte + pd.DateOffset(months=3), "bpd"].iloc[0]

    assert fila.bpd_real == pytest.approx(real)
    assert fila.y == pytest.approx(np.log(real / fila.ancla_bpd), rel=1e-6)


def test_reconstruir_bpd_invierte_el_objetivo():
    panel = _panel(meses=60)
    m = construir_muestras(panel, [1, 6, 12], origenes=[pd.Timestamp("2017-01-01")])

    recuperado = reconstruir_bpd(m.ancla_bpd.to_numpy(), m.y.to_numpy())

    assert np.allclose(recuperado, m.bpd_real.to_numpy(), rtol=1e-6)


def test_el_objetivo_se_recorta_a_un_rango_plausible():
    """Una reactivación extrema no debe dominar la función de pérdida."""
    panel = _panel(meses=40)
    # Un salto de 1000x en un mes.
    panel.loc[panel.index[-1], "bpd"] = 1_000_000.0
    corte = panel.fecha.iloc[-2]

    m = construir_muestras(panel, [1], origenes=[corte])

    assert m.y.between(OBJETIVO_MIN, OBJETIVO_MAX).all()


def test_el_ancla_es_la_mediana_reciente_y_no_el_ultimo_valor():
    """Un mes atípico no debe desplazar todas las predicciones del campo."""
    panel = _panel(meses=40, decaimiento=1.0, q0=100.0)
    corte = panel.fecha.iloc[-1]
    panel.loc[panel.index[-1], "bpd"] = 10_000.0  # atípico en el origen

    m = construir_muestras(panel, [1], origenes=[corte], con_objetivo=False)

    # Mediana de (100, 100, 10000) = 100, no 10000.
    assert m.ancla_bpd.iloc[0] == pytest.approx(100.0)


def test_los_huecos_no_desplazan_los_rezagos():
    """rel_lag12 debe significar 'hace doce meses', no 'doce reportes atrás'."""
    panel = _panel(meses=60)
    # Se eliminan tres meses intermedios.
    faltantes = pd.to_datetime(["2016-03-01", "2016-04-01", "2016-05-01"])
    con_huecos = panel[~panel.fecha.isin(faltantes)]
    corte = pd.Timestamp("2017-06-01")

    completo = construir_muestras(panel, [1], origenes=[corte], con_objetivo=False)
    hueco = construir_muestras(con_huecos, [1], origenes=[corte], con_objetivo=False)

    # El valor de hace 12 meses (2016-06) existe en ambos: el rezago coincide.
    assert hueco.rel_lag12.iloc[0] == pytest.approx(completo.rel_lag12.iloc[0])


def test_un_hueco_en_el_rezago_produce_nan_y_no_un_valor_erroneo():
    panel = _panel(meses=60)
    corte = pd.Timestamp("2017-06-01")
    faltante = corte - pd.DateOffset(months=12)
    con_hueco = panel[panel.fecha != faltante]

    m = construir_muestras(con_hueco, [1], origenes=[corte], con_objetivo=False)

    # PISO_BPD no debe usarse para inventar un rezago inexistente.
    assert np.isnan(m.rel_lag12.iloc[0])


def test_no_se_generan_muestras_sin_historia_suficiente():
    panel = _panel(meses=30)

    m = construir_muestras(panel, [1], min_historia=24)

    # Solo los meses con al menos 24 observaciones previas son orígenes válidos.
    assert (m.meses_observados >= 24).all()


def test_campos_demasiado_cortos_se_omiten():
    panel = _panel(meses=10)

    assert construir_muestras(panel, [1], min_historia=24).empty


def test_solo_se_generan_los_origenes_pedidos():
    panel = _panel(meses=60)
    origenes = [pd.Timestamp("2017-01-01"), pd.Timestamp("2018-01-01")]

    m = construir_muestras(panel, [1, 2], origenes=origenes, con_objetivo=False)

    assert set(m.origen) == set(origenes)
    assert len(m) == 4  # 2 orígenes x 2 horizontes


def test_la_fecha_objetivo_es_el_origen_mas_el_horizonte():
    panel = _panel(meses=60)
    corte = pd.Timestamp("2017-01-01")

    m = construir_muestras(panel, [1, 6, 12], origenes=[corte], con_objetivo=False)

    for _, fila in m.iterrows():
        assert fila.fecha_objetivo == corte + pd.DateOffset(months=int(fila.h))


def test_varios_campos_no_se_mezclan():
    panel = pd.concat([_panel("A", q0=100.0), _panel("B", q0=50_000.0)])
    corte = pd.Timestamp("2017-01-01")

    m = construir_muestras(panel, [1], origenes=[corte], con_objetivo=False)

    anclas = m.set_index("campo").ancla_bpd
    assert anclas["A"] < anclas["B"]
    assert len(m) == 2


def test_la_produccion_nula_no_produce_infinitos():
    """Un campo cerrado da log(0); el piso evita que se propague -inf."""
    panel = _panel(meses=40)
    panel.loc[panel.index[-5:], "bpd"] = 0.0
    corte = panel.fecha.iloc[-1]

    m = construir_muestras(panel, [1], origenes=[corte], con_objetivo=False)

    assert np.isfinite(m[columnas_predictoras(m)].select_dtypes("number")).all().all()
    assert m.ancla_bpd.iloc[0] >= PISO_BPD


def test_las_categoricas_no_entran_como_numeros():
    panel = _panel(meses=60)

    m = construir_muestras(panel, [1], origenes=[pd.Timestamp("2017-01-01")])

    assert isinstance(m.operadora.dtype, pd.CategoricalDtype)
    assert isinstance(m.departamento.dtype, pd.CategoricalDtype)


def test_columnas_predictoras_excluye_objetivo_e_identificadores():
    panel = _panel(meses=60)
    m = construir_muestras(panel, [1], origenes=[pd.Timestamp("2017-01-01")])

    cols = columnas_predictoras(m)

    for prohibida in ("y", "bpd_real", "campo", "origen", "fecha_objetivo", "ancla_bpd"):
        assert prohibida not in cols
    assert "h" in cols
