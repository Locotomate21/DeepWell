"""Pruebas de las transformaciones del panel.

Se prueban las funciones puras con datos construidos a mano; ninguna prueba
depende de la red ni del Parquet descargado.
"""

import numpy as np
import pandas as pd
import pytest

from oilai.clean import _add_panel_features, _collapse_municipios, _to_float


def test_convierte_miles_con_coma():
    s = pd.Series(["508,922.00", "8,731.00", "354.00", "1,234,567.89"])
    assert _to_float(s).tolist() == [508922.0, 8731.0, 354.0, 1234567.89]


def test_valores_no_numericos_quedan_en_nan():
    assert _to_float(pd.Series(["N/A", "", None])).isna().all()


def _fila(campo, fecha, produccion, municipio, **extra):
    base = dict(
        campo=campo,
        fecha=pd.Timestamp(fecha),
        produccion_bls=produccion,
        municipio=municipio,
        operadora="ECOPETROL S.A.",
        contrato="C1",
        tipo_contrato="E&P",
        departamento="META",
        latitud=4.0,
        longitud=-73.0,
    )
    base.update(extra)
    return base


def test_suma_la_produccion_de_un_campo_repartido_en_municipios():
    """Un campo que cruza municipios se reporta por separado y debe sumarse."""
    df = pd.DataFrame(
        [
            _fila("AKACIAS", "2014-10-01", 286_095.0, "ACACIAS"),
            _fila("AKACIAS", "2014-10-01", 55.0, "GUAMAL"),
        ]
    )

    out = _collapse_municipios(df)

    assert len(out) == 1
    assert out.produccion_bls.iloc[0] == pytest.approx(286_150.0)
    assert out.n_municipios.iloc[0] == 2
    # Se conserva el municipio de mayor aporte, no uno arbitrario.
    assert out.municipio.iloc[0] == "ACACIAS"


def test_campos_distintos_no_se_mezclan():
    df = pd.DataFrame(
        [
            _fila("CASTILLA", "2020-01-01", 100.0, "CASTILLA LA NUEVA"),
            _fila("CHICHIMENE", "2020-01-01", 200.0, "ACACIAS"),
        ]
    )

    out = _collapse_municipios(df)

    assert len(out) == 2
    assert set(out.campo) == {"CASTILLA", "CHICHIMENE"}


def test_bpd_normaliza_por_los_dias_del_mes():
    """31.000 bbl en enero (31 días) y 28.000 en febrero son el mismo caudal."""
    df = pd.DataFrame(
        [
            _fila("X", "2021-01-01", 31_000.0, "M"),
            _fila("X", "2021-02-01", 28_000.0, "M"),
        ]
    )
    df["n_municipios"] = 1

    out = _add_panel_features(df)

    assert out.dias_mes.tolist() == [31, 28]
    assert out.bpd.tolist() == pytest.approx([1000.0, 1000.0])


def test_meses_desde_inicio_cuenta_desde_el_primer_reporte():
    df = pd.DataFrame(
        [
            _fila("X", "2019-03-01", 100.0, "M"),
            _fila("X", "2019-04-01", 100.0, "M"),
            _fila("X", "2020-03-01", 100.0, "M"),
        ]
    )
    df["n_municipios"] = 1

    out = _add_panel_features(df).sort_values("fecha")

    assert out.meses_desde_inicio.tolist() == [0, 1, 12]


def test_marca_los_huecos_de_reporte():
    """Un mes ausente debe señalarse, no imputarse silenciosamente."""
    df = pd.DataFrame(
        [
            _fila("X", "2019-01-01", 100.0, "M"),
            _fila("X", "2019-02-01", 100.0, "M"),
            _fila("X", "2019-06-01", 100.0, "M"),  # faltan marzo, abril, mayo
        ]
    )
    df["n_municipios"] = 1

    out = _add_panel_features(df).sort_values("fecha")

    assert out.es_hueco.tolist() == [False, False, True]
    assert out.meses_desde_reporte_previo.iloc[-1] == 4


def test_cada_campo_tiene_su_propio_origen_temporal():
    """El campo B empieza después que A: su edad productiva parte de 0 igual."""
    df = pd.DataFrame(
        [
            _fila("A", "2015-01-01", 100.0, "M"),
            _fila("A", "2015-02-01", 100.0, "M"),
            _fila("B", "2018-07-01", 100.0, "M"),
            _fila("B", "2018-08-01", 100.0, "M"),
        ]
    )
    df["n_municipios"] = 1

    out = _add_panel_features(df)

    for campo in ("A", "B"):
        edades = out[out.campo == campo].meses_desde_inicio.tolist()
        assert edades == [0, 1]
