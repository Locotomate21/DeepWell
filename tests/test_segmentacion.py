"""Pruebas de la segmentación de campos."""

import numpy as np
import pandas as pd
import pytest

from oilai.segmentacion import (
    LIMITES,
    VARIABLES,
    _nombrar,
    elegir_k,
    perfil_segmentos,
    preparar,
    representantes,
)


def _caracterizacion(n: int = 40) -> pd.DataFrame:
    """Campos sintéticos en dos grupos claramente separados."""
    rng = np.random.default_rng(0)
    mitad = n // 2
    return pd.DataFrame(
        {
            "campo": [f"C{i}" for i in range(n)],
            "bpd_medio": np.r_[
                rng.normal(5000, 200, mitad), rng.normal(50, 5, n - mitad)
            ].clip(1),
            "bpd_ultimo": np.r_[
                rng.normal(4800, 200, mitad), rng.normal(20, 5, n - mitad)
            ].clip(1),
            "declinacion_anual_pct": np.r_[
                rng.normal(2, 0.5, mitad), rng.normal(40, 3, n - mitad)
            ],
            "volatilidad": np.r_[
                rng.normal(0.1, 0.02, mitad), rng.normal(0.8, 0.05, n - mitad)
            ],
            "madurez": np.r_[
                rng.normal(0.8, 0.05, mitad), rng.normal(0.05, 0.01, n - mitad)
            ],
            "meses_historia": 120,
            "activo": [True] * mitad + [False] * (n - mitad),
        }
    )


def test_preparar_crea_el_logaritmo_del_tamano():
    d = preparar(_caracterizacion())

    assert "log_bpd" in d.columns
    assert d.log_bpd.max() < 6  # log10 de 5000 ~ 3.7


def test_preparar_recorta_los_valores_extremos():
    """Un campo con declinación absurda debe acotarse, no eliminarse."""
    d = _caracterizacion(10)
    d.loc[0, "declinacion_anual_pct"] = 500.0
    d.loc[1, "volatilidad"] = 9.0

    out = preparar(d)

    assert len(out) == 10  # no se pierde ningún campo
    assert out.declinacion_anual_pct.max() <= LIMITES["declinacion_anual_pct"][1]
    assert out.volatilidad.max() <= LIMITES["volatilidad"][1]


def test_preparar_descarta_campos_sin_volatilidad():
    d = _caracterizacion(10)
    d.loc[0, "volatilidad"] = np.nan

    assert len(preparar(d)) == 9


def test_elegir_k_devuelve_una_silueta_por_candidato():
    tabla = elegir_k(preparar(_caracterizacion()), ks=range(2, 5))

    assert list(tabla.k) == [2, 3, 4]
    assert tabla.silueta.between(-1, 1).all()


def test_dos_grupos_bien_separados_dan_silueta_alta():
    """Control de cordura del procedimiento sobre datos con estructura conocida."""
    tabla = elegir_k(preparar(_caracterizacion()), ks=range(2, 3))

    assert tabla.silueta.iloc[0] > 0.5


def test_nombrar_usa_el_perfil_y_no_el_indice():
    """Los índices de KMeans son arbitrarios; el nombre debe salir del perfil."""
    df = pd.DataFrame(
        {
            "segmento": [0] * 3 + [1] * 3,
            "bpd_medio": [2000.0] * 3 + [50.0] * 3,
            "declinacion_anual_pct": [1.0] * 3 + [45.0] * 3,
            "volatilidad": [0.15] * 3 + [0.5] * 3,
            "madurez": [0.7] * 3 + [0.03] * 3,
        }
    )

    nombres = _nombrar(df)

    assert nombres[df.segmento == 0].unique().tolist() == ["Núcleo estable"]
    assert nombres[df.segmento == 1].unique().tolist() == ["En agotamiento"]


def test_nombrar_desempata_nombres_repetidos():
    """Dos grupos con el mismo perfil no deben quedar con la misma etiqueta."""
    df = pd.DataFrame(
        {
            "segmento": [0] * 2 + [1] * 2,
            "bpd_medio": [3000.0, 3000.0, 1000.0, 1000.0],
            "declinacion_anual_pct": [1.0] * 4,
            "volatilidad": [0.1] * 4,
            "madurez": [0.8] * 4,
        }
    )

    nombres = _nombrar(df)

    assert nombres.nunique() == 2


def _segmentado() -> pd.DataFrame:
    d = preparar(_caracterizacion())
    d["segmento"] = (d.bpd_medio < 500).astype(int)
    d["segmento_nombre"] = d.segmento.map({0: "Núcleo estable", 1: "En agotamiento"})
    return d


def test_representantes_elige_solo_campos_activos():
    """Ilustrar un segmento con un campo cerrado hace años induce a error."""
    d = _segmentado()

    elegidos = representantes(d)

    for nombre, campo in elegidos.items():
        fila = d[d.campo == campo].iloc[0]
        # El grupo "En agotamiento" es todo inactivo: allí se permite el repliegue.
        if d[(d.segmento_nombre == nombre)].activo.any():
            assert fila.activo, f"{nombre} quedó representado por un campo inactivo"


def test_representantes_devuelve_uno_por_segmento():
    d = _segmentado()

    elegidos = representantes(d)

    assert set(elegidos) == set(d.segmento_nombre.unique())
    assert len(set(elegidos.values())) == len(elegidos)


def test_perfil_reparte_cien_por_ciento_de_la_produccion_actual():
    d = _segmentado()

    perfil = perfil_segmentos(d)

    assert perfil.pct_produccion.sum() == pytest.approx(100.0)


def test_perfil_solo_cuenta_produccion_de_campos_activos():
    """La cuota debe calcularse sobre lo que el país produce hoy."""
    d = _segmentado()
    esperado = d[d.activo].groupby("segmento_nombre").bpd_ultimo.sum()

    perfil = perfil_segmentos(d)

    for nombre, valor in esperado.items():
        assert perfil.loc[nombre, "bpd_actual"] == pytest.approx(valor)


def test_perfil_distingue_campos_totales_de_activos():
    d = _segmentado()

    perfil = perfil_segmentos(d)

    assert (perfil.campos_activos <= perfil.campos).all()


def test_las_variables_de_agrupamiento_estan_documentadas():
    """Contrato del módulo: si cambian las variables, la prueba obliga a revisarlo."""
    assert VARIABLES == [
        "log_bpd",
        "declinacion_anual_pct",
        "volatilidad",
        "madurez",
    ]
