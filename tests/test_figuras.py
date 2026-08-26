"""Pruebas del formateo de ejes.

Las figuras en sí no se prueban por comparación de imágenes (frágil y de poco
valor); se prueba la lógica que produjo defectos reales, como el formateador que
colapsaba marcas distintas en la misma etiqueta.
"""

import pytest

from oilai.figuras import SERIE, _miles


@pytest.mark.parametrize(
    ("valor", "esperado"),
    [
        (0, "0"),
        (250, "250"),
        (999, "999"),
        (1_000, "1k"),
        (1_250, "1.2k"),
        (1_750, "1.8k"),
        (10_000, "10k"),
        (125_000, "125k"),
        (1_000_000, "1M"),
        (1_500_000, "1.5M"),
    ],
)
def test_formato_de_miles(valor, esperado):
    assert _miles(valor, None) == esperado


def test_no_colapsa_marcas_distintas_en_la_misma_etiqueta():
    """El defecto original: 1 250 y 1 750 se mostraban ambos como '1k' y '2k'."""
    marcas = [0, 250, 500, 750, 1_000, 1_250, 1_500, 1_750, 2_000]

    etiquetas = [_miles(m, None) for m in marcas]

    assert len(set(etiquetas)) == len(marcas)


def test_valores_negativos_conservan_el_signo():
    assert _miles(-1_500, None) == "-1.5k"


def test_la_paleta_tiene_cuatro_slots_distintos():
    assert len(SERIE) == 4
    assert len(set(SERIE)) == 4
    assert all(c.startswith("#") and len(c) == 7 for c in SERIE)
