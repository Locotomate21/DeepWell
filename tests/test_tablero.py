"""Pruebas de la capa de datos del tablero.

Se prueban con datos construidos a mano, sin depender de los artefactos que
genera el pipeline: la lógica debe ser correcta también en un repositorio recién
clonado, y las pruebas no deben tardar minutos.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oilai.tablero import (
    ArtefactoFaltante,
    _exigir,
    alertas_campo,
    alertas_recientes,
    historia,
    origen_por_defecto,
    pronostico,
)


def _intervalos(campos=("A", "B"), origenes=3, horizonte=12) -> pd.DataFrame:
    filas = []
    for campo in campos:
        for i in range(origenes):
            origen = pd.Timestamp("2024-01-01") + pd.DateOffset(months=i)
            # El último origen queda truncado, como en los datos reales.
            hs = range(1, horizonte + 1) if i < origenes - 1 else range(1, 3)
            for h in hs:
                filas.append(
                    {
                        "campo": campo,
                        "origen": origen,
                        "fecha_objetivo": origen + pd.DateOffset(months=h),
                        "h": h,
                        "y": 1000.0 - h * 10,
                        "punto": 1000.0 - h * 8,
                        "lo": 900.0 - h * 12,
                        "hi": 1100.0 - h * 4,
                        "metodo": "Conformal-clase",
                        "clase": "0.5-5k",
                    }
                )
    return pd.DataFrame(filas)


# --- Elección del origen ---------------------------------------------------


def test_el_origen_por_defecto_tiene_el_horizonte_completo():
    """Los últimos orígenes están truncados y darían una gráfica de una barra."""
    iv = _intervalos()
    g = iv[iv.campo == "A"]

    origen = origen_por_defecto(g)

    assert len(g[g.origen == origen]) == 12
    # No es el más reciente, precisamente porque ese está truncado.
    assert origen < g.origen.max()


def test_entre_varios_completos_se_elige_el_mas_reciente():
    iv = _intervalos(campos=("A",), origenes=4)
    g = iv[iv.campo == "A"]
    completos = g.groupby("origen").h.count()

    origen = origen_por_defecto(g)

    assert origen == completos[completos == 12].index.max()


def test_el_pronostico_usa_el_origen_por_defecto():
    iv = _intervalos()

    p = pronostico("A", intervalos=iv)

    assert len(p) == 12
    assert list(p.h) == list(range(1, 13))
    assert p.attrs["origen"] == origen_por_defecto(iv[iv.campo == "A"])


def test_se_puede_pedir_un_origen_concreto():
    iv = _intervalos()
    origen = pd.Timestamp("2024-03-01")

    p = pronostico("A", origen=origen, intervalos=iv)

    assert p.attrs["origen"] == origen
    assert len(p) == 2  # ese origen está truncado


def test_un_campo_sin_pronostico_devuelve_tabla_vacia():
    assert pronostico("INEXISTENTE", intervalos=_intervalos()).empty


def test_el_intervalo_siempre_contiene_al_pronostico():
    iv = _intervalos()

    p = pronostico("A", intervalos=iv)

    assert (p.lo <= p.punto).all()
    assert (p.punto <= p.hi).all()


def test_los_campos_no_se_mezclan():
    iv = _intervalos()
    iv.loc[iv.campo == "B", "punto"] = 5000.0

    p = pronostico("A", intervalos=iv)

    assert (p.punto < 2000).all()


# --- Historia --------------------------------------------------------------


def _panel() -> pd.DataFrame:
    fechas = pd.date_range("2020-01-01", periods=24, freq="MS")
    return pd.DataFrame(
        {
            "campo": ["A"] * 24 + ["B"] * 24,
            "fecha": list(fechas) * 2,
            "bpd": list(np.linspace(1000, 800, 24)) + list(np.linspace(50, 40, 24)),
            "operadora": "ECOPETROL S.A.",
            "departamento": "META",
            "municipio": "ACACIAS",
        }
    )


def test_la_historia_devuelve_solo_el_campo_pedido():
    h = historia("A", panel=_panel())

    assert len(h) == 24
    assert h.bpd.max() == pytest.approx(1000.0)


def test_la_historia_viene_ordenada_por_fecha():
    panel = _panel().sample(frac=1, random_state=0)

    h = historia("A", panel=panel)

    assert h.fecha.is_monotonic_increasing


# --- Alertas ---------------------------------------------------------------


def _alertas() -> pd.DataFrame:
    fechas = pd.date_range("2025-06-01", periods=6, freq="MS")
    return pd.DataFrame(
        {
            "campo": ["GRANDE", "CHICO", "GRANDE", "CHICO", "OTRO", "OTRO"],
            "fecha_objetivo": fechas,
            "clase": [">50k", "<0.5k", ">50k", "<0.5k", "0.5-5k", "0.5-5k"],
            "y": [50_000.0, 40.0, 55_000.0, 30.0, 900.0, 950.0],
            "lo": [58_000.0, 45.0, 56_000.0, 60.0, 1000.0, 940.0],
            "punto": [60_000.0, 50.0, 58_000.0, 70.0, 1100.0, 960.0],
            "severidad": [0.8, 0.5, 0.1, 3.0, 0.4, 0.0],
            "anomalia_baja": [True, True, True, True, True, False],
        }
    )


def test_las_alertas_de_un_campo_se_ordenan_por_severidad():
    a = alertas_campo("CHICO", alertas=_alertas())

    assert len(a) == 2
    assert a.severidad.is_monotonic_decreasing


def test_solo_se_listan_las_caidas():
    """Un repunte por encima del intervalo no es motivo de revisión."""
    a = alertas_campo("OTRO", alertas=_alertas())

    assert len(a) == 1


def test_las_alertas_recientes_priorizan_por_barriles_perdidos():
    """Una caída leve en un campo grande importa más que una grave en uno chico."""
    r = alertas_recientes(meses=12, alertas=_alertas())

    assert r.iloc[0].campo == "GRANDE"
    assert r.deficit_bpd.is_monotonic_decreasing
    # El campo con severidad 3.0 no encabeza: pierde 30 bpd, no 8 000.
    assert r.iloc[0].severidad < r.severidad.max()


def test_las_alertas_recientes_respetan_la_ventana():
    a = _alertas()
    r = alertas_recientes(meses=2, alertas=a)

    assert len(r) < int(a.anomalia_baja.sum())
    assert r.fecha_objetivo.min() > a.fecha_objetivo.max() - pd.DateOffset(months=3)


def test_sin_alertas_se_devuelve_una_tabla_vacia():
    a = _alertas()
    a["anomalia_baja"] = False

    assert alertas_recientes(alertas=a).empty


# --- Artefactos ------------------------------------------------------------


def test_un_artefacto_ausente_da_un_mensaje_accionable(tmp_path, monkeypatch):
    """El error debe decir qué ejecutar, no solo que falta un archivo."""
    import oilai.tablero as mod

    monkeypatch.setitem(mod.ARTEFACTOS, "intervalos", tmp_path / "no_existe.parquet")

    with pytest.raises(ArtefactoFaltante, match="oilai all"):
        _exigir("intervalos")


# --- Mapa ------------------------------------------------------------------


def _caracterizacion() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "campo": ["GRANDE", "CHICO", "CERRADO", "SIN_COORD"],
            "latitud": [3.8, 6.0, 4.5, np.nan],
            "longitud": [-71.5, -73.0, -72.0, -74.0],
            "bpd_ultimo": [90_000.0, 100.0, 500.0, 700.0],
            "operadora": "ECOPETROL S.A.",
            "departamento": "META",
            "activo": [True, True, False, True],
        }
    )


def _alertas_mapa() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "campo": ["GRANDE"],
            "fecha_objetivo": [pd.Timestamp("2026-02-01")],
            "clase": [">50k"],
            "y": [80_000.0],
            "lo": [85_000.0],
            "punto": [90_000.0],
            "severidad": [0.5],
            "anomalia_baja": [True],
        }
    )


def _mapa(**kwargs):
    from oilai.tablero import mapa_campos

    return mapa_campos(
        caracterizacion=_caracterizacion(), alertas=_alertas_mapa(), **kwargs
    )


def test_el_mapa_excluye_campos_sin_coordenadas():
    m = _mapa()

    assert "SIN_COORD" not in set(m.campo)


def test_el_mapa_puede_excluir_campos_cerrados():
    activos = _mapa(solo_activos=True)
    todos = _mapa(solo_activos=False)

    assert "CERRADO" not in set(activos.campo)
    assert "CERRADO" in set(todos.campo)


def test_el_area_del_circulo_es_proporcional_a_la_produccion():
    """Usar el radio directamente exageraría los campos grandes."""
    from oilai.tablero import ESCALA_RADIO, RADIO_MAX, RADIO_MIN

    m = _mapa(solo_activos=False).set_index("campo")
    esperado = np.clip(np.sqrt(500.0) * ESCALA_RADIO, RADIO_MIN, RADIO_MAX)

    assert m.loc["CERRADO", "radio"] == pytest.approx(esperado)


def test_el_radio_esta_acotado_por_arriba_y_por_abajo():
    from oilai.tablero import RADIO_MAX, RADIO_MIN

    m = _mapa(solo_activos=False)

    assert m.radio.min() >= RADIO_MIN
    assert m.radio.max() <= RADIO_MAX


def test_el_color_distingue_los_campos_en_alerta():
    from oilai.tablero import COLOR_ALERTA, COLOR_NORMAL

    m = _mapa().set_index("campo")

    assert m.loc["GRANDE", "en_alerta"]
    assert m.loc["GRANDE", "color"] == COLOR_ALERTA
    assert not m.loc["CHICO", "en_alerta"]
    assert m.loc["CHICO", "color"] == COLOR_NORMAL


def test_el_mapa_solo_usa_dos_colores():
    """Con cuatro categorías la paleta no supera el umbral de discriminación."""
    m = _mapa(solo_activos=False)

    assert m.color.nunique() <= 2


def test_el_mapa_se_ordena_por_produccion():
    m = _mapa()

    assert m.bpd_ultimo.is_monotonic_decreasing


def test_el_resumen_del_mapa_cuenta_produccion_en_alerta():
    from oilai.tablero import resumen_mapa

    m = _mapa()
    r = resumen_mapa(m)

    assert r["campos"] == 2
    assert r["en_alerta"] == 1
    assert r["bpd_en_alerta"] == pytest.approx(90_000.0)
    assert r["bpd_total"] == pytest.approx(90_100.0)


def test_el_resumen_de_un_mapa_vacio_no_falla():
    from oilai.tablero import resumen_mapa

    r = resumen_mapa(pd.DataFrame())

    assert r["campos"] == 0 and r["bpd_total"] == 0.0


# --- Prueba de humo de la aplicación ---------------------------------------


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "reports" / "intervalos.parquet").exists(),
    reason="requiere los artefactos del pipeline (`oilai all`)",
)
def test_la_aplicacion_se_ejecuta_sin_excepciones():
    """Ejecuta el script de Streamlit de principio a fin.

    Que el servidor arranque no prueba nada: el script solo corre cuando se abre
    una sesión. `AppTest` sí lo ejecuta y recoge cualquier excepción, que es la
    única forma de detectar que una vista está rota sin abrir el navegador.
    """
    from streamlit.testing.v1 import AppTest

    ruta = Path(__file__).resolve().parents[1] / "app" / "tablero.py"
    app = AppTest.from_file(str(ruta), default_timeout=300).run()

    assert not app.exception, [str(e.value) for e in app.exception]
    assert app.title[0].value.startswith("DeepWell")


# --- Comparador de modelos -------------------------------------------------


def _comparativa(campos=("A", "B"), modelos=("Naive", "ML-global", "Híbrido-regimen")):
    filas = []
    for campo in campos:
        for origen in (pd.Timestamp("2024-03-01"), pd.Timestamp("2025-03-01")):
            for h in range(1, 13):
                real = 1000.0 - h * 5
                for i, modelo in enumerate(modelos):
                    filas.append(
                        {
                            "campo": campo,
                            "origen": origen,
                            "h": h,
                            "y": real,
                            # El primer modelo acierta; los demás se equivocan
                            # progresivamente más.
                            "yhat": real + i * 20,
                            "escala": 10.0,
                            "clase": "0.5-5k",
                            "modelo": modelo,
                        }
                    )
    return pd.DataFrame(filas)


def test_los_modelos_sugeridos_van_primero():
    from oilai.tablero import MODELOS_SUGERIDOS, modelos_disponibles

    datos = _comparativa(modelos=("Arps-24m", "Naive", "ML-global", "Híbrido-regimen"))

    orden = modelos_disponibles(datos)

    assert orden[: len(MODELOS_SUGERIDOS)] == MODELOS_SUGERIDOS
    assert set(orden) == set(datos.modelo.unique())


def test_el_ranking_ordena_del_mejor_al_peor():
    from oilai.tablero import ranking_modelos

    tabla = ranking_modelos(_comparativa())

    assert tabla.index[0] == "Naive"  # el que acierta en los datos sintéticos
    assert tabla.MASE.is_monotonic_increasing


def test_el_mase_por_horizonte_cubre_todos_los_horizontes():
    from oilai.tablero import mase_por_horizonte

    piv = mase_por_horizonte(_comparativa())

    assert list(piv.columns) == list(range(1, 13))
    assert piv.loc["Naive"].sum() == pytest.approx(0.0)


def test_las_trayectorias_traen_una_columna_por_modelo():
    from oilai.tablero import trayectorias

    datos = _comparativa()
    tray = trayectorias("A", datos=datos)

    assert len(tray) == 12
    for modelo in datos.modelo.unique():
        assert modelo in tray.columns
    assert "y" in tray.columns


def test_las_trayectorias_usan_el_origen_mas_reciente():
    from oilai.tablero import trayectorias

    tray = trayectorias("A", datos=_comparativa())

    assert tray.attrs["origen"] == pd.Timestamp("2025-03-01")


def test_se_puede_comparar_en_un_origen_concreto():
    from oilai.tablero import trayectorias

    origen = pd.Timestamp("2024-03-01")
    tray = trayectorias("A", origen=origen, datos=_comparativa())

    assert tray.attrs["origen"] == origen


def test_un_campo_sin_comparacion_devuelve_tabla_vacia():
    from oilai.tablero import error_por_modelo, trayectorias

    datos = _comparativa()

    assert trayectorias("INEXISTENTE", datos=datos).empty
    assert error_por_modelo("INEXISTENTE", datos=datos).empty


def test_el_error_por_campo_identifica_al_mejor_modelo():
    from oilai.tablero import error_por_modelo

    tabla = error_por_modelo("A", datos=_comparativa())

    assert tabla.index[0] == "Naive"
    assert tabla.loc["Naive", "MAE_bpd"] == pytest.approx(0.0)
    assert tabla.MASE.is_monotonic_increasing


def test_el_sesgo_conserva_el_signo():
    """Los modelos sintéticos sobreestiman: el sesgo debe salir positivo."""
    from oilai.tablero import error_por_modelo

    tabla = error_por_modelo("A", datos=_comparativa())

    assert tabla.loc["ML-global", "sesgo_bpd"] > 0


def test_el_tope_de_modelos_graficados_es_tres():
    """En un mismo plano la paleta solo garantiza tres colores distinguibles."""
    from oilai.tablero import MAX_MODELOS_GRAFICA

    assert MAX_MODELOS_GRAFICA == 3


def test_los_campos_comparables_vienen_ordenados():
    from oilai.tablero import campos_comparables

    campos = campos_comparables(_comparativa(campos=("Z", "A", "M")))

    assert campos == ["A", "M", "Z"]


def test_un_horizonte_sin_dato_queda_como_hueco_explicito():
    """Sin esto la gráfica une los vecinos con una recta e inventa un valor.

    Ocurre de verdad: noviembre de 2025 es un mes de publicación incompleta de
    la ANH, y los campos que no reportaron se quedan sin objetivo en el
    horizonte que apunta a ese mes.
    """
    from oilai.tablero import trayectorias

    datos = _comparativa()
    sin_h8 = datos[datos.h != 8]

    tray = trayectorias("A", datos=sin_h8)

    assert list(tray.h) == list(range(1, 13))  # el horizonte sigue completo
    assert tray.loc[tray.h == 8, "y"].isna().all()
    assert tray.loc[tray.h == 7, "y"].notna().all()


def test_las_trayectorias_no_inventan_horizontes_de_mas():
    from oilai.tablero import trayectorias

    tray = trayectorias("A", datos=_comparativa())

    assert tray.h.max() == 12
    assert tray.y.notna().all()
