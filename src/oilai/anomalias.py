"""Detección de caídas anómalas de producción.

Una anomalía es un mes en el que la producción real cae **por debajo del
intervalo de predicción** construido con la información disponible el mes
anterior. No es lo mismo que una caída grande: un campo en declinación acelerada
cae mucho todos los meses y eso es lo esperado, no una anomalía. Lo que se
detecta es la desviación respecto a lo que el modelo, conociendo el
comportamiento del campo, consideraba plausible.

**El problema de validación.** No existe un registro público de intervenciones,
paros o fallos con el que contrastar las alertas, así que no se puede medir
precisión ni exhaustividad contra una verdad conocida. Declararlo y quedarse ahí
sería insuficiente, de modo que se valida de forma indirecta, con una pregunta
que sí se puede responder con los datos:

> ¿La producción de los meses siguientes a una alerta cae más que la de los meses
> siguientes a un mes normal?

Si una alerta no anticipa nada, el detector marca ruido y no sirve. Si anticipa
caídas sostenidas, es una señal de alerta temprana con valor operativo. Esa
comparación es la que decide.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Meses posteriores a la alerta que se observan para juzgar si la caída
# se sostuvo. Seis meses dan tiempo a que una intervención surta efecto y a
# distinguir un bache puntual de un declive real.
VENTANA_POSTERIOR = 6
VENTANA_PREVIA = 6


def detectar(predicciones: pd.DataFrame) -> pd.DataFrame:
    """Marca las observaciones que caen fuera de su intervalo de predicción.

    `predicciones` debe traer `campo`, `fecha_objetivo`, `y`, `lo`, `hi`.

    Se distingue la dirección porque las consecuencias son opuestas: una caída
    por debajo del intervalo sugiere un problema —falla de equipo, cierre de
    pozos, restricción de transporte—; un repunte por encima suele indicar una
    intervención exitosa o la entrada de pozos nuevos.
    """
    df = predicciones.copy()

    df["anomalia_baja"] = df.y < df.lo
    df["anomalia_alta"] = df.y > df.hi
    df["anomalia"] = df.anomalia_baja | df.anomalia_alta

    # Severidad: a cuántas anchuras de intervalo quedó el valor real. Permite
    # ordenar las alertas por gravedad en vez de tratarlas todas por igual.
    ancho = (df.hi - df.lo).replace(0, np.nan)
    exceso = np.where(df.anomalia_baja, df.lo - df.y, np.where(df.anomalia_alta, df.y - df.hi, 0.0))
    df["severidad"] = exceso / ancho

    return df


def tasa_por_mes(alertas: pd.DataFrame) -> pd.DataFrame:
    """Fracción de campos en alerta cada mes.

    Sirve de comprobación de sensatez: si el detector funciona, los periodos de
    perturbación conocida del sector deben destacar sobre el resto.
    """
    return (
        alertas.groupby("fecha_objetivo")
        .agg(
            campos=("campo", "nunique"),
            bajas=("anomalia_baja", "sum"),
            altas=("anomalia_alta", "sum"),
        )
        .assign(
            pct_bajas=lambda d: d.bajas / d.campos * 100,
            pct_altas=lambda d: d.altas / d.campos * 100,
        )
        .reset_index()
    )


def _serie_por_campo(panel: pd.DataFrame) -> dict[str, pd.Series]:
    """Serie mensual densa de cada campo, para consultar ventanas por fecha."""
    series: dict[str, pd.Series] = {}
    for campo, g in panel.groupby("campo", sort=False):
        s = g.set_index("fecha").bpd.sort_index()
        series[campo] = s.reindex(
            pd.date_range(s.index.min(), s.index.max(), freq="MS")
        )
    return series


def evolucion_posterior(
    panel: pd.DataFrame,
    alertas: pd.DataFrame,
    meses_despues: int = VENTANA_POSTERIOR,
    meses_antes: int = VENTANA_PREVIA,
) -> pd.DataFrame:
    """Cociente entre la producción posterior y la previa a cada observación.

    Un valor de 0.8 significa que en los seis meses siguientes el campo produjo
    un 20 % menos que en los seis anteriores. Se calcula igual para las
    observaciones en alerta y para las normales, que son el grupo de comparación.
    """
    series = _serie_por_campo(panel)
    filas = []

    for fila in alertas.itertuples():
        s = series.get(fila.campo)
        if s is None:
            continue

        t = pd.Timestamp(fila.fecha_objetivo)
        previa = s.loc[t - pd.DateOffset(months=meses_antes) : t]
        posterior = s.loc[
            t + pd.DateOffset(months=1) : t + pd.DateOffset(months=meses_despues)
        ]

        # Se exige historia a ambos lados: sin ella el cociente no significa nada.
        if previa.notna().sum() < 3 or posterior.notna().sum() < 3:
            continue

        base = previa.mean()
        if not np.isfinite(base) or base <= 0:
            continue

        filas.append(
            {
                "campo": fila.campo,
                "fecha": t,
                "anomalia_baja": bool(fila.anomalia_baja),
                "anomalia": bool(fila.anomalia),
                "cociente": float(posterior.mean() / base),
            }
        )

    return pd.DataFrame(filas)


def validar(evolucion: pd.DataFrame) -> pd.DataFrame:
    """Compara la evolución posterior de meses en alerta y meses normales."""
    grupos = {
        "alerta de caída": evolucion[evolucion.anomalia_baja],
        "sin alerta": evolucion[~evolucion.anomalia],
    }

    filas = []
    for nombre, g in grupos.items():
        if g.empty:
            continue
        filas.append(
            {
                "grupo": nombre,
                "n": len(g),
                "cociente_medio": g.cociente.mean(),
                "cociente_mediano": g.cociente.median(),
                "pct_cae_mas_20": float((g.cociente < 0.8).mean() * 100),
                "pct_cae_mas_50": float((g.cociente < 0.5).mean() * 100),
            }
        )

    return pd.DataFrame(filas).set_index("grupo")
