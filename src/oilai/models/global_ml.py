"""Modelo global de gradient boosting sobre todos los campos.

La diferencia conceptual con la Fase 1 es que aquí **un solo modelo** aprende de
los 452 campos a la vez, en lugar de ajustar una curva independiente por campo.
La apuesta es que los campos comparten estructura —cómo declina un campo maduro,
cuánto ruido tiene uno pequeño— y que un campo con historia corta puede
beneficiarse de lo aprendido en los demás. Un ajuste por campo no puede hacer eso.

Decisiones que conviene poder defender:

* **Pérdida L1.** La evaluación usa MAE y MASE, ambas basadas en error absoluto.
  Entrenar con L2 optimizaría la media condicional y penalizaría en exceso los
  meses atípicos, frecuentes en campos marginales. L1 optimiza la mediana
  condicional, que es lo que se está midiendo.
* **Un modelo para todos los horizontes**, con `h` como variable. La alternativa
  —doce modelos independientes— multiplica el costo y fragmenta los datos sin
  aportar: la relación entre las variables y el objetivo cambia de forma suave
  con el horizonte, y el modelo puede representarla.
* **Parada temprana sobre una partición temporal**, nunca aleatoria. Una
  partición aleatoria pondría meses futuros del mismo campo en validación y
  daría una estimación optimista.
* **Categóricas nativas.** LightGBM trata operadora y departamento sin necesidad
  de codificación por objetivo, que sería otra vía de fuga.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import columnas_predictoras, reconstruir_bpd

# Meses finales del tramo de entrenamiento reservados para la parada temprana.
MESES_VALIDACION = 12

PARAMETROS = {
    "objective": "regression_l1",
    "n_estimators": 1500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 60,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}


class ModeloGlobal:
    """Envuelve LightGBM con la reconstrucción a bpd y la partición temporal."""

    nombre = "ML-global"

    def __init__(self, **kwargs):
        self.parametros = {**PARAMETROS, **kwargs}
        self.modelo = None
        self.columnas: list[str] = []
        self.mejor_iteracion: int | None = None

    def fit(self, muestras: pd.DataFrame) -> "ModeloGlobal":
        """Entrena sobre las muestras dadas, reservando los últimos meses.

        `muestras` ya debe estar filtrado para que ningún objetivo sea posterior
        al origen de evaluación: este método no conoce el corte y no puede
        protegerse por sí solo.
        """
        import lightgbm as lgb

        self.columnas = columnas_predictoras(muestras)

        limite = muestras.fecha_objetivo.max() - pd.DateOffset(
            months=MESES_VALIDACION
        )
        entrena = muestras[muestras.fecha_objetivo <= limite]
        valida = muestras[muestras.fecha_objetivo > limite]

        # Con historia corta puede no quedar validación; entonces se entrena sin
        # parada temprana y con un número fijo de árboles.
        if len(valida) < 1000 or len(entrena) < 1000:
            entrena, valida = muestras, None

        self.modelo = lgb.LGBMRegressor(**self.parametros)

        if valida is None:
            self.modelo.set_params(n_estimators=400)
            self.modelo.fit(entrena[self.columnas], entrena.y)
        else:
            self.modelo.fit(
                entrena[self.columnas],
                entrena.y,
                eval_X=valida[self.columnas],
                eval_y=valida.y,
                eval_metric="l1",
                callbacks=[lgb.early_stopping(60, verbose=False)],
            )
            self.mejor_iteracion = self.modelo.best_iteration_

        return self

    def predict_log(self, muestras: pd.DataFrame) -> np.ndarray:
        """Predicción en la escala del objetivo: log(q_{t+h} / ancla)."""
        return self.modelo.predict(muestras[self.columnas])

    def predict_bpd(self, muestras: pd.DataFrame) -> np.ndarray:
        """Predicción en barriles por día."""
        return reconstruir_bpd(
            muestras.ancla_bpd.to_numpy(float), self.predict_log(muestras)
        )

    def importancias(self) -> pd.Series:
        """Ganancia por variable, normalizada a porcentaje."""
        imp = pd.Series(
            self.modelo.booster_.feature_importance("gain"), index=self.columnas
        )
        return (imp / imp.sum() * 100).sort_values(ascending=False)
