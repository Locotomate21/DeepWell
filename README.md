# DeepWell · Pronóstico de producción petrolera con IA híbrida

Pronóstico de producción de los campos petroleros de Colombia combinando
**modelos físicos de declinación** (Arps) con **aprendizaje automático**, sobre
datos oficiales de la Agencia Nacional de Hidrocarburos (ANH).

## Pregunta de investigación

En la industria, el estándar para pronosticar producción es el *análisis de
curvas de declinación* (DCA) de Arps, un modelo físico de tres parámetros
publicado en 1945 y todavía vigente. Cualquier propuesta basada en IA debe
demostrar que lo supera. La pregunta que guía el proyecto es:

> **¿Bajo qué condiciones un modelo de aprendizaje automático supera al análisis
> de declinación clásico, y puede un modelo híbrido combinar las ventajas de
> ambos?**

## Datos

**Producción Fiscalizada de Crudo Consolidada** — Agencia Nacional de
Hidrocarburos, publicada en el portal de Datos Abiertos de Colombia
([`fdvb-hsrf`](https://www.datos.gov.co/resource/fdvb-hsrf)).

| | |
|---|---|
| Registros crudos | 53 237 |
| Panel limpio (campo × mes) | 50 103 |
| Campos | 608 |
| Operadoras | 94 |
| Cobertura | enero 2014 – marzo 2026 (147 meses) |
| Historia media por campo | 82 meses |
| Campos aptos para backtesting | 425 |

Cada registro trae fecha, campo, operadora, contrato, tipo de contrato,
departamento, municipio, coordenadas geográficas y producción en barriles.

**Validación externa.** Agregando el panel a nivel nacional se obtienen
≈ 990 000 bpd en 2014 y ≈ 736 000 bpd en 2021, cifras que coinciden con las
estadísticas oficiales de producción de Colombia. Los datos son consistentes con
la realidad del sector.

## Resultados de la Fase 1

Backtesting *walk-forward* sobre 425 campos, 3 orígenes de pronóstico por campo
y horizontes de 1 a 12 meses: **106 428 predicciones evaluadas**.

La métrica de decisión es **MASE** (error escalado por el del pronóstico ingenuo
de un paso), que permite comparar campos de 500 bpd con campos de 120 000 bpd.
Menor es mejor; MASE < 1 significa batir al pronóstico ingenuo.

### El error crece con el horizonte, y hay un cruce

MASE medio por horizonte:

| Modelo | h=1 | h=3 | h=6 | h=9 | **h=12** |
|---|---|---|---|---|---|
| Naive | **0.81** | **1.62** | **2.23** | **2.55** | 3.04 |
| Media móvil 3m | 0.98 | 1.70 | 2.26 | 2.54 | 3.06 |
| Drift | 0.83 | 1.69 | 2.38 | 2.86 | 3.42 |
| Arps 24m | 1.57 | 2.07 | 2.39 | 2.58 | **2.97** |
| Arps 36m | 1.76 | 2.20 | 2.51 | 2.66 | 3.03 |

A corto plazo repetir el último valor es difícil de batir. **A 12 meses la
física toma la delantera**: Arps captura la tendencia de declinación que los
métodos estadísticos ignoran.

### El tamaño del campo cambia qué modelo gana

MASE a horizonte de 12 meses, por clase de campo:

| Modelo | < 500 bpd | 0.5–5k | 5–50k | **> 50k bpd** |
|---|---|---|---|---|
| Naive | 2.65 | 3.11 | **5.66** | 5.03 |
| Media móvil 3m | 2.59 | 3.19 | 5.94 | 4.45 |
| Arps 24m | **2.40** | 3.15 | 6.44 | **3.56** |
| Drift | 2.94 | 3.55 | 6.40 | 5.54 |

En los campos grandes —Rubiales, Castilla, Chichimene, los que sostienen la
producción nacional— **Arps reduce el error un 29 % frente al Naive**
(3.56 vs 5.03). Son también los campos donde un error de pronóstico cuesta más
dinero.

### Lectura

Ningún modelo de la línea base alcanza MASE < 1 a horizontes largos: **hay
margen amplio de mejora**, y ese es el espacio que ocupan las fases 3 y 4. El
cruce entre régimen estadístico (corto plazo) y régimen físico (largo plazo,
campos grandes) es la motivación directa del modelo híbrido.

## Instalación

```bash
git clone git@github.com:Locotomate21/DeepWell.git
cd DeepWell
python -m pip install -e ".[dev]"
```

Para las fases de modelado y la app:

```bash
python -m pip install -e ".[modelos,app]"
```

## Uso

```bash
oilai all              # pipeline completo: descarga -> panel -> benchmark
oilai ingest           # solo descarga el histórico de la ANH
oilai panel            # solo reconstruye el panel limpio
oilai benchmark        # solo corre el backtesting
oilai all --force      # ignora la caché y recalcula todo
```

Cada etapa cachea su salida en Parquet, así que repetir `oilai all` solo
recalcula lo que falte.

Pruebas:

```bash
python -m pytest
```

## Estructura

```
src/oilai/
├── config.py           rutas y definición de la fuente de datos
├── ingest.py           descarga paginada desde la API Socrata
├── clean.py            panel mensual: agregación, bpd, edad productiva
├── evaluate.py         backtesting walk-forward y métricas
├── pipeline.py         orquestador (comando `oilai`)
├── run_benchmark.py    benchmark sobre todos los campos
└── models/
    ├── arps.py         declinación exponencial, hiperbólica y armónica
    └── baselines.py    Naive, media móvil, drift, Arps

tests/                  34 pruebas, incluida la de ausencia de fuga temporal
docs/metodologia.md     decisiones metodológicas y su justificación
data/raw/               snapshot versionado de los datos de la ANH
```

## Hoja de ruta

| Fase | Contenido | Estado |
|---|---|---|
| **1. Fundación de datos y línea base** | Ingesta, panel, Arps, baselines, backtesting, pruebas | ✅ **completa** |
| 2. Análisis exploratorio | Caracterización de los 608 campos, figuras, mapa, segmentación | pendiente |
| 3. Modelo global de ML | Features + LightGBM multi-horizonte sobre todos los campos | pendiente |
| 4. Modelo híbrido física + ML | Arps + ML sobre residuales — aporte original | pendiente |
| 5. Incertidumbre y anomalías | Intervalos de predicción, detección de caídas atípicas | pendiente |
| 6. Dashboard y entregable | App Streamlit e informe metodológico | pendiente |

## Fuentes

- Arps, J. J. (1945). *Analysis of Decline Curves*. Transactions of the AIME, 160(01), 228–247.
- Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of forecast accuracy*. International Journal of Forecasting, 22(4), 679–688. — origen de MASE.
- Agencia Nacional de Hidrocarburos. *Producción Fiscalizada de Crudo Consolidada*. Datos Abiertos Colombia.

## Licencia

MIT.
