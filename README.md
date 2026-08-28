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

## Resultados de la Fase 2

El análisis exploratorio no describe por describir: **explica el resultado de la
Fase 1** y define dónde debe concentrarse el modelado.

### Por qué ningún modelo baja de MASE 1

La volatilidad mensual de un campo cae de forma sostenida con su tamaño: de
≈ 0.63 en el decil más pequeño a ≈ 0.11 en el más grande. Buena parte del error
en los campos pequeños es **irreducible**, no un defecto del modelo.

| Clase de campo | Declinación anual | Volatilidad | Madurez |
|---|---|---|---|
| < 500 bpd | 15.5 % | 0.51 | 0.14 |
| 0.5–5k bpd | 7.5 % | 0.22 | 0.31 |
| 5–50k bpd | 3.6 % | 0.15 | 0.50 |
| > 50k bpd | 3.2 % | 0.03 | 0.56 |

Los campos grandes declinan despacio y con poco ruido — exactamente el régimen
donde Arps ganaba en el benchmark.

### La producción está muy concentrada

De 608 campos del histórico, **294 siguen activos**. En 2025 producen 760 798 bpd
y la concentración es extrema:

- **10 campos = 51 %** de la producción nacional
- 50 campos = 84 %
- **Ecopetrol opera el 60.4 %**; el HHI por operadora es **4 031** (por encima de
  2 500 se considera un mercado altamente concentrado)
- **Meta aporta el 56.6 %**, seguido de Casanare (14.5 %) y Arauca (6.6 %)

### Cuatro segmentos de comportamiento

Agrupamiento *k*-medias sobre cuatro descriptores calculables solo con datos
pasados (tamaño en log, declinación, volatilidad, madurez):

| Segmento | Campos | Activos | Declinación | Volatilidad | Madurez | % producción actual |
|---|---|---|---|---|---|---|
| **Núcleo estable** | 99 | 83 | 1.5 % | 0.18 | 0.64 | **88.2 %** |
| Maduro en declinación | 179 | 135 | 8.5 % | 0.20 | 0.23 | 10.0 % |
| Marginal errático | 101 | 41 | 11.8 % | 0.75 | 0.09 | 1.2 % |
| En agotamiento | 77 | 35 | 47.4 % | 0.51 | 0.04 | 0.6 % |

**83 campos activos producen el 88 % del crudo del país.** Son estables, poco
volátiles y de declinación lenta: el segmento donde un buen pronóstico tiene
valor económico real y donde la Fase 4 debe concentrar el esfuerzo.

La silueta media es 0.26 (k = 4), modesta: los segmentos son **interpretables
pero no netamente separados**, algo esperable en un fenómeno continuo. Se
reportan como una partición útil, no como clases naturales.

### Dos defectos de datos detectados y corregidos

1. **Noviembre de 2025 es un mes de publicación incompleta**: solo 93 campos
   reportan frente a ~300 habituales, y el agregado nacional cae a 184 000 bpd.
   No es una caída real —los campos que sí reportaron traen valores normales—,
   faltan las filas de los demás. Se detecta automáticamente comparando la
   cobertura mensual contra una mediana móvil y se excluye de toda serie
   agregada.
2. **El 36 % de los campos del histórico ya no reporta.** Calcular cuotas de
   producción sobre la media histórica de todos los campos sobrerrepresenta a
   los cerrados: el núcleo estable pasaba de un 88 % real a un 75 % aparente.
   Las cuotas se calculan sobre producción actual de campos activos.

### Figuras

Generadas en `reports/figures/` por `oilai eda`:

| Figura | Contenido |
|---|---|
| `01_produccion_nacional.png` | Producción del país 2014–2026, con el mínimo COVID anotado |
| `02_concentracion.png` | Curva acumulada: 10 campos explican la mitad de la producción |
| `03_volatilidad_vs_tamano.png` | Por qué los campos pequeños no son pronosticables |
| `04_segmentos.png` | Los cuatro segmentos en el plano madurez–declinación |
| `05_mapa_campos.png` | Distribución geográfica, área ∝ producción |
| `06_curvas_ejemplo.png` | Un campo representativo de cada segmento |

## Resultados de la Fase 3

Primer modelo propio: **un único modelo de gradient boosting entrenado sobre los
452 campos a la vez**, en lugar de una curva ajustada campo por campo.

### Cambio de protocolo

La Fase 1 situaba los orígenes de pronóstico por **posición** en la serie de cada
campo. Eso no sirve para un modelo global: entrenar con todos los campos a la vez
exige un corte temporal único, o el modelo aprendería del futuro de un campo para
predecir el pasado de otro.

La Fase 3 introduce **cortes de calendario comunes** —marzo de 2023, 2024 y
2025— y reevalúa bajo ellos todos los modelos, incluidas las líneas base. Regla
de entrenamiento: solo entran muestras cuyo objetivo **ya había ocurrido** en la
fecha de corte. Filtrar por el origen no basta; una muestra con origen anterior
al corte pero objetivo posterior contiene justamente el dato que se quiere
predecir.

> ⚠️ Los valores de MASE de la Fase 3 **no son comparables** con los de la Fase 1:
> cambian los orígenes, los campos evaluados y la definición del horizonte. Solo
> son comparables entre sí, dentro de esta tabla.

### El modelo global gana en las cuatro métricas

61 458 predicciones sobre 336 campos, 3 cortes, horizontes de 1 a 12 meses:

| Modelo | MAE (bpd) | RMSE (bpd) | sMAPE (%) | **MASE** | Sesgo (bpd) |
|---|---|---|---|---|---|
| **ML-global** | **262.9** | **871.4** | **20.5** | **1.595** | −45.6 |
| Naive | 268.8 | 885.0 | 21.8 | 1.660 | +30.5 |
| Media móvil 3m | 283.8 | 976.2 | 21.7 | 1.702 | +28.0 |
| Drift | 296.2 | 961.9 | 31.6 | 1.864 | +8.2 |
| Arps 24m | 328.4 | 1463.8 | 24.6 | 1.939 | −147.7 |
| Arps 36m | 391.1 | 1764.0 | 26.1 | 2.154 | −187.1 |

La mejora sobre el pronóstico ingenuo es del **3.9 % en MASE**. Es real pero
modesta, y conviene decirlo así.

### Dónde gana y dónde no

| Horizonte | h=1 | h=3 | h=6 | h=9 | h=12 |
|---|---|---|---|---|---|
| ML-global | 0.908 | **1.261** | **1.633** | **1.936** | **2.109** |
| Naive | **0.831** | 1.264 | 1.715 | 2.060 | 2.263 |
| **Ventaja del ML** | −9.3 % | +0.3 % | +4.8 % | +6.0 % | **+6.8 %** |

**El cruce está en h = 3.** A un mes, repetir el último valor sigue siendo mejor;
a partir del tercer mes el modelo global domina y la ventaja crece con el
horizonte. Es el patrón esperable: a corto plazo la persistencia lo explica casi
todo, y solo a plazo largo hay estructura que aprender.

Por tamaño de campo (todos los horizontes):

| Modelo | < 500 bpd | 0.5–5k | 5–50k | > 50k |
|---|---|---|---|---|
| ML-global | **1.160** | 2.422 | **1.872** | 1.112 |
| Naive | 1.274 | **2.363** | 2.049 | **0.958** |
| Arps 24m | 1.331 | 2.975 | 2.755 | 1.003 |

El modelo global gana con claridad en los campos pequeños y medianos. **En los
campos de más de 50 000 bpd sigue ganando el Naive, con Arps muy cerca** — pero
esa columna solo contiene 2 campos, así que no soporta una conclusión firme.

### Qué mira el modelo

| Variable | Ganancia |
|---|---|
| **operadora** | 15.3 % |
| `rel_lag0` (mes del origen / ancla) | 9.9 % |
| `log_ancla` (escala del campo) | 9.2 % |
| `h` (horizonte) | 8.2 % |
| `rel_maximo` (madurez causal) | 7.2 % |
| `volatilidad12` | 6.8 % |

Que la operadora sea la variable más importante es un resultado interesante —las
prácticas operativas dejan huella medible en la trayectoria de producción— pero
hay que leerlo con cautela: en operadoras con un solo campo, la variable funciona
en parte como identificador del campo.

Durante el desarrollo detecté que faltaba `rel_lag0`: los rezagos empezaban en el
mes anterior, y el mes del propio origen solo entraba difuminado dentro del ancla.
Añadirlo movió el cruce de h = 6 a **h = 3** y bajó el MASE de 1.639 a 1.595.

### Figuras

| Figura | Contenido |
|---|---|
| `07_ml_vs_referencias.png` | MASE por horizonte y ventaja sobre el Naive |
| `08_importancia_variables.png` | Ganancia aportada por cada variable |

## Resultados de la Fase 4

El aporte central de la tesis. Las fases anteriores establecieron que **ningún
modelo gana en todos los regímenes**: el pronóstico ingenuo domina a un mes, el
modelo global a partir del tercero, Arps se defiende en los campos grandes. La
Fase 4 prueba **dos maneras de combinarlos**, deliberadamente opuestas en
filosofía.

**Vía A — Arps como variable (híbrido implícito).** Se calcula el pronóstico de
Arps para cada campo, origen y horizonte y se entrega al modelo global como una
variable más. El modelo decide solo cuándo confiar en la física. Flexible, pero
la combinación queda enterrada en los árboles.

**Vía B — Combinación convexa por régimen (híbrido explícito).** Se estiman pesos
que suman uno sobre los tres modelos base, por separado para cada horizonte y
clase de campo. Menos flexible, pero cada peso es legible: dice literalmente
cuánto confiar en cada modelo en cada situación.

Los pesos se estiman en un origen de validación **doce meses anterior** al corte
de evaluación. Usar el propio corte y luego medir sobre él sería hacer trampa.

### El híbrido explícito gana

| Modelo | MAE (bpd) | sMAPE (%) | **MASE** | Sesgo (bpd) |
|---|---|---|---|---|
| **Híbrido-régimen** | **260.7** | **20.1** | **1.551** | **−13.2** |
| Híbrido-Arps-ML | 265.8 | 20.4 | 1.591 | −49.9 |
| ML-global (Fase 3) | 262.9 | 20.5 | 1.595 | −45.6 |
| Naive | 268.8 | 21.8 | 1.660 | +30.5 |
| Arps 24m | 328.4 | 24.6 | 1.939 | −147.7 |

**+6.6 % frente al pronóstico ingenuo** y **+2.8 % frente al modelo global**. El
sesgo cae de −45.6 a −13.2 bpd: la combinación cancela los sesgos opuestos de sus
componentes (Naive sobreestima, Arps subestima con fuerza).

### El resultado contraintuitivo

**La vía A apenas aporta nada**: 1.591 frente a 1.595 del modelo global, una
mejora del 0.25 %. Inyectar el pronóstico de Arps como variable no añade
información que el modelo no tuviera ya —los rezagos y las pendientes ya
describen la trayectoria—, así que el modelo lo ignora casi por completo.

**El híbrido explícito, más rígido, gana al implícito, más flexible.** No explota
información nueva sino **diversidad de errores**: promediar modelos que se
equivocan en direcciones opuestas reduce el error aunque ninguno mejore por su
cuenta. Es un resultado que merece defenderse como tal.

### Es el único modelo que gana en todos los regímenes

Por horizonte:

| | h=1 | h=3 | h=6 | h=12 |
|---|---|---|---|---|
| **Híbrido-régimen** | **0.823** | **1.195** | **1.608** | **2.078** |
| ML-global | 0.908 | 1.261 | 1.633 | 2.109 |
| Naive | 0.831 | 1.264 | 1.715 | 2.263 |

Gana **también a un mes**, donde el modelo global perdía un 9.3 % contra el
Naive. Por clase de campo:

| Modelo | < 500 bpd | 0.5–5k | 5–50k | > 50k |
|---|---|---|---|---|
| **Híbrido-régimen** | **1.209** | **2.050** | 2.019 | **0.891** |
| ML-global | 1.247 | 2.128 | **1.971** | 1.112 |
| Naive | 1.348 | 2.118 | 2.086 | 0.958 |
| Arps 24m | 1.425 | 2.684 | 2.666 | 1.003 |

Gana en tres de las cuatro clases y empata prácticamente en la cuarta. **Ningún
modelo puro lo consigue.**

### Los pesos aprendidos son el entregable interpretable

Promedio de los tres cortes:

| Horizonte | Naive | Arps 24m | ML-global |
|---|---|---|---|
| 1 mes | **0.72** | 0.03 | 0.25 |
| 3 meses | 0.53 | 0.15 | 0.32 |
| 6 meses | 0.33 | 0.20 | 0.47 |
| 12 meses | 0.28 | 0.22 | **0.50** |

La rotación es continua y no se le impuso al modelo: **la persistencia pierde
peso de 0.72 a 0.28 mientras el aprendizaje sube de 0.25 a 0.50**, y la física
aparece a partir del segundo mes. Es la confirmación cuantitativa, derivada de
los datos, de lo que las fases 1 a 3 habían mostrado por separado.

Para un ingeniero de producción esta tabla es directamente accionable: dice qué
método usar según el plazo que necesite.

### Figuras

| Figura | Contenido |
|---|---|
| `09_hibrido_vs_modelos.png` | MASE por horizonte del híbrido frente a los modelos puros |
| `10_pesos_hibrido.png` | Los pesos aprendidos, de la persistencia al aprendizaje |

## Instalación

```bash
git clone git@github.com:Locotomate21/DeepWell.git
cd DeepWell
python -m pip install -e ".[dev]"
```

Las fases 3 en adelante necesitan además los paquetes de modelado:

```bash
python -m pip install -e ".[modelos]"
```

## Uso

```bash
oilai all              # pipeline completo: de la descarga a los modelos híbridos
oilai ingest           # solo descarga el histórico de la ANH
oilai panel            # solo reconstruye el panel limpio
oilai benchmark        # solo corre el backtesting
oilai eda              # caracteriza, segmenta y regenera las figuras
oilai ml               # entrena y evalúa el modelo global
oilai hibrido          # combina física y aprendizaje, y compara las dos vías
oilai all --force      # ignora la caché y recalcula todo
```

Cada etapa cachea su salida en Parquet, así que repetir `oilai all` solo
recalcula lo que falte.

Pruebas:

```bash
python -m pytest    # 124 pruebas
```

## Estructura

```
src/oilai/
├── config.py           rutas y definición de la fuente de datos
├── ingest.py           descarga paginada desde la API Socrata
├── clean.py            panel mensual: agregación, bpd, edad productiva
├── evaluate.py         backtesting walk-forward y métricas
├── eda.py              caracterización de campos y agregados nacionales
├── segmentacion.py     agrupamiento de campos por comportamiento
├── features.py         conjunto supervisado causal para el modelo global
├── backtest_global.py  protocolo de evaluación con cortes de calendario
├── backtest_hibrido.py evaluación de los dos híbridos
├── figuras.py          figuras del análisis y del modelo
├── pipeline.py         orquestador (comando `oilai`)
├── run_benchmark.py    benchmark sobre todos los campos
└── models/
    ├── arps.py         declinación exponencial, hiperbólica y armónica
    ├── baselines.py    Naive, media móvil, drift, Arps
    ├── global_ml.py    modelo global de gradient boosting
    └── hibrido.py      Arps como variable y combinación por régimen

tests/                  124 pruebas, incluidas las de ausencia de fuga temporal
docs/metodologia.md     decisiones metodológicas y su justificación
data/raw/               snapshot versionado de los datos de la ANH
reports/figures/        figuras generadas por `oilai eda`
```

## Hoja de ruta

| Fase | Contenido | Estado |
|---|---|---|
| **1. Fundación de datos y línea base** | Ingesta, panel, Arps, baselines, backtesting, pruebas | ✅ **completa** |
| **2. Análisis exploratorio** | Caracterización de los 608 campos, figuras, mapa, segmentación | ✅ **completa** |
| **3. Modelo global de ML** | Features + LightGBM multi-horizonte sobre todos los campos | ✅ **completa** |
| **4. Modelo híbrido física + ML** | Arps como variable y combinación por régimen — aporte original | ✅ **completa** |
| 5. Incertidumbre y anomalías | Intervalos de predicción, detección de caídas atípicas | pendiente |
| 6. Dashboard y entregable | App Streamlit e informe metodológico | pendiente |

## Fuentes

- Arps, J. J. (1945). *Analysis of Decline Curves*. Transactions of the AIME, 160(01), 228–247.
- Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of forecast accuracy*. International Journal of Forecasting, 22(4), 679–688. — origen de MASE.
- Agencia Nacional de Hidrocarburos. *Producción Fiscalizada de Crudo Consolidada*. Datos Abiertos Colombia.

## Licencia

MIT.
