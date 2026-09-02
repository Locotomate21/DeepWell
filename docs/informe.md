# DeepWell · Informe final

Pronóstico de producción de los campos petroleros de Colombia mediante modelos
híbridos de física y aprendizaje automático.

> Este documento sintetiza qué se hizo, qué se encontró y qué queda abierto.
> Las decisiones metodológicas y su justificación están en
> [`metodologia.md`](metodologia.md); las cifras completas y el código, en el
> repositorio.

---

## 1. Resumen ejecutivo

Se construyó y evaluó un sistema de pronóstico de producción sobre **50 103
observaciones mensuales de 608 campos petroleros colombianos** (2014–2026),
tomadas de la producción fiscalizada que publica la Agencia Nacional de
Hidrocarburos.

El resultado principal es que **ningún modelo domina en todos los regímenes**, y
que una combinación explícita de tres de ellos —ponderada por horizonte y tamaño
de campo— supera a cada uno por separado:

| Modelo | MAE (bpd) | sMAPE | MASE | vs. ingenuo |
|---|---|---|---|---|
| **Híbrido por régimen** | **260.7** | **20.1 %** | **1.551** | **+6.6 %** |
| Híbrido con Arps como variable | 265.8 | 20.4 % | 1.591 | +4.1 % |
| Modelo global de aprendizaje | 262.9 | 20.5 % | 1.595 | +3.9 % |
| Pronóstico ingenuo | 268.8 | 21.8 % | 1.660 | — |
| Curvas de declinación de Arps | 328.4 | 24.6 % | 1.939 | −16.8 % |

*51 215 predicciones, 336 campos, 3 cortes de calendario.*

Además se entregan **intervalos de predicción calibrados** —cobertura observada
del 79.9 % frente a un 80 % nominal— y un **detector de anomalías validado**: una
alerta multiplica por doce el riesgo de que la producción del campo caiga más de
la mitad en los seis meses siguientes.

---

## 2. Pregunta de investigación

> ¿Bajo qué condiciones un modelo de aprendizaje automático supera al análisis de
> curvas de declinación clásico, y puede un modelo híbrido combinar las ventajas
> de ambos?

**La respuesta tiene dos partes.**

Sobre la primera: el aprendizaje automático supera a Arps de forma clara y
consistente —un 17.7 % menos de MASE en el agregado— pero **no supera al
pronóstico ingenuo a horizontes cortos**. A un mes pierde un 9.3 %; el cruce
ocurre en el tercer mes y a doce la ventaja es del 6.8 %. La estructura que el
modelo puede aprender solo se manifiesta cuando el horizonte da tiempo a que la
declinación y la madurez del campo se expresen.

Sobre la segunda: **sí, y de una forma concreta**. El híbrido que gana no es el
que da más información al modelo, sino el que combina modelos ya existentes
explotando la diversidad de sus errores.

---

## 3. Datos

**Producción Fiscalizada de Crudo Consolidada** (ANH, identificador `fdvb-hsrf`),
publicada en el portal de Datos Abiertos de Colombia. *Fiscalizada* significa que
el volumen fue medido y certificado para efectos de regalías: es la cifra oficial
del Estado, no una estimación del operador.

| | |
|---|---|
| Registros crudos | 53 237 |
| Panel limpio (campo × mes) | 50 103 |
| Campos | 608 (294 activos) |
| Operadoras | 94 |
| Cobertura | enero 2014 – marzo 2026 |

**Validación externa.** Agregado a nivel nacional, el panel da ≈ 990 000 bpd en
2014 y ≈ 736 000 bpd en 2021, cifras que coinciden con las estadísticas oficiales
de producción del país.

**Dos defectos de datos detectados durante el trabajo:**

1. **Noviembre de 2025 es un mes de publicación incompleta.** Solo 93 campos
   reportan frente a ~300 habituales, y el agregado nacional cae a 184 000 bpd.
   No es una caída real: los campos que sí reportaron traen valores normales,
   faltan las filas de los demás. Se detectó al graficar la serie nacional y ver
   un desplome imposible. Ahora se identifica de forma automática comparando la
   cobertura mensual contra una mediana móvil.
2. **El 36 % de los campos del histórico ya no reporta.** Calcular cuotas de
   producción sobre la media histórica de todos ellos sobrerrepresenta a los
   cerrados: el segmento principal aparecía con un 75 % de la producción cuando
   su cuota real sobre producción actual es del 88 %.

---

## 4. Método

**Referencia obligada.** El estándar de la industria son las curvas de
declinación de Arps (1945). Cualquier propuesta basada en IA debe demostrar que
las supera, así que Arps se implementó en sus tres formas y se usó como línea
base junto al pronóstico ingenuo, la media móvil y la deriva.

**Protocolo de evaluación.** Backtesting con **cortes de calendario comunes**
(marzo de 2023, 2024 y 2025): todos los modelos se evalúan sobre los mismos
campos, orígenes y horizontes. La regla que sostiene la validez es que solo
entran al entrenamiento las muestras cuyo **objetivo ya había ocurrido** en la
fecha de corte; filtrar por el origen dejaría pasar precisamente el dato que se
quiere predecir.

**Métrica de decisión: MASE**, que escala el error por el del pronóstico ingenuo
sobre el propio historial de cada campo. Es lo que permite comparar campos de
120 000 bpd con campos de 50 bpd sin que los primeros dominen ni los segundos
hagan explotar el error relativo.

**Verificación de ausencia de fuga temporal.** Las 21 variables del modelo se
construyen dos veces —con la serie completa y truncada en el origen— y una prueba
exige que ninguna cambie. Sin esa garantía, todas las métricas serían inválidas.

---

## 5. Resultados por fase

### Fase 1 · Línea base

106 428 predicciones sobre 425 campos. **El pronóstico ingenuo gana en el
agregado** (MASE 2.139) y Arps queda por detrás (2.350). Pero la desagregación
invierte la conclusión: a doce meses y en los campos de más de 50 000 bpd, Arps
reduce el error un 29 % frente al ingenuo. El cruce entre régimen estadístico y
régimen físico quedó identificado desde aquí.

### Fase 2 · Análisis exploratorio

La volatilidad mensual cae de 0.63 en el decil de campos más pequeños a 0.11 en
el más grande: **buena parte del error en campos pequeños es irreducible**, no un
defecto del modelo. La producción está muy concentrada —10 campos explican el
51 %, Ecopetrol opera el 60.4 %— y una segmentación en cuatro grupos muestra que
**83 campos activos producen el 88 % del crudo del país**.

### Fase 3 · Modelo global

Un único modelo de gradient boosting entrenado con los 452 campos a la vez, sobre
un objetivo adimensional —el cambio logarítmico respecto a un ancla reciente— que
evita que los campos grandes acaparen la capacidad del modelo. Mejora un 3.9 % en
MASE sobre el pronóstico ingenuo, con el cruce en el tercer horizonte.

### Fase 4 · Modelos híbridos

Se probaron **dos filosofías opuestas**: dar el pronóstico de Arps al modelo como
una variable más (flexible, opaco), o combinar los modelos con pesos convexos
estimados por horizonte y tamaño de campo (rígido, legible). Ganó el segundo.

### Fase 5 · Incertidumbre y anomalías

Tres métodos de intervalos comparados sobre 102 672 observaciones. La conformal
condicionada por clase resultó la única a la vez calibrada en agregado (79.9 %
frente a 80 % nominal) y razonable dentro de cada subpoblación. El detector de
anomalías se validó de forma indirecta y mostró valor operativo real.

### Fase 6 · Tablero

Aplicación con cuatro vistas —mapa, ficha de campo, alertas priorizadas y
comparador de modelos— construida sobre los pronósticos fuera de muestra ya
producidos, sin reentrenar nada.

---

## 6. Los tres hallazgos que sostienen la tesis

### 6.1 El modelo ganador depende del régimen, y los pesos lo demuestran

Los pesos que el híbrido aprende, promediados sobre los tres cortes:

| Horizonte | Ingenuo | Arps | Aprendizaje |
|---|---|---|---|
| 1 mes | **0.72** | 0.03 | 0.25 |
| 6 meses | 0.33 | 0.20 | 0.47 |
| 12 meses | 0.28 | 0.22 | **0.50** |

La rotación de la persistencia hacia el aprendizaje **no se impuso**: emerge de
minimizar el error en un periodo de validación anterior al corte. Que reproduzca
lo que las fases 1 a 3 habían encontrado por separado es una validación cruzada
de todo el trabajo previo.

Para una operadora, esa tabla es directamente accionable: dice qué método usar
según el plazo de planeación.

### 6.2 La combinación explícita gana a la implícita, y por una razón concreta

Cabía esperar lo contrario. Dar Arps al modelo como variable es estrictamente más
flexible: puede representar cualquier combinación que los pesos expresen, y
muchas más. Sin embargo apenas mejora (1.591 frente a 1.595 del modelo solo).

La explicación tiene dos partes que conviene separar:

1. **Arps no aporta información nueva.** Los rezagos, las pendientes y la
   volatilidad ya describen la trayectoria reciente, que es exactamente lo que
   Arps resume en tres parámetros. El modelo lo recibe y lo ignora.
2. **La combinación no explota información, sino diversidad de errores.** Sus
   componentes se equivocan en direcciones opuestas —el ingenuo sobreestima en
   +30 bpd, Arps subestima en −148— y promediarlos cancela el sesgo, que cae a
   −13 bpd. Ese mecanismo es inaccesible para un modelo único por flexible que
   sea, porque no consiste en aprender mejor sino en **agregar estimadores
   sesgados de forma complementaria**.

La conclusión metodológica es que, en este problema, la ganancia no estaba en
darle más información al modelo sino en **cómo se agregan modelos que ya
existen**.

### 6.3 Una cobertura agregada correcta puede ocultar dos errores opuestos

La predicción conformal marginal calibra casi a la perfección en agregado:
80.08 % observado frente a 80 % nominal, un desvío de 0.08 puntos. Parecía
resuelto. Al mirar **dentro** de cada clase de campo:

| Método | < 500 bpd | 0.5–5k | 5–50k | > 50k |
|---|---|---|---|---|
| Conformal | 76.1 | 83.6 | 92.8 | 99.7 |
| **Conformal por clase** | **79.6** | **80.0** | **81.9** | 99.7 |
| Cuantílica | 73.6 | 73.9 | 72.5 | 69.3 |

Ese 80 % global era la media de dos errores de signo contrario: sobrecubría los
campos grandes al 99.7 %, con intervalos de 44 431 bpd de ancho, y subcubría los
pequeños. Calibrar por clase lo corrige donde hay datos suficientes.

**Ningún método domina.** La regresión cuantílica obtiene mejor puntuación de
Winkler porque adapta la anchura al campo, pero promete un 80 % y entrega un
73.5 %, degradándose hasta el 70 % a doce meses. Para planeación, un intervalo
que se queda corto de forma sistemática es peor que uno algo ancho: induce
confianza injustificada.

---

## 7. Qué se entrega

| Componente | Descripción |
|---|---|
| Pipeline reproducible | `oilai all` reconstruye todo desde el snapshot versionado de la ANH |
| Modelo híbrido | Combinación convexa por régimen, con pesos legibles |
| Intervalos calibrados | Conformal condicionada por clase, nivel 80 % |
| Detector de anomalías | Lista priorizada por barriles perdidos, validada |
| Tablero | Aplicación con mapa, ficha de campo, alertas y comparador |
| 194 pruebas automatizadas | Incluidas las de causalidad y ausencia de fuga temporal |
| 13 figuras | Listas para insertar en el documento |

**Valor operativo de las alertas.** Sobre 10 301 observaciones monitoreadas entre
2023 y 2026:

| | Tras una alerta | Sin alerta |
|---|---|---|
| Producción posterior / previa (mediana) | 0.859 | 0.944 |
| Cae más del 20 % | **40.8 %** | 14.8 % |
| Cae más del 50 % | **12.7 %** | 1.0 % |

---

## 8. Limitaciones

Se declaran las que afectan la interpretación de los resultados. La lista
completa, con su contexto técnico, está en
[`metodologia.md` § 9](metodologia.md).

1. **Granularidad de campo, no de pozo.** La ANH no publica producción por pozo
   en un recurso abierto. Las curvas agregadas mezclan pozos en distintas etapas
   de su vida, y los resultados no son extrapolables a pozo individual sin
   validación adicional.
2. **Sin variables de intervención.** No se dispone de fechas de workover,
   perforación de infill ni cambios de sistema de levantamiento. Parte del error
   irreducible corresponde a esos eventos no observados.
3. **Sin variables de precio ni de decisión comercial.** Los recortes de 2020
   responden al precio del crudo y a decisiones de la OPEP+, no a la geología.
   Ningún modelo del benchmark puede anticiparlos.
4. **La mejora es modesta en términos absolutos.** Un MASE de 1.551 sigue por
   encima de 1: el pronóstico ingenuo de un paso continúa siendo un rival duro.
5. **Solo tres cortes de calendario**, y los pesos del híbrido se estiman sobre
   un único origen de validación por corte.
6. **La clase de campos mayores a 50 000 bpd contiene solo dos campos.**
   Cualquier conclusión sobre ella —incluida su descalibración de intervalos— es
   frágil por construcción.
7. **Hiperparámetros sin optimizar.** Se fijaron valores razonables y la parada
   temprana determinó el número de árboles.
8. **La validación de anomalías es indirecta.** Demuestra que la alerta
   correlaciona con caídas posteriores, no que identifique correctamente sus
   causas: no existe un registro público de intervenciones con el que contrastar.

---

## 9. Trabajo futuro

En orden de relación entre esfuerzo y beneficio esperado:

1. **Combinación con pesos dependientes de la calidad del ajuste.** La actual es
   lineal y convexa; no puede expresar reglas del tipo «usar Arps solo si su
   ajuste fue bueno en este campo». Es la extensión natural del hallazgo 6.2.
2. **Más cortes de calendario y conformal cruzada.** Ambos usarían los datos de
   forma más eficiente y darían estimaciones de error más estables.
3. **Búsqueda de hiperparámetros** sobre la partición de validación temporal,
   nunca sobre los cortes de evaluación.
4. **Incorporar variables exógenas**: precio del crudo, y en un entorno
   corporativo, el registro de intervenciones. Es lo que más podría reducir el
   error irreducible identificado en la Fase 2.
5. **Envolver el híbrido con los intervalos**, en lugar del modelo global. El
   procedimiento es aplicable sin cambios pero multiplica los entrenamientos.
6. **Validar las alertas contra un registro real de eventos**, que convertiría la
   validación indirecta en una medición de precisión y exhaustividad.

---

## 10. Conclusión

El trabajo responde su pregunta con una respuesta matizada, que es la que los
datos permiten sostener: **el aprendizaje automático supera al análisis de
declinación clásico, pero no de forma uniforme**, y la mejor estrategia no es
elegir un modelo sino combinarlos según el régimen.

Tres resultados merecen destacarse por encima del número final:

- El modelo ganador **cambia con el horizonte y el tamaño del campo**, y los
  pesos aprendidos lo cuantifican de forma legible.
- La ganancia vino de **cómo se agregan modelos existentes**, no de darle más
  información a uno solo — un resultado contraintuitivo que se sostiene con una
  explicación mecánica, no anecdótica.
- Una métrica agregada correcta **puede ocultar errores compensados**, y solo la
  evaluación condicional los revela.

Los tres son negativos o matizados en algún sentido, y ninguno se ocultó. Esa es
la contribución que un trabajo de este tipo puede hacer con honestidad: no un
modelo que gana siempre, sino un mapa fiable de dónde gana cada cosa y cuánto.

---

## Referencias

- Arps, J. J. (1945). *Analysis of Decline Curves*. Transactions of the AIME,
  160(01), 228–247.
- Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of forecast
  accuracy*. International Journal of Forecasting, 22(4), 679–688.
- Winkler, R. L. (1972). *A Decision-Theoretic Approach to Interval Estimation*.
  Journal of the American Statistical Association, 67(337), 187–191.
- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random
  World*. Springer. — predicción conformal.
- Agencia Nacional de Hidrocarburos. *Producción Fiscalizada de Crudo
  Consolidada* [conjunto de datos]. Datos Abiertos Colombia.
  <https://www.datos.gov.co/d/fdvb-hsrf>
