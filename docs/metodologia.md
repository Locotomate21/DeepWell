# Metodología

Documento de decisiones metodológicas y su justificación. Cada decisión que
afecta los resultados queda registrada aquí, junto con la alternativa que se
descartó y el motivo.

---

## 1. Fuente de datos

Se usa **Producción Fiscalizada de Crudo Consolidada** de la Agencia Nacional de
Hidrocarburos (ANH), publicada en Datos Abiertos Colombia con identificador
`fdvb-hsrf`.

*Fiscalizada* significa que el volumen fue medido y certificado para efectos de
regalías. Es la cifra oficial, no una estimación del operador, lo que la hace
apropiada como variable objetivo de un trabajo académico.

**Granularidad: campo, no pozo.** La ANH no publica producción por pozo en un
recurso abierto. Se trabaja entonces a nivel de campo, lo que tiene una
implicación que debe declararse: la producción de un campo es la suma de pozos
que entran y salen de operación, de modo que las curvas de declinación
observadas son *agregadas* y pueden mostrar escalones que un pozo individual no
tendría (perforación de infill, workovers, cierres parciales).

**Snapshot versionado.** El archivo `data/raw/anh_produccion_cruda.parquet` se
versiona en el repositorio. La ANH revisa cifras retroactivamente, así que sin
un snapshot fijo los resultados de la tesis no serían reproducibles. Los
archivos derivados (panel y backtest) sí se ignoran, porque se regeneran de
forma determinista con `oilai all`.

---

## 2. Limpieza y construcción del panel

### 2.1 Agregación municipal

Un campo que se extiende sobre varios municipios aparece en una fila por
municipio. Ejemplo real: AKACIAS en octubre de 2014 aparece con 286 095 bbl en
ACACIAS y 55 bbl en GUAMAL.

**Decisión:** sumar la producción por campo y mes, conservando el municipio de
mayor aporte como atributo representativo.

**Alternativa descartada:** tratar cada par campo–municipio como una serie
independiente. Se descartó porque el reparto municipal responde a límites
administrativos, no a la geología del yacimiento; separarlos fragmentaría la
señal de declinación sin ganancia física.

Esta agregación reduce 53 237 registros a **50 103** filas de panel.

### 2.2 Normalización por días del mes

La producción mensual en barriles no es comparable entre meses de 28 y 31 días:
una caída del 10 % de enero a febrero es en su mayor parte un artefacto del
calendario.

**Decisión:** modelar **bpd** (barriles por día calendario) = barriles del mes ÷
días del mes. Es además la unidad en que la industria habla de producción.

### 2.3 Tratamiento de huecos

Un mes sin reporte puede significar campo cerrado, campo sin producción o
simplemente ausencia de reporte administrativo. Los tres casos son
indistinguibles en el dato.

**Decisión:** no imputar. El panel marca los huecos con `es_hueco` y
`meses_desde_reporte_previo`, y el eje temporal de los modelos es
`meses_desde_inicio`, que respeta la distancia real entre observaciones.

**Alternativa descartada:** rellenar con ceros. Habría introducido caídas
abruptas ficticias que contaminarían el ajuste de declinación, sesgando Di
hacia arriba.

### 2.4 Operadora variable en el tiempo

La operadora de un campo cambia con las cesiones de contrato. Se conserva la
operadora **del mes**, no una única por campo, porque el cambio de operador es
una covariable potencialmente informativa para las fases de modelado: un
operador entrante suele intervenir pozos y alterar la trayectoria de producción.

---

## 3. Modelo físico de referencia

### 3.1 Formulación

Curvas de declinación de Arps (1945):

```
q(t) = qi / (1 + b·Di·t)^(1/b)      hiperbólica   (0 < b < 1)
q(t) = qi · exp(-Di·t)              exponencial   (b = 0)
q(t) = qi / (1 + Di·t)              armónica      (b = 1)
```

con `q` en bpd, `t` en meses, `qi` el caudal inicial y `Di` la declinación
nominal inicial. El caso `b = 0` se implementa de forma explícita para evitar la
indeterminación numérica de `1/b`, y se verifica en pruebas que el límite
`b → 0` es continuo.

### 3.2 Ajuste

Mínimos cuadrados no lineales acotados (`scipy.optimize.curve_fit`, método
trust-region reflective), con `b` en `[0, 1]` y `Di` en `[1e-6, 1]`. La semilla
de `Di` se obtiene de la pendiente de `log q` contra `t`, que es la declinación
exponencial equivalente.

Se ajustan las tres variantes (b fijo en 0, b fijo en 1, y b libre) y se elige
la de menor RMSE de ajuste. Los meses con producción nula o negativa se excluyen
del ajuste: un campo cerrado no aporta información sobre su tasa de declinación.

### 3.3 Correlación entre Di y b

Un hallazgo que debe declararse: `Di` y `b` están fuertemente correlacionados en
el ajuste. En la prueba de recuperación con datos sintéticos (`qi`=15 000,
`Di`=0.030, `b`=0.60 más 4 % de ruido) el ajuste devuelve `qi`=15 045,
`Di`=0.033, `b`=0.77 — reproduce la **curva** con precisión pero no el par
`(Di, b)` original.

**Consecuencia:** los parámetros individuales `Di` y `b` no deben interpretarse
como propiedades físicas del yacimiento sin un análisis de identificabilidad. Lo
que el modelo estima con fiabilidad es la trayectoria, no su descomposición.

### 3.4 Ventana de ajuste

Se evalúan tres variantes: últimos 24 meses, últimos 36 meses y toda la
historia. La ventana importa porque Arps describe la declinación de un *régimen*
de producción, y doce años de historia contienen múltiples regímenes separados
por intervenciones. Los resultados confirman la hipótesis: **Arps-24m supera a
Arps-todo en todos los horizontes y todas las clases de campo**.

---

## 4. Protocolo de evaluación

### 4.1 Walk-forward con origen rodante

Para cada campo se seleccionan 3 orígenes de pronóstico equiespaciados sobre el
tramo donde caben un entrenamiento mínimo y el horizonte completo. En cada
origen el modelo se entrena **exclusivamente** con observaciones anteriores al
origen y pronostica de 1 a 12 meses hacia adelante.

- Entrenamiento mínimo: 24 meses.
- Horizonte: 12 meses.
- Campos elegibles: los que tienen al menos 36 meses de historia, **425 de 608**.
- Total de predicciones evaluadas: **106 428**.

**Verificación de ausencia de fuga temporal.** La prueba
`test_el_backtest_no_ve_el_futuro` inyecta un modelo espía que registra los
instantes que recibe en cada entrenamiento, y verifica que el máximo observado
sea siempre anterior al primer instante evaluado. Sin esta garantía todas las
métricas del proyecto serían inválidas.

### 4.2 Métricas

| Métrica | Uso |
|---|---|
| **MASE** | Métrica de decisión. Escala el error absoluto por el MAE del pronóstico ingenuo de un paso *sobre el tramo de entrenamiento*. Adimensional, comparable entre campos de escalas muy distintas. |
| MAE, RMSE (bpd) | Interpretabilidad operativa: cuántos barriles diarios se erra. RMSE penaliza más los errores grandes. |
| sMAPE | Error relativo acotado. Se prefiere sobre MAPE porque no diverge cuando la producción real tiende a cero, situación frecuente en campos maduros. |
| Sesgo | Error medio con signo. Distingue un modelo que sobreestima sistemáticamente de uno simplemente impreciso, distinción relevante para planeación de producción. |

**Por qué MASE y no MAPE.** El panel contiene campos de 120 000 bpd y campos de
50 bpd. Promediar errores absolutos los haría dominar por los campos grandes;
MAPE, en cambio, explota en los campos pequeños con producción cercana a cero.
MASE resuelve ambos problemas usando como referencia la dificultad intrínseca de
cada serie.

**Denominador nulo.** En series constantes el MAE ingenuo es 0 y MASE sería
infinito. Esos casos se marcan como nulos y se excluyen del promedio, en lugar
de propagar infinitos.

### 4.3 Desagregación

Los errores se reportan **por horizonte** y **por clase de tamaño de campo**.
Promediar sobre horizontes oculta que un modelo puede ser bueno a un mes y malo
a doce; promediar sobre tamaños oculta que el modelo ganador cambia según la
escala del campo. Ambas desagregaciones resultaron decisivas: el ranking global,
donde gana el pronóstico ingenuo, invierte su orden en el horizonte de 12 meses
y en los campos de más de 50 000 bpd.

---

## 5. Análisis exploratorio y segmentación (Fase 2)

El propósito de esta fase no es descriptivo sino explicativo: entender **por qué**
el benchmark de la Fase 1 dio los resultados que dio, y decidir dónde vale la
pena concentrar el modelado.

### 5.1 Descriptores por campo

Cada campo con al menos 24 meses de historia se resume en cuatro descriptores,
todos calculables usando únicamente datos pasados (requisito para que puedan
alimentar un modelo sin fuga temporal):

| Descriptor | Definición | Por qué |
|---|---|---|
| Tamaño | media histórica de bpd, en log₁₀ | Abarca cinco órdenes de magnitud; en escala lineal los campos grandes dominarían cualquier distancia euclídea. |
| Declinación anual | 1 − exp(−Di·12), en % | `Di` de Arps es una tasa nominal instantánea, poco intuitiva. La declinación efectiva anual es como la industria reporta el agotamiento. |
| Volatilidad | desviación estándar de Δlog(q) | Adimensional, comparable entre un campo de 120 000 bpd y uno de 50 bpd. Es el indicador más directo del error irreducible de pronóstico. |
| Madurez | caudal actual / caudal pico | Distingue un campo en meseta (≈1) de uno casi agotado (≈0). |

El ajuste de Arps para la declinación usa la ventana de 36 meses, la misma que
resultó ganadora en la Fase 1.

### 5.2 Campos activos frente a campos históricos

**162 de 456 campos caracterizados (36 %) llevan más de tres meses sin reportar.**
Se marcan con `activo` y el umbral de tres meses absorbe el rezago habitual de
publicación de la ANH sin dar por vivo un campo cerrado hace años.

La distinción no es cosmética. Calcular la cuota de producción de un segmento
sumando la **media histórica** de todos sus campos sobrerrepresenta a los ya
cerrados: el segmento «Núcleo estable» aparecía con un 75 % de la producción
cuando su cuota real sobre producción actual es del **88 %**. Todas las cuotas
del proyecto se calculan sobre producción actual de campos activos.

### 5.3 Detección de meses con publicación incompleta

Noviembre de 2025 aparece en los datos con 93 campos reportando frente a una
mediana de ~300, y un agregado nacional de 184 000 bpd contra los ~750 000
habituales. No es una caída de producción: los campos que sí reportaron ese mes
traen valores normales (Chichimene 35 477 bpd, contra 37 463 el mes anterior);
**faltan las filas** de los demás.

La detección compara el número de campos que reportan cada mes contra una
mediana móvil centrada de 13 meses, que absorbe la tendencia de largo plazo —el
número de campos activos baja con los años— sin dejarse arrastrar por el mes
anómalo. Un mes por debajo del 70 % de esa referencia se marca como incompleto y
se excluye de toda serie agregada.

**Alcance del defecto.** Solo afecta a los agregados. Las series por campo no se
corrompen: los valores publicados son correctos y la ausencia de un mes ya
quedaba registrada como hueco en la Fase 1, con `meses_desde_inicio` respetando
la distancia temporal real. Por eso los resultados del benchmark de la Fase 1 no
requieren recálculo.

### 5.4 Segmentación

Agrupamiento *k*-medias sobre los cuatro descriptores estandarizados.

**Recortes en vez de exclusiones.** La declinación se acota a [0, 60] % anual y
la volatilidad a [0, 2]. No se elimina ningún campo: uno con 200 % de declinación
aparente sigue siendo un campo agotándose y debe clasificarse, pero sin capturar
los centroides. El recorte es visible en las figuras como acumulación en el borde
superior, y así se anota.

**Elección de k.** La silueta media es máxima en k = 2 (0.296) y decrece
suavemente: 0.285 en k = 3, **0.261 en k = 4**, 0.229 en k = 5. Se adopta k = 4
por interpretabilidad operativa —los cuatro grupos corresponden a regímenes que
un ingeniero de producción reconoce— aceptando el costo en silueta.

**Declaración honesta:** una silueta de 0.26 indica grupos **interpretables pero
no netamente separados**. Es lo esperable en un fenómeno continuo: no hay cuatro
clases naturales de campo petrolero, hay un espectro. Los segmentos se reportan
como una partición útil para decidir estrategia de modelado, no como una
taxonomía descubierta en los datos.

**Nombres reproducibles.** Los índices que devuelve *k*-medias son arbitrarios y
cambian entre corridas. Los segmentos se nombran por reglas sobre las medianas
de sus descriptores, de modo que la etiqueta sea auditable y estable:

- declinación ≥ 25 % → «En agotamiento»
- volatilidad ≥ 0.45 → «Marginal errático»
- madurez ≥ 0.45 y declinación < 5 % → «Núcleo estable»
- resto → «Maduro en declinación»

**Campos representativos.** Se elige el campo más cercano al centroide en el
espacio estandarizado, no el de producción mediana: el campo mediano en tamaño
puede ser atípico en declinación o volatilidad y resultaría engañoso como
ilustración. La búsqueda se restringe además a campos activos, porque ilustrar un
segmento con un campo que dejó de reportar en 2021 induce a error.

### 5.5 Diseño de las figuras

La paleta categórica de referencia se validó con el script de verificación
correspondiente. Resultado relevante: con cuatro segmentos, los cuatro colores
**no** superan el piso de discriminación cuando todos los pares coexisten en el
mismo plano —amarillo y naranja quedan en ΔE 13.7, bajo el piso de 15 para visión
normal—. Por eso las figuras que comparan los cuatro segmentos usan *small
multiples*: un segmento por panel, un solo color por panel y el resto de campos
en gris de contexto.

Dos de los colores quedan por debajo de 3:1 de contraste contra el fondo claro,
así que cada panel lleva su nombre como rótulo directo: la identidad nunca
depende solo del color. Ninguna figura usa ejes dobles.

---

## 6. Modelo global de aprendizaje automático (Fase 3)

### 6.1 Por qué un modelo global

Las fases anteriores ajustaban un modelo por campo. Un modelo **global** entrena
un único estimador con las observaciones de los 452 campos a la vez. La hipótesis
es que los campos comparten estructura —cómo declina un campo maduro, cuánto
ruido tiene uno pequeño, qué ocurre tras un cambio de operadora— y que un campo
con historia corta puede beneficiarse de lo aprendido en los demás. Un ajuste por
campo no puede transferir nada entre series.

### 6.2 Rediseño del protocolo de evaluación

La Fase 1 situaba los orígenes de pronóstico por **posición** dentro de la serie
de cada campo. Ese diseño es válido para modelos ajustados campo a campo, pero
inutilizable aquí: si cada campo tiene su propio origen, entrenar un modelo común
significa usar el futuro de un campo para predecir el pasado de otro.

**Decisión:** cortes de calendario comunes en marzo de 2023, 2024 y 2025, y
reevaluación de **todos** los modelos bajo ese protocolo, incluidas las líneas
base de la Fase 1. Así la comparación es directa: mismos campos, mismos orígenes,
mismos horizontes.

**Consecuencia que debe declararse:** los valores de MASE de la Fase 3 **no son
comparables** con los de la Fase 1. Cambian los orígenes, el conjunto de campos
evaluados y la definición del horizonte (meses de calendario en vez de pasos de
observación). Solo son comparables entre sí.

**La regla de entrenamiento.** Solo entran muestras cuyo objetivo ya había
ocurrido en la fecha de corte: `fecha_objetivo <= corte`. Filtrar por el origen
—`origen <= corte`— sería insuficiente y es el error más fácil de cometer: una
muestra con origen en enero y horizonte de seis meses tiene su objetivo en julio,
que en un corte de marzo todavía no ha ocurrido. La prueba
`test_el_filtro_de_entrenamiento_excluye_objetivos_futuros` verifica que el caso
peligroso existe en los datos y que el filtro correcto lo elimina.

### 6.3 Objetivo adimensional

La producción abarca cinco órdenes de magnitud. Un modelo entrenado sobre
barriles dedicaría toda su capacidad a los campos grandes y trataría a los
pequeños como ruido.

**Decisión:** predecir `log(q_{t+h} / ancla_t)`, el cambio logarítmico respecto a
un ancla reciente. Un campo que cae un 10 % aporta la misma señal produzca 50 o
50 000 bpd. La reconstrucción a barriles es directa:
`q̂ = ancla · exp(ŷ)`.

**El ancla es la mediana de los últimos tres meses observados**, no el último
valor: en campos ruidosos un solo mes atípico desplazaría todas las predicciones
del campo. La prueba `test_el_ancla_es_la_mediana_reciente_y_no_el_ultimo_valor`
fija esa decisión.

**Recorte del objetivo** a `[-3, 1.5]` en escala logarítmica, es decir entre el
5 % y 4.5 veces el ancla. Cubre cualquier transición plausible en doce meses y
evita que un puñado de reactivaciones extremas domine la función de pérdida.

### 6.4 Variables

Veintiuna variables, todas calculadas con operaciones que solo miran hacia atrás
(`shift`, `rolling`, `cummax`, `cumsum`), nunca con estadísticos de la serie
completa:

- **Escala:** `log_ancla`. Permite al modelo aprender que los campos pequeños son
  más ruidosos sin que la escala contamine el objetivo.
- **Historia reciente:** rezagos 0, 1, 2, 3, 6 y 12, todos relativos al ancla.
- **Tendencia:** pendiente media de `log q` en ventanas de 3, 6 y 12 meses.
- **Volatilidad:** desviación estándar de `Δlog q` en 6 y 12 meses.
- **Madurez causal:** ancla contra el máximo visto **hasta ese momento**. Usar el
  máximo de toda la serie sería mirar al futuro.
- **Reporte:** meses observados, edad, fracción de huecos, meses sin reportar.
- **Contexto:** operadora y departamento como categóricas nativas de LightGBM
  (una codificación por objetivo sería otra vía de fuga), horizonte y mes.

**Densificación.** Las series se reindexan a una malla mensual continua con
huecos explícitos, para que `shift(12)` signifique «hace doce meses» y no «doce
reportes atrás», que en un campo con huecos es otra cosa. Los huecos quedan como
nulos, que LightGBM maneja de forma nativa: un hueco es información, no un dato
que haya que inventar.

**El rezago 0.** La primera versión empezaba los rezagos en el mes anterior, con
el mes del propio origen entrando solo difuminado dentro del ancla. Añadirlo
—el dato está disponible, porque solo se pronostica desde meses con reporte—
adelantó el cruce con el pronóstico ingenuo de h = 6 a **h = 3** y bajó el MASE
global de 1.639 a 1.595.

**Verificación de causalidad.** `test_las_variables_no_miran_al_futuro` construye
las variables dos veces —con la serie completa y truncada en el origen— y exige
que ninguna cambie.

### 6.5 Modelo

LightGBM con:

- **Pérdida L1.** La evaluación usa MAE y MASE, ambas de error absoluto. L2
  optimizaría la media condicional y penalizaría en exceso los meses atípicos,
  frecuentes en campos marginales; L1 optimiza la mediana condicional, que es lo
  que se está midiendo.
- **Un modelo para los doce horizontes**, con `h` como variable. Doce modelos
  independientes multiplicarían el costo y fragmentarían los datos sin aportar:
  la relación entre variables y objetivo cambia de forma suave con el horizonte.
- **Parada temprana sobre partición temporal**, los últimos doce meses del tramo
  de entrenamiento. Una partición aleatoria pondría meses futuros del mismo campo
  en validación y daría una estimación optimista del error.

Entrenamiento por corte: entre 291 000 y 376 000 muestras, 418–442 campos.

### 6.6 Resultados y lectura honesta

El modelo global gana en las cuatro métricas, con una mejora del **3.9 % en MASE**
sobre el pronóstico ingenuo (1.595 frente a 1.660). Es una ganancia real pero
modesta, y así debe presentarse.

El comportamiento por horizonte es el hallazgo aprovechable: **el modelo pierde
un 9.3 % a un mes y gana un 6.8 % a doce**, con el cruce en h = 3. A corto plazo
la persistencia explica casi todo y no hay estructura que aprender; la ventaja
del modelo aparece cuando el horizonte da tiempo a que la declinación y la
madurez del campo se manifiesten.

**Reservas que conviene declarar:**

1. La columna de campos de más de 50 000 bpd contiene **solo dos campos**. Que
   allí gane el Naive no soporta ninguna conclusión firme.
2. **La operadora es la variable más importante** (15.3 % de la ganancia). Es un
   resultado plausible —las prácticas operativas dejan huella en la trayectoria—
   pero en operadoras con un único campo la variable funciona en parte como
   identificador del campo, no como información transferible.
3. Los hiperparámetros **no se optimizaron**. Se fijaron valores razonables y se
   dejó que la parada temprana determinara el número de árboles. Una búsqueda
   sistemática podría mejorar el resultado, y debe hacerse sobre la partición de
   validación temporal, nunca sobre los cortes de evaluación.

---

## 7. Modelos híbridos (Fase 4)

### 7.1 Planteamiento

Las fases 1 a 3 dejaron establecido que ningún modelo domina en todos los
regímenes: la persistencia gana a un mes, el modelo global a partir del tercero,
y Arps se defiende en los campos grandes. Eso plantea una pregunta que no es de
implementación sino de diseño: **¿cómo se combinan?**

Se contrastan dos respuestas deliberadamente opuestas, porque la comparación
entre ellas es en sí misma un resultado.

### 7.2 Vía A — Arps como variable (híbrido implícito)

Se ajusta Arps en cada par (campo, origen) con ventana de 24 meses, se proyecta
al horizonte correspondiente, y el pronóstico se entrega al modelo global como
una variable más, en la misma escala del objetivo: `log(q_arps / ancla)`. Se
acompaña de `Di` y `b`, que describen el régimen de declinación estimado.

El modelo decide por sí solo cuánto peso dar a la física. Es la opción flexible.

**Eficiencia y corrección.** Los 38 257 ajustes de Arps se calculan **una sola
vez** y se cachean: el ajuste en el origen `t` depende únicamente de datos
anteriores o iguales a `t`, así que el mismo resultado sirve para los tres cortes
sin riesgo de fuga. El eje temporal del ajuste es de meses de calendario, no de
posiciones: en un campo con huecos, doce reportes atrás no son doce meses atrás y
la declinación saldría sesgada.

`arps_multiple` es una reescritura vectorizada de la curva, necesaria porque la
versión original recibe `b` como escalar —lo que exige `curve_fit`— y su rama
`if b < 1e-6` no admite vectores. Una prueba comprueba que ambas coinciden en
todo el rango de `b`, incluidos los extremos.

### 7.3 Vía B — Combinación convexa por régimen (híbrido explícito)

Se estiman pesos no negativos que suman uno sobre los pronósticos de Naive,
Arps-24m y el modelo global, por separado para cada **horizonte** y **clase de
tamaño de campo**.

**Estimación de los pesos.** Búsqueda exhaustiva sobre una malla del símplex en
pasos de 0.1: con tres modelos son 66 combinaciones, barato y suficiente. Se
prefiere a una optimización continua porque esta última daría pesos con una
precisión que los datos no respaldan. El criterio es el error absoluto medio, el
mismo que se usa para evaluar.

**Dónde se estiman.** En un origen de validación **doce meses anterior** al corte
de evaluación. En ese origen los valores reales ya se conocen en la fecha de
corte, así que no hay fuga; estimar los pesos sobre el propio corte y luego medir
sobre él sería circular. Esto obliga a entrenar el modelo global dos veces por
corte, una para la validación y otra para la evaluación.

**Repliegue.** Un bucket con menos de 60 observaciones no da para estimar tres
pesos, así que se usan los del horizonte completo, y si tampoco alcanzan, los
globales. La clase de tamaño se calcula con los doce meses **previos** al origen:
usar la media de toda la serie metería información posterior en una variable de
estratificación.

### 7.4 Resultados

| Modelo | MAE | sMAPE | MASE | Sesgo |
|---|---|---|---|---|
| Híbrido-régimen | 260.7 | 20.1 % | **1.551** | −13.2 |
| Híbrido-Arps-ML | 265.8 | 20.4 % | 1.591 | −49.9 |
| ML-global | 262.9 | 20.5 % | 1.595 | −45.6 |
| Naive | 268.8 | 21.8 % | 1.660 | +30.5 |
| Arps-24m | 328.4 | 24.6 % | 1.939 | −147.7 |

**La vía B gana; la vía A apenas aporta.** El híbrido explícito mejora un 6.6 %
sobre el pronóstico ingenuo y un 2.8 % sobre el modelo global. El implícito mejora
un 0.25 % sobre el modelo global, es decir, prácticamente nada.

### 7.5 Interpretación del resultado contraintuitivo

Cabía esperar lo opuesto: la vía A es estrictamente más flexible —puede
representar cualquier combinación que la vía B exprese, y muchas más— y dispone
de la misma información. Que pierda tiene dos explicaciones que conviene separar:

1. **Arps no aporta información nueva.** Los rezagos, las pendientes y la
   volatilidad ya describen la trayectoria reciente del campo, que es exactamente
   lo que Arps resume en tres parámetros. El modelo global no gana nada al
   recibirlo y lo ignora casi por completo.

2. **La vía B no explota información, sino diversidad de errores.** Sus
   componentes se equivocan en direcciones opuestas —Naive sobreestima en +30 bpd,
   Arps subestima en −148— y promediarlos cancela el sesgo: el del híbrido cae a
   −13 bpd. Ese mecanismo es inaccesible para un modelo único por flexible que
   sea, porque no consiste en aprender mejor sino en **combinar estimadores
   sesgados de forma complementaria**.

La conclusión metodológica es que, en este problema, la ganancia no está en darle
más información al modelo sino en **cómo se agregan modelos que ya existen**.

### 7.6 Los pesos como entregable

Promedio de los tres cortes:

| Horizonte | Naive | Arps-24m | ML-global |
|---|---|---|---|
| 1 mes | 0.72 | 0.03 | 0.25 |
| 3 meses | 0.53 | 0.15 | 0.32 |
| 6 meses | 0.33 | 0.20 | 0.47 |
| 12 meses | 0.28 | 0.22 | 0.50 |

La rotación es continua y **no se le impuso al procedimiento**: emerge de
minimizar el error en el periodo de validación. Que reproduzca lo que las fases 1
a 3 habían encontrado por separado —persistencia a corto plazo, aprendizaje y
física a largo— es una validación cruzada de todo el trabajo anterior.

Para una operadora la tabla es directamente accionable: indica qué método usar
según el plazo de planeación. Esa legibilidad es la ventaja práctica del híbrido
explícito sobre el implícito, y pesa tanto como los 0.04 puntos de MASE.

### 7.7 Reservas

1. **La mejora es modesta en términos absolutos.** Un MASE de 1.551 sigue estando
   por encima de 1: el pronóstico ingenuo de un paso continúa siendo un rival
   duro, y una parte grande del error es irreducible (Fase 2, §5).
2. **Los pesos se estiman sobre un solo origen de validación** por corte. Usar
   varios daría estimaciones más estables, a costa de más entrenamientos.
3. **La malla de pesos es gruesa** (pasos de 0.1). Es una decisión consciente
   frente al sobreajuste, pero impide distinguir diferencias finas.
4. **La combinación es lineal y convexa.** No puede representar reglas del tipo
   «usar Arps solo si el ajuste fue bueno»; una combinación con pesos dependientes
   de la calidad del ajuste queda como extensión natural.

---

## 8. Limitaciones declaradas

1. **Granularidad de campo.** Las curvas agregadas mezclan pozos en distintas
   etapas de su vida. Los resultados no son extrapolables a pozo individual sin
   validación adicional.
2. **Sin variables de intervención.** No se dispone de fechas de workover,
   perforación de infill ni cambios de sistema de levantamiento. Parte del error
   irreducible corresponde a estos eventos no observados.
3. **Sin variables de precio ni de decisión comercial.** Los recortes de
   producción de 2020 responden a la caída del precio del crudo y a decisiones
   de la OPEP+, no a la geología. Ningún modelo del benchmark puede anticiparlos.
4. **Tres orígenes por campo.** Un número mayor daría estimaciones de error más
   estables, a costa de tiempo de cómputo. Se fijó en 3 para la Fase 1 y se
   revisará al introducir modelos de mayor costo.
5. **Los campos con menos de 36 meses quedan fuera** (183 de 608). Son campos
   nuevos o de vida corta, y su exclusión sesga la muestra evaluada hacia campos
   consolidados.
6. **La segmentación tiene silueta baja** (0.26). Los grupos son interpretables
   pero no netamente separados; un campo cerca de una frontera podría cambiar de
   segmento ante pequeñas variaciones de sus descriptores.
7. **El horizonte del backtesting de la Fase 1 se cuenta en observaciones, no
   en meses de calendario.** En un campo con huecos de reporte, doce pasos
   abarcan más de doce meses. El protocolo de la Fase 3 corrige esto usando
   meses de calendario, pero por eso mismo sus cifras no son comparables con las
   de la Fase 1.
8. **Los hiperparámetros del modelo global no se optimizaron.** Se fijaron
   valores razonables y la parada temprana determinó el número de árboles.
9. **Solo tres cortes de calendario** en las fases 3 y 4. Con más cortes las
   estimaciones de error serían más estables, a costa de tiempo de cómputo.
10. **Los pesos del híbrido se estiman sobre un único origen de validación** por
    corte, y sobre una malla gruesa de pasos de 0.1.
11. **La combinación híbrida es lineal y convexa.** No puede expresar reglas
    condicionadas a la calidad del ajuste de Arps en cada campo.

---

## 9. Reproducibilidad

```bash
python -m pip install -e ".[dev]"
oilai all --force      # reconstruye todo desde el snapshot versionado
python -m pytest       # 124 pruebas
```

Las semillas aleatorias están fijadas en las pruebas que las requieren. El
pipeline es determinista: partiendo del mismo snapshot en `data/raw/`, produce
exactamente las mismas métricas.
