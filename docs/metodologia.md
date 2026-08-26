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

## 6. Limitaciones declaradas

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
7. **El horizonte del backtesting se cuenta en observaciones, no en meses de
   calendario.** En un campo con huecos de reporte, doce pasos abarcan más de
   doce meses. Los instantes `t` que reciben los modelos sí son correctos, de
   modo que las predicciones se emiten en el momento debido, pero la etiqueta
   «h = 12» no equivale a un año exacto en esos campos.

---

## 7. Reproducibilidad

```bash
python -m pip install -e ".[dev]"
oilai all --force      # reconstruye todo desde el snapshot versionado
python -m pytest       # 34 pruebas
```

Las semillas aleatorias están fijadas en las pruebas que las requieren. El
pipeline es determinista: partiendo del mismo snapshot en `data/raw/`, produce
exactamente las mismas métricas.
