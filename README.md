# Pre-entrega 2: Pipeline de procesamiento validado

Curso AI Engineering (Coderhouse), módulo 2.

Un pipeline de extracción de entidades técnicas. Recibe un párrafo de texto sin procesar (una descripción de arquitectura o un log de error) y devuelve un objeto validado con Pydantic: las tecnologías mencionadas, el nivel de criticidad y un resumen técnico.

```python
chain = prompt | model.with_structured_output(EntidadesTecnicas)   # más verificación y .with_retry()
resultado = await process_text("API en FastAPI con caché en Redis...")
```

## Cómo ejecutarlo

Requiere Python 3.12 o superior.

```bash
git clone https://github.com/mgrandinelli/coder_ia_engineering_pre_entrega_2.git
cd coder_ia_engineering_pre_entrega_2

python3 -m venv .venv
source .venv/bin/activate
pip install langchain langchain-openai langchain-anthropic pydantic python-dotenv

python main.py
```

Las claves se leen de variables de entorno o de un archivo `.env` en la misma carpeta, por ejemplo:

```
LLM_PROVIDER=openai
OPENAI_API_KEY=...
```

## Variables de entorno

| Variable | Obligatoria | Qué es |
|---|---|---|
| `LLM_PROVIDER` | No (por defecto `openai`) | Proveedor a usar: `openai` o `anthropic` |
| `OPENAI_API_KEY` | Con OpenAI | Clave de la API de OpenAI |
| `ANTHROPIC_API_KEY` | Con Anthropic | Clave de la API de Anthropic |
| `OPENAI_MODEL` | No (por defecto `gpt-4o-mini`) | Modelo de OpenAI |
| `ANTHROPIC_MODEL` | No (por defecto `claude-haiku-4-5`) | Modelo de Anthropic |

Solo hace falta la clave del proveedor elegido, y solo la integración de ese proveedor (`langchain-openai` o `langchain-anthropic`).

## Archivos

| Archivo | Contenido |
|---|---|
| `schemas.py` | Contrato de datos: `EntidadesTecnicas` y el enum `NivelCriticidad` |
| `chain.py` | Modelo, `ChatPromptTemplate`, cadena LCEL con salida estructurada y reintentos, `process_text()` y `process_batch()` |
| `main.py` | Mini-script de prueba asíncrono con tres textos |
| `test_main.py` | 21 tests, sin red ni claves |

## Ejemplo de salida

Para este texto de entrada:

> El servicio de pagos, una API en FastAPI con caché en Redis y persistencia en PostgreSQL, empezó a devolver errores 503 en producción: el pool de conexiones a la base se agota cuando hay más de 200 requests concurrentes.

el pipeline devuelve un objeto `EntidadesTecnicas` como este (el texto del resumen varía en cada ejecución):

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "API con caché en Redis y persistencia en PostgreSQL; el pool de conexiones se agota con más de 200 requests concurrentes."
}
```

Sin clave configurada el script no se interrumpe: el log informa `Falta la variable de entorno OPENAI_API_KEY` y cada caso imprime `Sin resultado: el motivo está en el log.`

## Arquitectura y decisiones de diseño

### Esquema Pydantic (`schemas.py`)

`EntidadesTecnicas` tiene los tres campos de la consigna y sus restricciones:

| Campo | Tipo | Restricción |
|---|---|---|
| `tecnologias` | `list[str]` | No puede quedar vacía; se quitan espacios y duplicados |
| `nivel_de_criticidad` | `NivelCriticidad` | Enum: `baja`, `media` o `alta` |
| `resumen_tecnico` | `str` | Al menos 10 caracteres |

Las restricciones están en validadores de Pydantic y no en el esquema JSON que recibe el proveedor. Así se aplican siempre del lado del cliente, y una respuesta que las incumple produce un error de validación que la cadena puede reintentar.

### Prompt (`chain.py`)

`ChatPromptTemplate.from_messages` con un mensaje `system` y uno `human`. Tiene dos variables: `{texto}`, que llega en cada llamada, e `{instrucciones_formato}`, que se fija con `.partial()`. Las instrucciones de formato viven en una constante aparte y se pueden cambiar sin tocar el template. No hay f-strings dentro de la cadena.

### Cadena LCEL

```python
extraccion = model.with_structured_output(EntidadesTecnicas, include_raw=True)
con_reintentos = (extraccion | RunnableLambda(_verificar)).with_retry(
    retry_if_exception_type=(SalidaInvalidaError,),
    wait_exponential_jitter=True,
    stop_after_attempt=3,
)
chain = prompt | con_reintentos
```

El recorrido de los datos es:

1. `prompt` arma los mensajes con el texto y las instrucciones de formato.
2. `model.with_structured_output(EntidadesTecnicas, include_raw=True)` devuelve el mensaje crudo del modelo, el objeto ya validado y, si lo hubo, el error de validación.
3. `_verificar` revisa primero el `finish_reason`: si la respuesta se cortó por falta de tokens (`length` en OpenAI, `max_tokens` en Anthropic), lanza `RespuestaIncompletaError` antes de usar el objeto. Después revisa el error de validación y, si existe, lanza `SalidaInvalidaError`.
4. `.with_retry()` repite los pasos 2 y 3 hasta 3 veces, con espera exponencial entre intentos.

### Qué se reintenta y qué no

| Tipo de error | Ejemplo | Quién lo maneja |
|---|---|---|
| De formato | JSON mal formado, lista de tecnologías vacía, respuesta cortada | `.with_retry()` de la cadena, hasta 3 intentos |
| Transitorio | Red, límite de tasa (429), error 5xx | Reintentos propios del SDK del proveedor |
| Permanente | Clave inválida, modelo inexistente | No se reintenta: se informa en el log |

Reintentar un error permanente solo gasta llamadas, por eso `.with_retry()` se limita a `SalidaInvalidaError`.

### `process_text()` y `process_batch()`

- `process_text(text)` ejecuta la cadena con `await chain.ainvoke({"texto": text})`. Devuelve un `EntidadesTecnicas`, o `None` si no hubo salida válida; el motivo queda en el log.
- `process_batch(texts)` procesa varios textos a la vez con `await chain.abatch(...)`. Un texto que falla devuelve `None` en su posición y no afecta a los demás.

### Logs

El logger `pipeline` registra cada etapa. Este es el log de una ejecución en la que el primer intento devolvió una lista de tecnologías vacía y el segundo fue válido (generado con el modelo falso de los tests):

```
INFO    Procesando texto de 220 caracteres
WARNING Salida rechazada por el validador (1 validation error for EntidadesTecnicas); se reintenta
INFO    Salida validada contra EntidadesTecnicas
INFO    Extracción lista: 3 tecnologías, criticidad alta
```

Cuando la respuesta se corta por tokens el aviso es `Respuesta incompleta (finish_reason=length); se reintenta`, y si se agotan los intentos, `Sin salida válida tras 3 intentos`.

### Prueba de estrés

`main.py` incluye un texto ambiguo que no menciona ninguna tecnología ("Ayer anduvo todo bastante lento..."). El esquema exige al menos una, así que hay dos resultados posibles: el modelo se recupera en un reintento, o los tres intentos fallan y `process_text` devuelve `None` con el aviso en el log. En ningún caso el programa se interrumpe.

### Modelo y `temperature`

`ChatOpenAI` y `ChatAnthropic` se crean con `temperature=0`, para que la extracción sea lo más estable posible. El modelo por defecto de Anthropic es `claude-haiku-4-5` porque acepta ese parámetro; los modelos Claude más recientes (por ejemplo `claude-opus-5-5`) ya no lo usan y lo rechazan con un error 400.

Si falta la variable de entorno de la clave, el modelo se construye con un valor de relleno, así los módulos se pueden importar sin credenciales; `process_text()` verifica la variable antes de llamar a la API.

## Tests

`test_main.py` tiene 21 tests. No usan la red ni claves: el modelo se reemplaza por un doble que devuelve respuestas preparadas y cuenta las llamadas.

```bash
pip install pytest pytest-asyncio
pytest -v
```

Cubren los cuatro criterios de la rúbrica:

- **Esquema y salida estructurada:** el ejemplo de la consigna valida; lista vacía, criticidad desconocida y resumen corto se rechazan; la cadena devuelve un `EntidadesTecnicas`.
- **Cadena LCEL y prompt:** `chain` es `prompt` seguido del modelo estructurado con reintentos; el prompt tiene roles `system` y `human` y recibe el texto y las instrucciones de formato.
- **Asincronía y resiliencia:** se reintenta ante una salida inválida y ante una respuesta cortada; se abandona al tercer intento; un error que no es de formato no se reintenta; los logs registran validación y reintentos.
- **Script de prueba:** `process_text()`, `process_batch()` y `main()` funcionan sin argumentos y sin clave, sin interrumpirse.

## Reporte de tests

Salida de `pytest -v`:

```
============================= test session starts ==============================
collecting ... collected 21 items

test_main.py::test_el_esquema_acepta_el_ejemplo_de_la_consigna PASSED    [  4%]
test_main.py::test_el_esquema_rechaza_datos_invalidos[campos0] PASSED    [  9%]
test_main.py::test_el_esquema_rechaza_datos_invalidos[campos1] PASSED    [ 14%]
test_main.py::test_el_esquema_rechaza_datos_invalidos[campos2] PASSED    [ 19%]
test_main.py::test_el_esquema_rechaza_datos_invalidos[campos3] PASSED    [ 23%]
test_main.py::test_el_esquema_limpia_y_deduplica_las_tecnologias PASSED  [ 28%]
test_main.py::test_la_cadena_devuelve_un_objeto_validado PASSED          [ 33%]
test_main.py::test_la_cadena_del_modulo_es_prompt_mas_modelo_estructurado_con_reintentos PASSED [ 38%]
test_main.py::test_el_prompt_recibe_el_texto_y_las_instrucciones_de_formato PASSED [ 42%]
test_main.py::test_reintenta_cuando_la_salida_no_cumple_el_esquema PASSED [ 47%]
test_main.py::test_reintenta_cuando_la_respuesta_se_corta_por_tokens PASSED [ 52%]
test_main.py::test_se_rinde_despues_del_maximo_de_intentos PASSED        [ 57%]
test_main.py::test_no_reintenta_errores_que_no_son_de_formato PASSED     [ 61%]
test_main.py::test_process_text_devuelve_el_objeto_y_registra_la_validacion PASSED [ 66%]
test_main.py::test_process_text_devuelve_none_si_el_modelo_nunca_cumple PASSED [ 71%]
test_main.py::test_process_text_devuelve_none_si_falla_la_api PASSED     [ 76%]
test_main.py::test_process_text_rechaza_un_texto_vacio[] PASSED          [ 80%]
test_main.py::test_process_text_rechaza_un_texto_vacio[   ] PASSED       [ 85%]
test_main.py::test_process_text_sin_argumentos_ni_clave_no_se_rompe PASSED [ 90%]
test_main.py::test_process_batch_procesa_varios_textos_y_aisla_los_fallos PASSED [ 95%]
test_main.py::test_main_corre_sin_clave_y_sin_romperse PASSED            [100%]

============================== 21 passed in 0.89s ==============================
```
