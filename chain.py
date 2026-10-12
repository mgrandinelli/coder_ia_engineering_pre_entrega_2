"""Pipeline de extracción de entidades técnicas: prompt | modelo con salida estructurada.

La cadena reintenta sola cuando el modelo devuelve un objeto mal formado o incompleto.
"""

import logging
import os

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from schemas import EntidadesTecnicas

load_dotenv()

log = logging.getLogger("pipeline")

TEMPERATURE = 0
MAX_TOKENS = 1024
MAX_INTENTOS = 3
API_KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
SIN_CLAVE = "sin-configurar"  # permite construir el modelo e importar el módulo sin credenciales

TEXTO_EJEMPLO = (
    "El servicio de pagos, una API en FastAPI con caché en Redis y persistencia en PostgreSQL, "
    "empezó a devolver errores 503 en producción: el pool de conexiones a la base se agota "
    "cuando hay más de 200 requests concurrentes."
)

INSTRUCCIONES_FORMATO = (
    "- tecnologias: nombres propios de las tecnologías mencionadas, sin repetir. "
    "Nunca una lista vacía.\n"
    "- nivel_de_criticidad: 'alta' si hay caída o pérdida de datos en producción, "
    "'media' si hay degradación o riesgo, 'baja' si es informativo.\n"
    "- resumen_tecnico: una o dos oraciones en español, sin opiniones."
)


class SalidaInvalidaError(Exception):
    """El modelo respondió, pero el objeto no cumple el esquema. Es el error que se reintenta."""


class RespuestaIncompletaError(SalidaInvalidaError):
    """La respuesta se cortó por falta de tokens (finish_reason 'length' / 'max_tokens')."""


# --- Modelo: misma selección por variable de entorno que en el módulo 1 ---
def build_model(provider: str | None = None) -> BaseChatModel:
    provider = provider or os.getenv("LLM_PROVIDER") or "openai"
    # Cada integración se importa solo si se usa: alcanza con tener instalada una de las dos.
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL") or "claude-haiku-4-5",
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            api_key=os.getenv("ANTHROPIC_API_KEY") or SIN_CLAVE,
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL") or "gpt-4o-mini",
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        api_key=os.getenv("OPENAI_API_KEY") or SIN_CLAVE,
    )


# --- Prompt modular: el texto y las instrucciones de formato son variables ---
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Sos un ingeniero de software que extrae entidades técnicas de textos sin procesar "
            "(descripciones de arquitectura o logs de error).\n\n"
            "Instrucciones de formato:\n{instrucciones_formato}",
        ),
        ("human", "Texto a analizar:\n\n{texto}"),
    ]
).partial(instrucciones_formato=INSTRUCCIONES_FORMATO)


def _verificar(salida: dict) -> EntidadesTecnicas:
    """Revisa la respuesta cruda antes de entregar el objeto: primero si se cortó, después si valida."""
    metadatos = salida["raw"].response_metadata
    motivo = metadatos.get("finish_reason") or metadatos.get("stop_reason")
    if motivo in ("length", "max_tokens"):
        log.warning("Respuesta incompleta (finish_reason=%s); se reintenta", motivo)
        raise RespuestaIncompletaError(f"respuesta cortada por límite de tokens ({motivo})")

    error = salida["parsing_error"]
    if error is not None or salida["parsed"] is None:
        detalle = str(error).splitlines()[0] if error else "el modelo no devolvió el objeto"
        log.warning("Salida rechazada por el validador (%s); se reintenta", detalle)
        raise SalidaInvalidaError(detalle) from error

    log.info("Salida validada contra %s", EntidadesTecnicas.__name__)
    return salida["parsed"]


# --- Cadena LCEL con salida estructurada y reintentos ---
def build_chain(model: BaseChatModel, espera: bool = True) -> Runnable[dict, EntidadesTecnicas]:
    chain = prompt | model.with_structured_output(EntidadesTecnicas, include_raw=True) | _verificar
    # Solo se reintentan los errores de formato. Los transitorios (red, 429) ya los reintenta
    # el SDK del proveedor, y reintentar los permanentes (clave inválida) sería gastar llamadas.
    return chain.with_retry(
        retry_if_exception_type=(SalidaInvalidaError,),
        wait_exponential_jitter=espera,  # backoff exponencial entre intentos
        stop_after_attempt=MAX_INTENTOS,
    )


model = build_model()
chain = build_chain(model)


def _error_de_configuracion() -> str | None:
    provider = os.getenv("LLM_PROVIDER") or "openai"
    variable = API_KEY_ENV.get(provider)
    if variable is None:
        return f"LLM_PROVIDER debe ser 'openai' o 'anthropic', no '{provider}'"
    if not os.getenv(variable):
        return f"Falta la variable de entorno {variable}"
    return None


def _informar_fallo(exc: BaseException) -> None:
    if isinstance(exc, SalidaInvalidaError):
        log.error("Sin salida válida tras %d intentos: %s", MAX_INTENTOS, exc)
    else:
        # Clave inválida, límite de tasa o red, ya agotados los reintentos del SDK.
        log.error("Falló la llamada al modelo (%s): %s", type(exc).__name__, exc)


async def process_text(text: str = TEXTO_EJEMPLO) -> EntidadesTecnicas | None:
    """Ejecuta la cadena sobre un texto. Devuelve None, con el motivo en el log, si no hay salida válida."""
    if not text or not text.strip():
        log.error("El texto de entrada está vacío")
        return None
    if error := _error_de_configuracion():
        log.error(error)
        return None

    log.info("Procesando texto de %d caracteres", len(text))
    try:
        resultado = await chain.ainvoke({"texto": text})
    except Exception as exc:
        _informar_fallo(exc)
        return None

    log.info(
        "Extracción lista: %d tecnologías, criticidad %s",
        len(resultado.tecnologias),
        resultado.nivel_de_criticidad.value,
    )
    return resultado


async def process_batch(texts: list[str] | None = None) -> list[EntidadesTecnicas | None]:
    """Procesa varios textos a la vez con .abatch(). Un texto que falla no afecta a los demás."""
    texts = [TEXTO_EJEMPLO] if texts is None else texts
    if error := _error_de_configuracion():
        log.error(error)
        return [None] * len(texts)

    log.info("Procesando lote de %d textos", len(texts))
    salidas = await chain.abatch([{"texto": t} for t in texts], return_exceptions=True)
    resultados: list[EntidadesTecnicas | None] = []
    for salida in salidas:
        if isinstance(salida, BaseException):
            _informar_fallo(salida)
            resultados.append(None)
        else:
            resultados.append(salida)
    log.info("Lote listo: %d de %d con salida válida", sum(r is not None for r in resultados), len(texts))
    return resultados
