"""Tests de la pre-entrega 2. No usan la red ni claves: el modelo se reemplaza por un doble.

Ejecutar con: pytest -v
"""

import logging
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables.retry import RunnableRetry
from pydantic import ValidationError

import chain as pipeline
import main
from schemas import EntidadesTecnicas, NivelCriticidad

pytestmark = pytest.mark.asyncio

VALIDA = {
    "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
    "nivel_de_criticidad": "alta",
    "resumen_tecnico": "API con caché en Redis y persistencia en PostgreSQL.",
}
SIN_TECNOLOGIAS = {**VALIDA, "tecnologias": []}


def respuesta(args: dict, finish_reason: str = "tool_calls") -> AIMessage:
    """Mensaje del modelo con la llamada a la herramienta que genera with_structured_output."""
    return AIMessage(
        content="",
        tool_calls=[{"name": "EntidadesTecnicas", "args": args, "id": "1"}],
        response_metadata={"finish_reason": finish_reason},
    )


class ModeloFalso(BaseChatModel):
    """Devuelve las respuestas preparadas, en orden, y cuenta las llamadas."""

    respuestas: list[Any]
    llamadas: int = 0

    @property
    def _llm_type(self) -> str:
        return "falso"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.llamadas += 1
        siguiente = self.respuestas[min(self.llamadas, len(self.respuestas)) - 1]
        if isinstance(siguiente, Exception):
            raise siguiente
        return ChatResult(generations=[ChatGeneration(message=siguiente)])


def cadena_con(*respuestas):
    modelo = ModeloFalso(respuestas=list(respuestas))
    return modelo, pipeline.build_chain(modelo, espera=False)


@pytest.fixture
def con_clave(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "clave-de-prueba")


# --- Esquema y salida estructurada ---


async def test_el_esquema_acepta_el_ejemplo_de_la_consigna():
    entidades = EntidadesTecnicas(**VALIDA)

    assert entidades.tecnologias == ["FastAPI", "Redis", "PostgreSQL"]
    assert entidades.nivel_de_criticidad is NivelCriticidad.ALTA


@pytest.mark.parametrize(
    "campos",
    [
        {"tecnologias": []},
        {"tecnologias": ["  ", ""]},
        {"nivel_de_criticidad": "urgente"},
        {"resumen_tecnico": "corto"},
    ],
)
async def test_el_esquema_rechaza_datos_invalidos(campos):
    with pytest.raises(ValidationError):
        EntidadesTecnicas(**{**VALIDA, **campos})


async def test_el_esquema_limpia_y_deduplica_las_tecnologias():
    entidades = EntidadesTecnicas(**{**VALIDA, "tecnologias": [" Redis ", "Redis", "Kafka"]})

    assert entidades.tecnologias == ["Redis", "Kafka"]


async def test_la_cadena_devuelve_un_objeto_validado():
    _, cadena = cadena_con(respuesta(VALIDA))

    resultado = await cadena.ainvoke({"texto": "API en FastAPI con Redis y PostgreSQL"})

    assert isinstance(resultado, EntidadesTecnicas)
    assert resultado.model_dump(mode="json") == VALIDA


# --- Cadena LCEL y prompt ---


async def test_la_cadena_del_modulo_es_prompt_mas_modelo_estructurado_con_reintentos():
    assert pipeline.chain.first is pipeline.prompt
    assert isinstance(pipeline.chain.last, RunnableRetry)
    assert pipeline.chain.last.max_attempt_number == pipeline.MAX_INTENTOS


async def test_el_prompt_recibe_el_texto_y_las_instrucciones_de_formato():
    mensajes = pipeline.prompt.format_messages(texto="log de prueba")

    assert pipeline.prompt.input_variables == ["texto"]
    assert "instrucciones_formato" in pipeline.prompt.partial_variables
    assert isinstance(mensajes[0], SystemMessage) and isinstance(mensajes[1], HumanMessage)
    assert pipeline.INSTRUCCIONES_FORMATO in mensajes[0].content
    assert "log de prueba" in mensajes[1].content


# --- Resiliencia ---


async def test_reintenta_cuando_la_salida_no_cumple_el_esquema(caplog):
    modelo, cadena = cadena_con(respuesta(SIN_TECNOLOGIAS), respuesta(VALIDA))

    with caplog.at_level(logging.WARNING, logger="pipeline"):
        resultado = await cadena.ainvoke({"texto": "texto"})

    assert modelo.llamadas == 2
    assert resultado.tecnologias == VALIDA["tecnologias"]
    assert "Salida rechazada por el validador" in caplog.text


async def test_reintenta_cuando_la_respuesta_se_corta_por_tokens(caplog):
    modelo, cadena = cadena_con(respuesta(VALIDA, finish_reason="length"), respuesta(VALIDA))

    with caplog.at_level(logging.WARNING, logger="pipeline"):
        resultado = await cadena.ainvoke({"texto": "texto"})

    assert modelo.llamadas == 2
    assert isinstance(resultado, EntidadesTecnicas)
    assert "Respuesta incompleta (finish_reason=length)" in caplog.text


async def test_se_rinde_despues_del_maximo_de_intentos():
    modelo, cadena = cadena_con(respuesta(SIN_TECNOLOGIAS))

    with pytest.raises(pipeline.SalidaInvalidaError):
        await cadena.ainvoke({"texto": "texto"})

    assert modelo.llamadas == pipeline.MAX_INTENTOS


async def test_no_reintenta_errores_que_no_son_de_formato():
    modelo, cadena = cadena_con(PermissionError("clave inválida"))

    with pytest.raises(PermissionError):
        await cadena.ainvoke({"texto": "texto"})

    assert modelo.llamadas == 1


# --- process_text y process_batch ---


async def test_process_text_devuelve_el_objeto_y_registra_la_validacion(monkeypatch, con_clave, caplog):
    monkeypatch.setattr(pipeline, "chain", cadena_con(respuesta(VALIDA))[1])

    with caplog.at_level(logging.INFO, logger="pipeline"):
        resultado = await pipeline.process_text("API en FastAPI con Redis y PostgreSQL")

    assert resultado.nivel_de_criticidad is NivelCriticidad.ALTA
    assert "Salida validada contra EntidadesTecnicas" in caplog.text
    assert "Extracción lista: 3 tecnologías, criticidad alta" in caplog.text


async def test_process_text_devuelve_none_si_el_modelo_nunca_cumple(monkeypatch, con_clave, caplog):
    monkeypatch.setattr(pipeline, "chain", cadena_con(respuesta(SIN_TECNOLOGIAS))[1])

    with caplog.at_level(logging.ERROR, logger="pipeline"):
        resultado = await pipeline.process_text("texto ambiguo")

    assert resultado is None
    assert f"Sin salida válida tras {pipeline.MAX_INTENTOS} intentos" in caplog.text


async def test_process_text_devuelve_none_si_falla_la_api(monkeypatch, con_clave):
    monkeypatch.setattr(pipeline, "chain", cadena_con(ConnectionError("sin red"))[1])

    assert await pipeline.process_text("texto") is None


@pytest.mark.parametrize("texto", ["", "   "])
async def test_process_text_rechaza_un_texto_vacio(con_clave, texto):
    assert await pipeline.process_text(texto) is None


async def test_process_text_sin_argumentos_ni_clave_no_se_rompe(monkeypatch, caplog):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with caplog.at_level(logging.ERROR, logger="pipeline"):
        resultado = await pipeline.process_text()

    assert resultado is None
    assert "Falta la variable de entorno OPENAI_API_KEY" in caplog.text


async def test_process_batch_procesa_varios_textos_y_aisla_los_fallos(monkeypatch, con_clave):
    # Con concurrencia 1 el orden de las respuestas es determinista: válida, luego siempre inválida.
    _, cadena = cadena_con(respuesta(VALIDA), respuesta(SIN_TECNOLOGIAS))
    monkeypatch.setattr(pipeline, "chain", cadena.with_config(max_concurrency=1))

    resultados = await pipeline.process_batch(["texto uno", "texto dos"])

    assert isinstance(resultados[0], EntidadesTecnicas)
    assert resultados[1] is None


async def test_main_corre_sin_clave_y_sin_romperse(monkeypatch, capsys):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    await main.main()

    assert "Sin resultado: el motivo está en el log." in capsys.readouterr().out
