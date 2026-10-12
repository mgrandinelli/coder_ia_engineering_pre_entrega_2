"""Pre-entrega 2 - Script de prueba del pipeline de extracción de entidades técnicas."""

import asyncio
import logging

from chain import TEXTO_EJEMPLO, chain, process_batch, process_text
from schemas import EntidadesTecnicas, NivelCriticidad

__all__ = ["EntidadesTecnicas", "NivelCriticidad", "chain", "main", "process_batch", "process_text"]

LOG_DE_ERROR = (
    "2026-03-02 14:07:55 ERROR [worker-3] celery.app.trace: Task reportes.generar_pdf raised "
    "MemoryError. Kubernetes reinició el pod (OOMKilled, límite 512Mi). 14 tareas quedaron "
    "sin procesar en la cola de RabbitMQ."
)
# Prueba de estrés: no menciona ninguna tecnología, así que el esquema no se puede cumplir bien.
TEXTO_AMBIGUO = "Ayer anduvo todo bastante lento, pero hoy parece que está mejor. No sé qué habrá sido."



def mostrar(resultado: EntidadesTecnicas | None) -> None:
    if resultado is None:
        print("Sin resultado: el motivo está en el log.")
    else:
        print(resultado.model_dump_json(indent=2))


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    print(f"=== Descripción de arquitectura ===\n{TEXTO_EJEMPLO}\n")
    mostrar(await process_text(TEXTO_EJEMPLO))

    print("\n=== Lote: log de error y texto ambiguo (prueba de estrés) ===")
    lote = [LOG_DE_ERROR, TEXTO_AMBIGUO]
    for texto, resultado in zip(lote, await process_batch(lote)):
        print(f"\n{texto}\n")
        mostrar(resultado)


if __name__ == "__main__":
    asyncio.run(main())
