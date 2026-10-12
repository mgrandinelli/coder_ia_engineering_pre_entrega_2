"""Contrato de datos del pipeline: la estructura que el modelo está obligado a devolver."""

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class NivelCriticidad(str, Enum):
    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class EntidadesTecnicas(BaseModel):
    """Entidades técnicas extraídas de un texto (descripción de arquitectura o log de error)."""

    tecnologias: list[str] = Field(
        description="Tecnologías, frameworks, lenguajes o servicios mencionados en el texto."
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        description="Gravedad de la situación descripta: baja, media o alta."
    )
    resumen_tecnico: str = Field(
        description="Resumen técnico del texto en una o dos oraciones."
    )

    # Las restricciones van en validadores y no en Field(min_length=...): así se aplican
    # siempre del lado del cliente, sin depender de qué acepte el esquema JSON del proveedor.
    @field_validator("tecnologias")
    @classmethod
    def _tecnologias_no_vacias(cls, valor: list[str]) -> list[str]:
        limpias = list(dict.fromkeys(t.strip() for t in valor if t.strip()))
        if not limpias:
            raise ValueError("la lista de tecnologías no puede estar vacía")
        return limpias

    @field_validator("resumen_tecnico")
    @classmethod
    def _resumen_con_contenido(cls, valor: str) -> str:
        valor = valor.strip()
        if len(valor) < 10:
            raise ValueError("el resumen técnico debe tener al menos 10 caracteres")
        return valor
