from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Complaint:
    """Modelo de domínio para uma reclamação"""
    nu_registro: str
    ds_assunto: str
    cd_usuario: str
    sub_assunto: str
    ds_observacao: str
    dt_registro_atendimento: datetime
    ds_filial: str

    def __post_init__(self):
        """Validações básicas"""
        if not self.nu_registro:
            raise ValueError("NU_REGISTRO não pode ser vazio")
        if not isinstance(self.dt_registro_atendimento, datetime):
            raise ValueError("DT_REGISTRO_ATENDIMENTO deve ser datetime")
