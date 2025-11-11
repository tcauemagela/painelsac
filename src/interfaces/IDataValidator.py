from abc import ABC, abstractmethod
import pandas as pd
from ..models.ValidationResult import ValidationResult


class IDataValidator(ABC):
    """Interface para validação de dados"""

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Valida DataFrame conforme regras de negócio

        Args:
            df: DataFrame a ser validado

        Returns:
            ValidationResult com status e mensagens de erro/warning
        """
        pass
