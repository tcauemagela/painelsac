from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict


class IColumnMapper(ABC):
    """Interface para mapeamento de colunas do Excel"""

    @abstractmethod
    def map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mapeia colunas do DataFrame para padrão esperado

        Args:
            df: DataFrame com colunas originais

        Returns:
            DataFrame com colunas renomeadas para padrão
        """
        pass

    @abstractmethod
    def get_mapping_report(self) -> Dict[str, str]:
        """
        Retorna relatório do mapeamento realizado

        Returns:
            Dicionário com mapeamento original -> padrão
        """
        pass
