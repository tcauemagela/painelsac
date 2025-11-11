from abc import ABC, abstractmethod
import pandas as pd
from typing import BinaryIO


class IExcelReader(ABC):
    """Interface para leitura de arquivos Excel"""

    @abstractmethod
    def read_excel(self, file: BinaryIO) -> pd.DataFrame:
        """
        Lê arquivo Excel e retorna DataFrame

        Args:
            file: Arquivo Excel em formato binário

        Returns:
            DataFrame com os dados do Excel
        """
        pass
