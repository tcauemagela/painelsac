import pandas as pd
from typing import BinaryIO, Union
from ...core.domain.interfaces.IExcelReader import IExcelReader
import io


class ExcelReaderService(IExcelReader):
    """Implementação do leitor de Excel e CSV"""

    def read_excel(self, file: Union[BinaryIO, str]) -> pd.DataFrame:
        """Lê arquivo Excel ou CSV e retorna DataFrame"""
        try:
            file_name = getattr(file, 'name', '')

            if file_name.lower().endswith('.csv'):
                return self._read_csv(file)
            else:
                df = pd.read_excel(file)
                return df
        except Exception as e:
            raise ValueError(f"Erro ao ler arquivo: {str(e)}")

    def _read_csv(self, file: BinaryIO) -> pd.DataFrame:
        """Lê arquivo CSV com detecção automática de separador e encoding"""
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        separators = [';', ',', '\t']

        for encoding in encodings:
            for separator in separators:
                try:
                    file.seek(0)  # Voltar ao início do arquivo
                    df = pd.read_csv(file, sep=separator, encoding=encoding, low_memory=False)

                    if len(df.columns) > 1:
                        return df
                except:
                    continue

        file.seek(0)
        return pd.read_csv(file, low_memory=False)
