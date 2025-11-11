"""
Interface para serviço de classificação de assuntos.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd


class IAssuntoClassifier(ABC):
    """
    Interface para serviço de classificação automática de DS_ASSUNTO.
    """

    @abstractmethod
    def classify_assunto(self, texto: str) -> Dict[str, Any]:
        """
        Classifica um texto único retornando a categoria de assunto prevista.

        Args:
            texto: Texto a ser classificado

        Returns:
            dict: {
                'categoria': str,           # Categoria prevista
                'confianca': float,         # Score de confiança (0-1)
                'metodo': str,             # 'auto' ou 'manual_review'
                'top_similares': List[str] # Top 3 textos similares usados na classificação
            }
        """
        pass

    @abstractmethod
    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classifica todos os registros de um DataFrame que precisam de classificação.

        Args:
            df: DataFrame com colunas de texto

        Returns:
            DataFrame: DataFrame enriquecido com colunas:
                - DS_ASSUNTO (preenchido)
                - AUTO_CATEGORIZADO_ASSUNTO (bool)
                - CONFIANCA_ASSUNTO (float)
                - REQUER_REVISAO_ASSUNTO (bool)
        """
        pass

    @abstractmethod
    def get_classification_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Retorna estatísticas sobre a classificação realizada.

        Args:
            df: DataFrame classificado

        Returns:
            dict: Estatísticas de classificação
        """
        pass
