from abc import ABC, abstractmethod
import pandas as pd
from typing import Any


class IDashboardGenerator(ABC):
    """Interface para geração de dashboards"""

    @abstractmethod
    def generate_weekly_chart(self, df: pd.DataFrame) -> Any:
        """Gera gráfico de reclamações semanais"""
        pass

    @abstractmethod
    def generate_monthly_chart(self, df: pd.DataFrame) -> Any:
        """Gera gráfico de reclamações mensais"""
        pass

    @abstractmethod
    def generate_category_chart(self, df: pd.DataFrame, filter_outros: bool = False) -> Any:
        """Gera gráfico de barras de categorias

        Args:
            df: DataFrame com os dados
            filter_outros: Se True, remove "Outros" do gráfico (após classificação)
        """
        pass

    @abstractmethod
    def generate_subcategory_chart(self, df: pd.DataFrame, filter_outros: bool = False) -> Any:
        """Gera gráfico de barras de subcategorias

        Args:
            df: DataFrame com os dados
            filter_outros: Se True, remove "Outros" do gráfico (após classificação)
        """
        pass

    @abstractmethod
    def generate_branch_ranking(self, df: pd.DataFrame) -> Any:
        """Gera ranking de filiais"""
        pass

    @abstractmethod
    def generate_operator_ranking(self, df: pd.DataFrame) -> Any:
        """Gera ranking de operadoras"""
        pass
