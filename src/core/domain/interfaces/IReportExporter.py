from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any


class IReportExporter(ABC):
    """Interface para exportação de relatórios"""

    @abstractmethod
    def export_to_excel(self, df: pd.DataFrame, summary_stats: Dict, output_path: Path) -> bool:
        """Exporta dados para Excel com múltiplas abas"""
        pass

    @abstractmethod
    def export_to_csv(self, df: pd.DataFrame, output_path: Path) -> bool:
        """Exporta dados para CSV"""
        pass
