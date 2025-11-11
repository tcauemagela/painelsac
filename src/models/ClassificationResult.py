"""
Model for classification results
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ClassificationResult:
    """
    Representa o resultado de uma classificação de assunto.
    """
    categoria: Optional[str]
    confianca: float
    metodo: str  # 'auto' ou 'manual_review'
    top_similares: List[str]

    def needs_review(self) -> bool:
        """
        Indica se a classificação requer revisão manual.

        Returns:
            bool: True se categoria é None ou metodo é 'manual_review'
        """
        return self.categoria is None or self.metodo == 'manual_review'

    def to_dict(self) -> dict:
        """
        Converte para dicionário.

        Returns:
            dict: Representação em dicionário
        """
        return {
            'categoria': self.categoria,
            'confianca': self.confianca,
            'metodo': self.metodo,
            'top_similares': self.top_similares,
            'needs_review': self.needs_review()
        }
