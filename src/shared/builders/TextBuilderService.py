"""
Service for building text representations from complaint records.
"""
import pandas as pd
from typing import Any


class TextBuilderService:
    """
    Service responsible for building text representations from DataFrame rows.
    Combines multiple text columns to create rich context for embeddings.
    """

    @staticmethod
    def needs_classification(value: Any) -> bool:
        """
        Check if a DS_ASSUNTO value needs classification.

        Args:
            value: Value to check

        Returns:
            bool: True if value is empty, None, or 'Outros'/'Outro'
        """
        if pd.isna(value):
            return True

        if isinstance(value, str):
            cleaned = value.strip().upper()
            if cleaned == '' or cleaned in ['OUTROS', 'OUTRO']:
                return True

        return False

    @staticmethod
    def build_text_from_row(row: pd.Series) -> str:
        """
        Build a text representation from a DataFrame row.
        Combines multiple text columns to create context.

        Args:
            row: Row from DataFrame

        Returns:
            str: Combined text for embedding generation
        """
        parts = []

        text_columns = [
            'DS_OBSERVACAO',
            'DS_MOTIVO',
            'DS_TRATATIVA',
            'DS_RETORNO',
            'SUB_ASSUNTO'
        ]

        for col in text_columns:
            if col in row.index and pd.notna(row[col]):
                text = str(row[col]).strip()
                if text and len(text) > 5:  # Only add meaningful text
                    parts.append(text)

        combined = ' '.join(parts).strip()

        if len(combined) > 2000:
            combined = combined[:2000]

        return combined

    @staticmethod
    def validate_text_length(text: str, min_length: int = 10) -> bool:
        """
        Validate if text has sufficient length for classification.

        Args:
            text: Text to validate
            min_length: Minimum required length

        Returns:
            bool: True if text is valid
        """
        return len(text.strip()) >= min_length
