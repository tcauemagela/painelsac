import pandas as pd
from ..interfaces.IDataValidator import IDataValidator
from ..models.ValidationResult import ValidationResult


class DataValidatorService(IDataValidator):
    """Implementação do validador de dados"""

    REQUIRED_COLUMNS = [
        'NU_REGISTRO',
        'DS_ASSUNTO',
        'CD_USUARIO',
        'SUB_ASSUNTO',
        'DS_OBSERVACAO',
        'DT_REGISTRO_ATENDIMENTO',
        'DS_FILIAL'
    ]

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Valida DataFrame conforme regras de negócio"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])

        missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            result.add_error(f"Colunas faltando: {', '.join(missing_columns)}")
            return result

        duplicates = df[df.duplicated(subset=['NU_REGISTRO'], keep=False)]
        if not duplicates.empty:
            duplicate_ids = duplicates['NU_REGISTRO'].unique().tolist()
            result.add_error(
                f"NU_REGISTRO duplicados encontrados: {duplicate_ids[:10]}"
                + (f" e mais {len(duplicate_ids) - 10}..." if len(duplicate_ids) > 10 else "")
            )

        for col in self.REQUIRED_COLUMNS:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                result.add_error(f"Coluna '{col}' possui {null_count} valores nulos")

        try:
            pd.to_datetime(df['DT_REGISTRO_ATENDIMENTO'], errors='coerce')
            invalid_dates = df['DT_REGISTRO_ATENDIMENTO'].isnull().sum()
            if invalid_dates > 0:
                result.add_warning(f"{invalid_dates} datas inválidas encontradas em DT_REGISTRO_ATENDIMENTO")
        except Exception as e:
            result.add_error(f"Erro ao validar datas: {str(e)}")

        return result
