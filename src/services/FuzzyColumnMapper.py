import pandas as pd
from typing import Dict, List
from rapidfuzz import fuzz, process
from ..interfaces.IColumnMapper import IColumnMapper


class FuzzyColumnMapper(IColumnMapper):
    """Mapeamento fuzzy de colunas usando similaridade de strings"""

    REQUIRED_COLUMNS = {
        'NU_REGISTRO': ['nu_registro', 'nu registro', 'numero registro', 'numero_registro', 'id', 'registro'],
        'DS_ASSUNTO': ['ds_assunto', 'ds assunto', 'assunto', 'categoria', 'descricao assunto'],
        'CD_USUARIO': ['cd_usuario', 'cd usuario', 'codigo usuario', 'codigo_usuario', 'usuario', 'user'],
        'SUB_ASSUNTO': ['sub_assunto', 'sub assunto', 'subassunto', 'subcategoria'],
        'DS_OBSERVACAO': ['ds_observacao', 'ds observacao', 'observacao', 'descricao', 'texto', 'reclamacao'],
        'DT_REGISTRO_ATENDIMENTO': ['dt_registro_atendimento', 'dt registro atendimento', 'data registro',
                                     'data_registro', 'dt_registro', 'dt registro', 'data atendimento',
                                     'data_atendimento', 'dt_atendimento'],
        'DS_FILIAL': ['ds_filial', 'ds filial', 'filial', 'unidade', 'loja'],
        'OPERADORA': ['operadora', 'operator', 'operador', 'empresa', 'carrier']
    }

    def __init__(self, threshold: int = 70):
        """
        Args:
            threshold: Limiar mínimo de similaridade (0-100) para aceitar match
        """
        self.threshold = threshold
        self.mapping_report: Dict[str, str] = {}

    def map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mapeia colunas usando fuzzy matching"""
        self.mapping_report = {}
        df_mapped = df.copy()

        current_columns = {col: self._normalize(col) for col in df.columns}

        for expected_col, variations in self.REQUIRED_COLUMNS.items():
            best_match = self._find_best_match(
                expected_col,
                current_columns,
                variations
            )

            if best_match:
                original_col = best_match
                if original_col != expected_col:
                    df_mapped = df_mapped.rename(columns={original_col: expected_col})
                    self.mapping_report[original_col] = expected_col
                else:
                    self.mapping_report[original_col] = expected_col + " (sem alteração)"

        return df_mapped

    def _normalize(self, text: str) -> str:
        """Normaliza texto para comparação"""
        return text.strip().lower().replace('_', ' ')

    def _find_best_match(
        self,
        expected_col: str,
        current_columns: Dict[str, str],
        variations: List[str]
    ) -> str:
        """
        Encontra melhor correspondência para coluna esperada

        Returns:
            Nome original da coluna ou None se não encontrar
        """
        expected_normalized = self._normalize(expected_col)

        expected_variations = [expected_normalized] + [self._normalize(v) for v in variations]

        best_score = 0
        best_original_col = None

        for original_col, normalized_col in current_columns.items():
            for variation in expected_variations:
                score = fuzz.token_sort_ratio(normalized_col, variation)

                if score > best_score and score >= self.threshold:
                    best_score = score
                    best_original_col = original_col

        return best_original_col

    def get_mapping_report(self) -> Dict[str, str]:
        """Retorna relatório do mapeamento"""
        return self.mapping_report
