import pandas as pd
from pathlib import Path
from typing import Dict
from ...core.domain.interfaces.IReportExporter import IReportExporter


class ReportExporterService(IReportExporter):
    """Serviço de exportação de relatórios"""

    def export_to_excel(self, df: pd.DataFrame, summary_stats: Dict, output_path: Path) -> bool:
        """Exporta para Excel com múltiplas abas"""
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Dados Completos', index=False)

                if 'DS_ASSUNTO' in df.columns:
                    cat_summary = df['DS_ASSUNTO'].value_counts().reset_index()
                    cat_summary.columns = ['Categoria', 'Quantidade']
                    cat_summary['Percentual'] = (cat_summary['Quantidade'] / cat_summary['Quantidade'].sum() * 100).round(2)
                    cat_summary.to_excel(writer, sheet_name='Resumo Categorias', index=False)

                if 'DS_FILIAL' in df.columns:
                    branch_summary = df['DS_FILIAL'].value_counts().reset_index()
                    branch_summary.columns = ['Filial', 'Quantidade']
                    branch_summary['Percentual'] = (branch_summary['Quantidade'] / branch_summary['Quantidade'].sum() * 100).round(2)
                    branch_summary.to_excel(writer, sheet_name='Resumo Filiais', index=False)

                if 'DT_REGISTRO_ATENDIMENTO' in df.columns:
                    temporal_df = df.copy()
                    temporal_df['DT_REGISTRO_ATENDIMENTO'] = pd.to_datetime(temporal_df['DT_REGISTRO_ATENDIMENTO'], errors='coerce')
                    temporal_df['Mês'] = temporal_df['DT_REGISTRO_ATENDIMENTO'].dt.to_period('M').astype(str)
                    monthly_summary = temporal_df.groupby('Mês').size().reset_index(name='Quantidade')
                    monthly_summary.to_excel(writer, sheet_name='Análise Mensal', index=False)

                stats_df = pd.DataFrame([summary_stats])
                stats_df.to_excel(writer, sheet_name='Estatísticas', index=False)

            return True
        except Exception as e:
            print(f"Erro ao exportar Excel: {e}")
            return False

    def export_to_csv(self, df: pd.DataFrame, output_path: Path) -> bool:
        """Exporta para CSV"""
        try:
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            return True
        except Exception as e:
            print(f"Erro ao exportar CSV: {e}")
            return False
