import streamlit as st
import pandas as pd
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.services.ExcelReaderService import ExcelReaderService
from src.services.DashboardService import DashboardService
from src.services.FuzzyColumnMapper import FuzzyColumnMapper
from src.services.CacheService import SQLiteCacheService
from src.services.ReportExporterService import ReportExporterService
from src.services.EmbeddingService import EmbeddingService
from src.services.AssuntoClassifierService import AssuntoClassifierService
from src.services.SubAssuntoClassifierService import SubAssuntoClassifierService
import plotly.io as pio
import io


def apply_filters(df, filters):
    """Aplica os filtros selecionados no dataframe"""
    filtered_df = df.copy()

    if 'DT_REGISTRO_ATENDIMENTO' in filtered_df.columns and filters.get('date_range'):
        date_start, date_end = filters['date_range']
        filtered_df = filtered_df[
            (filtered_df['DT_REGISTRO_ATENDIMENTO'] >= pd.to_datetime(date_start)) &
            (filtered_df['DT_REGISTRO_ATENDIMENTO'] <= pd.to_datetime(date_end))
        ]

    return filtered_df


def auto_classify_data(df):
    """Classifica automaticamente categorias e subcategorias após carregar arquivo"""
    from src.services.TextBuilderService import TextBuilderService

    df_result = df.copy()
    text_builder = TextBuilderService()

    # Verificar se precisa classificar categorias
    if 'DS_ASSUNTO' in df_result.columns:
        num_categorias_nao_class = df_result['DS_ASSUNTO'].apply(text_builder.needs_classification).sum()

        if num_categorias_nao_class > 0:
            try:
                with st.spinner(f"🤖 Classificando {num_categorias_nao_class} categorias automaticamente..."):
                    embedding_service = EmbeddingService(
                        model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
                    )

                    classifier = AssuntoClassifierService(
                        embedding_service=embedding_service,
                        threshold=0.45,
                        k_neighbors=5
                    )

                    df_result = classifier.classify_dataframe(df_result)
                    st.success(f"✅ {num_categorias_nao_class} categorias classificadas!")
            except Exception as e:
                st.warning(f"⚠️ Não foi possível classificar categorias: {str(e)}")

    # Verificar se precisa classificar subcategorias
    if 'SUB_ASSUNTO' in df_result.columns:
        mask_needs_sub = (
            df_result['SUB_ASSUNTO'].isna() |
            (df_result['SUB_ASSUNTO'].str.strip() == '') |
            df_result['SUB_ASSUNTO'].str.upper().str.contains('OUTRO', na=False)
        )
        num_subcategorias_nao_class = mask_needs_sub.sum()

        if num_subcategorias_nao_class > 0:
            try:
                with st.spinner(f"🤖 Classificando {num_subcategorias_nao_class} subcategorias automaticamente..."):
                    embedding_service_sub = EmbeddingService(
                        model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
                    )

                    sub_classifier = SubAssuntoClassifierService(
                        embedding_service=embedding_service_sub,
                        threshold=0.45,
                        k_neighbors=5
                    )

                    df_result = sub_classifier.classify_dataframe(df_result)
                    st.success(f"✅ {num_subcategorias_nao_class} subcategorias classificadas!")
            except Exception as e:
                st.warning(f"⚠️ Não foi possível classificar subcategorias: {str(e)}")

    return df_result


def main():
    st.set_page_config(
        page_title="Monitoramento NIP",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
        /* Dark Mode Global */
        .stApp {
            background-color: #0E1117;
        }

        /* Cores principais */
        :root {
            --primary-color: #6366F1;
            --secondary-color: #8B5CF6;
            --accent-color: #EC4899;
            --bg-dark: #1A1D29;
            --bg-card: #242837;
            --text-primary: #FFFFFF;
            --text-secondary: #B4B4B4;
        }

        /* Animações */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes slideIn {
            from { transform: translateX(-30px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }

        /* Header */
        .main-header {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            padding: 2.5rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            box-shadow: 0 20px 60px rgba(99, 102, 241, 0.4);
            animation: fadeIn 0.8s ease-out;
        }

        .main-header h1 {
            color: white;
            font-size: 2.8rem;
            font-weight: 800;
            margin: 0;
            text-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }

        .main-header p {
            color: rgba(255, 255, 255, 0.95);
            font-size: 1.1rem;
            margin-top: 0.5rem;
        }

        /* Cards Metrics */
        .metric-card {
            background: var(--bg-card);
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            border-left: 5px solid var(--primary-color);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            animation: slideIn 0.6s ease-out;
        }

        .metric-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 12px 40px rgba(99, 102, 241, 0.3);
        }

        .metric-card h3 {
            color: var(--text-secondary);
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin: 0 0 0.5rem 0;
        }

        .metric-card h2 {
            color: var(--text-primary);
            font-size: 2.5rem;
            font-weight: 800;
            margin: 0.5rem 0;
        }

        .metric-card p {
            color: var(--text-secondary);
            margin: 0;
            font-size: 0.9rem;
        }

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            background-color: var(--bg-card);
            padding: 1rem;
            border-radius: 16px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        }

        .stTabs [data-baseweb="tab"] {
            height: 55px;
            background-color: var(--bg-dark);
            border-radius: 12px;
            color: var(--text-secondary);
            font-weight: 600;
            font-size: 1rem;
            border: 2px solid transparent;
            transition: all 0.3s ease;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background-color: #2A2D3A;
            transform: translateY(-2px);
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        }

        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5);
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1A1D29 0%, #242837 100%);
        }

        /* Charts container */
        .chart-container {
            background: var(--bg-card);
            padding: 1.5rem;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            margin: 1rem 0;
        }

        /* Override Streamlit defaults */
        .stMarkdown, .stText {
            color: var(--text-primary) !important;
        }

        /* Dataframe */
        .stDataFrame {
            background: var(--bg-card);
            border-radius: 12px;
        }

        /* Date input */
        .stDateInput > div > div {
            background-color: var(--bg-dark);
        }

        /* Expander */
        .streamlit-expanderHeader {
            background-color: var(--bg-card);
            color: var(--text-primary);
            border-radius: 12px;
        }

        /* Section headers */
        h3 {
            color: var(--text-primary) !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="main-header">
        <h1>📊 Monitoramento NIP</h1>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <p style="margin: 0;">Painel de Reclamações e Ouvidoria</p>
            <p style="margin: 0; font-size: 0.75rem; opacity: 0.85;">Feito por: Cauê Magela [AI-SPEC/CS]</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        with st.expander("📁 Upload de Dados", expanded=True):
            uploaded_files = st.file_uploader(
                "Selecione um ou mais arquivos",
                type=['xlsx', 'xls', 'csv'],
                help="Faça upload de um ou mais arquivos de reclamações (Excel ou CSV)",
                accept_multiple_files=True
            )

        # Indicador de planilha padrão
        default_file_check = Path("data/default/planilha_padrao.xlsx")
        if default_file_check.exists():
            st.info("📌 Planilha padrão ativa")

        with st.expander("🗑️ Limpar Cache/Memória", expanded=False):
            if st.button("Limpar Dados", use_container_width=True, type="secondary"):
                for key in list(st.session_state.keys()):
                    if key != 'cache_service':
                        del st.session_state[key]

                if 'cache_service' in st.session_state:
                    cache = st.session_state.cache_service
                    cache.clear_expired()

                st.success("✅ Cache e memória limpos!")
                st.rerun()

    if 'cache_service' not in st.session_state:
        st.session_state.cache_service = SQLiteCacheService()

    # Verificar e carregar planilha padrão
    default_file_path = Path("data/default/planilha_padrao.xlsx")
    has_default_file = default_file_path.exists()

    # Combinar planilha padrão com uploads adicionais
    files_to_process = []
    file_sources = []  # Para rastrear origem dos arquivos

    if has_default_file:
        files_to_process.append(("default", default_file_path))
        file_sources.append("📌 Planilha Padrão")

    if uploaded_files is not None and len(uploaded_files) > 0:
        for uploaded_file in uploaded_files:
            files_to_process.append(("upload", uploaded_file))
            file_sources.append(f"📤 {uploaded_file.name}")

    if len(files_to_process) > 0:
        # Criar ID único baseado em todos os arquivos (padrão + uploads)
        file_ids = []
        if has_default_file:
            file_ids.append(f"default_{default_file_path.stat().st_size}")
        if uploaded_files is not None and len(uploaded_files) > 0:
            file_ids.extend([f"{f.name}_{f.size}" for f in uploaded_files])

        current_files_id = "_".join(sorted(file_ids))
        last_files_id = st.session_state.get('_last_files_id', None)

        if current_files_id != last_files_id:
            try:
                with st.spinner(f"📥 Carregando {len(files_to_process)} arquivo(s)..."):
                    excel_reader = ExcelReaderService()
                    mapper = FuzzyColumnMapper()

                    all_dfs = []
                    file_stats = []

                    for idx, (source_type, file_data) in enumerate(files_to_process, 1):
                        file_name = "Planilha Padrão" if source_type == "default" else file_data.name
                        with st.spinner(f"📄 Processando arquivo {idx}/{len(files_to_process)}: {file_name}"):
                            raw_df = excel_reader.read_excel(file_data)

                            df_temp = mapper.map_columns(raw_df)

                            date_columns = ['DT_REGISTRO_ATENDIMENTO', 'DT_ATRIBUICAO', 'DT_CONCLUSAO',
                                           'DT_LIMITE', 'DT_LIMITE_CONCLUSAO', 'DT_SAC']
                            for col in date_columns:
                                if col in df_temp.columns:
                                    df_temp[col] = pd.to_datetime(df_temp[col], errors='coerce')

                            all_dfs.append(df_temp)
                            file_stats.append({
                                'nome': file_name,
                                'registros': len(df_temp),
                                'source': file_sources[idx - 1]
                            })

                    with st.spinner("🔗 Juntando arquivos e removendo duplicatas..."):
                        df_combined = pd.concat(all_dfs, ignore_index=True)
                        total_antes = len(df_combined)

                        if 'NU_REGISTRO' in df_combined.columns:
                            df_combined = df_combined.drop_duplicates(subset=['NU_REGISTRO'], keep='first')
                            total_depois = len(df_combined)
                            duplicatas_removidas = total_antes - total_depois
                        else:
                            total_depois = total_antes
                            duplicatas_removidas = 0

                        df = df_combined

                # Log do upload
                upload_description = []
                if has_default_file:
                    upload_description.append("Planilha Padrão")
                if uploaded_files is not None and len(uploaded_files) > 0:
                    upload_description.append(f"{len(uploaded_files)} upload(s)")

                st.session_state.cache_service.log_upload(
                    filename=" + ".join(upload_description),
                    num_records=len(df)
                )

                with st.sidebar:
                    st.success(f"✅ {len(files_to_process)} arquivo(s) processado(s)")

                    with st.expander("📊 Detalhes do Processamento", expanded=False):
                        st.markdown("**Arquivos processados:**")
                        for stat in file_stats:
                            st.write(f"{stat['source']}: {stat['registros']:,} registros")

                        st.markdown("---")
                        st.write(f"**Total antes:** {total_antes:,} registros")
                        if duplicatas_removidas > 0:
                            st.write(f"**Duplicatas removidas:** {duplicatas_removidas:,}")
                        st.write(f"**Total final:** {total_depois:,} registros")

                # Classificação automática
                df = auto_classify_data(df)

                st.session_state['df'] = df
                st.session_state['_last_files_id'] = current_files_id
                st.session_state['classification_done'] = True
                st.session_state['subclassification_done'] = True

            except Exception as e:
                st.error(f"❌ Erro ao processar arquivo(s): {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                st.stop()

    if 'df' in st.session_state:
        df = st.session_state['df']

        with st.sidebar:
            st.markdown("---")

            filters = {}

            with st.expander("📅 Filtros", expanded=False):
                if 'DT_REGISTRO_ATENDIMENTO' in df.columns:
                    date_col = pd.to_datetime(df['DT_REGISTRO_ATENDIMENTO'], errors='coerce')
                    min_date = date_col.min().date()
                    max_date = date_col.max().date()

                    date_range = st.date_input(
                        "Período",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date
                    )

                    if len(date_range) == 2:
                        filters['date_range'] = date_range

                if st.button("🔄 Aplicar Filtros", use_container_width=True):
                    st.rerun()

            st.markdown("---")

            with st.expander("🤖 Status da Classificação", expanded=False):
                num_categorias_nao_class = 0
                num_subcategorias_nao_class = 0
                num_vazios_total = 0

                if 'DS_ASSUNTO' in df.columns:
                    from src.services.TextBuilderService import TextBuilderService
                    text_builder = TextBuilderService()
                    num_categorias_nao_class = df['DS_ASSUNTO'].apply(text_builder.needs_classification).sum()

                    vazios_ds_assunto = (df['DS_ASSUNTO'].isna() | (df['DS_ASSUNTO'].str.strip() == '')).sum()
                else:
                    vazios_ds_assunto = 0

                if 'SUB_ASSUNTO' in df.columns:
                    mask_needs_sub = (
                        df['SUB_ASSUNTO'].isna() |
                        (df['SUB_ASSUNTO'].str.strip() == '') |
                        df['SUB_ASSUNTO'].str.upper().str.contains('OUTRO', na=False)
                    )
                    num_subcategorias_nao_class = mask_needs_sub.sum()

                    vazios_sub_assunto = (df['SUB_ASSUNTO'].isna() | (df['SUB_ASSUNTO'].str.strip() == '')).sum()
                else:
                    vazios_sub_assunto = 0

                num_vazios_total = vazios_ds_assunto + vazios_sub_assunto

                st.markdown(f"📂 **Campos de Categorias não classificadas:** `{num_categorias_nao_class}`")
                st.markdown(f"🔖 **Campos de SubCategorias não classificadas:** `{num_subcategorias_nao_class}`")
                st.markdown(f"⚠️ **Campos Vazios:** `{num_vazios_total}`")

                if num_categorias_nao_class == 0 and num_subcategorias_nao_class == 0:
                    st.success("✅ Todos os dados foram classificados automaticamente!")

        filtered_df = apply_filters(df, filters)

        if len(filtered_df) == 0:
            st.warning("⚠️ Nenhum dado encontrado com os filtros aplicados")
            st.stop()

        dashboard_generator = DashboardService()

        st.markdown("### 📈 Métricas Principais")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 TOTAL</h3>
                <h2>{len(filtered_df):,}</h2>
                <p>Reclamações</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            if 'DS_ASSUNTO' in filtered_df.columns:
                unique_categories = filtered_df['DS_ASSUNTO'].nunique()
            else:
                unique_categories = 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>📂 CATEGORIAS</h3>
                <h2>{unique_categories}</h2>
                <p>Diferentes</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            if 'DS_FILIAL' in filtered_df.columns:
                unique_branches = filtered_df['DS_FILIAL'].nunique()
            else:
                unique_branches = 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>🏢 FILIAIS</h3>
                <h2>{unique_branches}</h2>
                <p>Envolvidas</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            if 'DT_REGISTRO_ATENDIMENTO' in filtered_df.columns:
                date_col = pd.to_datetime(filtered_df['DT_REGISTRO_ATENDIMENTO'], errors='coerce')
                days_range = (date_col.max() - date_col.min()).days
            else:
                days_range = 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>📅 PERÍODO</h3>
                <h2>{days_range}</h2>
                <p>Dias</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### 📥 Exportar")

        col1, col2, col3 = st.columns(3)

        exporter = ReportExporterService()

        with col1:
            print(f"\n[DEBUG EXPORT] ====== VERIFICACAO DE EXPORTACAO ======")
            print(f"[DEBUG EXPORT] Total de registros a exportar: {len(filtered_df)}")
            print(f"[DEBUG EXPORT] Colunas no DataFrame: {list(filtered_df.columns)}")

            if 'SUB_ASSUNTO' in filtered_df.columns:
                outros_count = filtered_df['SUB_ASSUNTO'].str.upper().str.contains('OUTRO', na=False).sum()
                print(f"[DEBUG EXPORT] Registros com 'OUTRO' no SUB_ASSUNTO: {outros_count}")
                print(f"[DEBUG EXPORT] Top 10 SUB_ASSUNTO:")
                print(filtered_df['SUB_ASSUNTO'].value_counts().head(10))

                print(f"\n[DEBUG EXPORT] Amostra de 5 registros SUB_ASSUNTO:")
                print(filtered_df['SUB_ASSUNTO'].head(10).tolist())

            print(f"[DEBUG EXPORT] ====================================\n")

            summary_stats = {
                'Total_Reclamacoes': len(filtered_df),
                'Total_Categorias': filtered_df['DS_ASSUNTO'].nunique() if 'DS_ASSUNTO' in filtered_df.columns else 0,
                'Total_Filiais': filtered_df['DS_FILIAL'].nunique() if 'DS_FILIAL' in filtered_df.columns else 0,
                'Periodo_Dias': (pd.to_datetime(filtered_df['DT_REGISTRO_ATENDIMENTO'], errors='coerce').max() -
                                pd.to_datetime(filtered_df['DT_REGISTRO_ATENDIMENTO'], errors='coerce').min()).days
                                if 'DT_REGISTRO_ATENDIMENTO' in filtered_df.columns else 0
            }

            excel_buffer = io.BytesIO()
            excel_path = Path("temp_report.xlsx")
            if exporter.export_to_excel(filtered_df, summary_stats, excel_path):
                with open(excel_path, 'rb') as f:
                    excel_data = f.read()
                excel_path.unlink()  # Deletar arquivo temporário

                st.download_button(
                    label="📊 Baixar Excel",
                    data=excel_data,
                    file_name=f"relatorio_nip_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

        with col2:
            csv = filtered_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📄 Baixar CSV",
                data=csv,
                file_name=f"dados_nip_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col3:
            json_data = filtered_df.to_json(orient='records', force_ascii=False, indent=2)
            st.download_button(
                label="📋 Baixar JSON",
                data=json_data,
                file_name=f"dados_nip_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        with st.spinner("🎨 Gerando visualizações..."):
            tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🎯 Painel",
                "📊 Análise Temporal",
                "📂 Categorização",
                "🏢 Filiais",
                "📱 Operadoras",
                "📊 Auditoria"
            ])

            filter_outros_assunto = st.session_state.get('classification_done', False)
            filter_outros_subassunto = st.session_state.get('subclassification_done', False)

            weekly_chart = dashboard_generator.generate_weekly_chart(filtered_df)
            monthly_chart = dashboard_generator.generate_monthly_chart(filtered_df)
            category_chart = dashboard_generator.generate_category_chart(filtered_df, filter_outros=filter_outros_assunto)
            subcategory_chart = dashboard_generator.generate_subcategory_chart(filtered_df, filter_outros=filter_outros_subassunto)
            branch_chart = dashboard_generator.generate_branch_ranking(filtered_df)
            operator_chart = dashboard_generator.generate_operator_ranking(filtered_df)

            with tab0:
                st.markdown("### Visão Geral")

                # Variações Percentuais (Mensal e Semanal)
                st.markdown("#### Variação Percentual entre Períodos")

                col_var_mensal, col_var_semanal = st.columns(2)

                # VARIAÇÃO MENSAL
                with col_var_mensal:
                    st.markdown("**Variação Mensal**")
                    if 'DT_REGISTRO_ATENDIMENTO' in filtered_df.columns:
                        date_col = pd.to_datetime(filtered_df['DT_REGISTRO_ATENDIMENTO'], errors='coerce')

                        # Preparar dados mensais
                        filtered_df_temp = filtered_df.copy()
                        filtered_df_temp['month'] = date_col.dt.to_period('M')
                        filtered_df_temp['month_str'] = filtered_df_temp['month'].astype(str)
                        months = sorted(filtered_df_temp['month'].dropna().unique())

                        # Pegar os 2 últimos meses
                        if len(months) >= 2:
                            periodo_1 = months[-2]
                            periodo_2 = months[-1]
                        elif len(months) == 1:
                            periodo_1 = months[0]
                            periodo_2 = months[0]
                        else:
                            periodo_1 = None
                            periodo_2 = None

                        if periodo_1 and periodo_2:
                            df_periodo_1 = filtered_df_temp[filtered_df_temp['month'] == periodo_1]
                            df_periodo_2 = filtered_df_temp[filtered_df_temp['month'] == periodo_2]

                            # Formatar em português (mês/ano)
                            import locale
                            try:
                                locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')
                            except:
                                pass
                            label_1 = periodo_1.strftime('%B/%Y').capitalize()
                            label_2 = periodo_2.strftime('%B/%Y').capitalize()

                            count_1 = len(df_periodo_1)
                            count_2 = len(df_periodo_2)

                        if count_1 > 0:
                            variacao_pct = ((count_2 - count_1) / count_1) * 100
                            variacao_abs = count_2 - count_1

                            if variacao_pct > 0:
                                emoji = "📈"
                                cor = "#FF5C5C"
                                texto = "AUMENTO"
                            elif variacao_pct < 0:
                                emoji = "📉"
                                cor = "#6BCB77"
                                texto = "REDUÇÃO"
                            else:
                                emoji = "➡️"
                                cor = "#FFD93D"
                                texto = "ESTÁVEL"

                            # Card de resultado
                            st.markdown(f"""
                            <div style="background: rgba(0,0,0,0.2); padding: 1.5rem; border-radius: 12px; border-left: 5px solid {cor}; margin-top: 1rem;">
                                <h3 style="margin: 0; color: {cor}; font-size: 1.2rem;">{emoji} {texto}</h3>
                                <h2 style="margin: 0.5rem 0; color: {cor}; font-size: 2.5rem;">{variacao_pct:+.1f}%</h2>
                                <p style="margin: 0; color: #FAFAFA; font-size: 0.9rem;">
                                    {label_1}: <strong>{count_1}</strong><br>
                                    {label_2}: <strong>{count_2}</strong><br>
                                    Variação: <strong>{variacao_abs:+d}</strong>
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                            # Criar figura para download
                            import plotly.graph_objects as go
                            fig_variacao_mensal = go.Figure()
                            fig_variacao_mensal.add_annotation(
                                text=f"<b>{emoji} {texto}</b>", x=0.5, y=0.75,
                                xref="paper", yref="paper", showarrow=False,
                                font=dict(size=28, color=cor, family="Arial Black"), xanchor="center"
                            )
                            fig_variacao_mensal.add_annotation(
                                text=f"<b>{variacao_pct:+.1f}%</b>", x=0.5, y=0.5,
                                xref="paper", yref="paper", showarrow=False,
                                font=dict(size=60, color=cor, family="Arial Black"), xanchor="center"
                            )
                            fig_variacao_mensal.add_annotation(
                                text=f"{label_1}: <b>{count_1}</b> reclamações<br>{label_2}: <b>{count_2}</b> reclamações<br>Variação: <b>{variacao_abs:+d}</b>",
                                x=0.5, y=0.2, xref="paper", yref="paper", showarrow=False,
                                font=dict(size=16, color="#FAFAFA"), xanchor="center"
                            )
                            fig_variacao_mensal.update_layout(
                                width=800, height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='#0E1117',
                                xaxis=dict(visible=False), yaxis=dict(visible=False),
                                margin=dict(t=20, b=20, l=20, r=20),
                                shapes=[dict(type="rect", x0=0, y0=0, x1=1, y1=1, xref="paper", yref="paper",
                                           line=dict(color=cor, width=4), fillcolor="rgba(0,0,0,0.2)")]
                            )

                            img_bytes_mensal = pio.to_image(fig_variacao_mensal, format='png', width=1600, height=800, scale=2)
                            st.download_button(
                                label="📸 Baixar", data=img_bytes_mensal,
                                file_name=f"variacao_mensal_{periodo_1}_{periodo_2}.png",
                                mime="image/png", use_container_width=True, key="dl_var_mensal"
                            )

                # VARIAÇÃO SEMANAL
                with col_var_semanal:
                    st.markdown("**Variação Semanal**")
                    if 'DT_REGISTRO_ATENDIMENTO' in filtered_df.columns:
                        date_col = pd.to_datetime(filtered_df['DT_REGISTRO_ATENDIMENTO'], errors='coerce')

                        # Preparar dados semanais
                        filtered_df_temp = filtered_df.copy()
                        filtered_df_temp['week_start'] = date_col.dt.to_period('W-SUN').dt.start_time
                        filtered_df_temp['week_end'] = filtered_df_temp['week_start'] + pd.Timedelta(days=6)
                        filtered_df_temp['week_label'] = filtered_df_temp['week_start'].dt.strftime('%d/%m') + '-' + filtered_df_temp['week_end'].dt.strftime('%d/%m')

                        week_mapping = filtered_df_temp[['week_label', 'week_start']].drop_duplicates()
                        week_mapping = week_mapping.sort_values('week_start')
                        weeks = week_mapping['week_label'].tolist()

                        # Pegar as 2 últimas semanas
                        if len(weeks) >= 2:
                            periodo_1_label = weeks[-2]
                            periodo_2_label = weeks[-1]
                        elif len(weeks) == 1:
                            periodo_1_label = weeks[0]
                            periodo_2_label = weeks[0]
                        else:
                            periodo_1_label = None
                            periodo_2_label = None

                        if periodo_1_label and periodo_2_label:
                            df_periodo_1 = filtered_df_temp[filtered_df_temp['week_label'] == periodo_1_label]
                            df_periodo_2 = filtered_df_temp[filtered_df_temp['week_label'] == periodo_2_label]

                            label_1 = f"Sem. {periodo_1_label}"
                            label_2 = f"Sem. {periodo_2_label}"

                            count_1 = len(df_periodo_1)
                            count_2 = len(df_periodo_2)

                        if count_1 > 0:
                            variacao_pct = ((count_2 - count_1) / count_1) * 100
                            variacao_abs = count_2 - count_1

                            if variacao_pct > 0:
                                emoji = "📈"
                                cor = "#FF5C5C"
                                texto = "AUMENTO"
                            elif variacao_pct < 0:
                                emoji = "📉"
                                cor = "#6BCB77"
                                texto = "REDUÇÃO"
                            else:
                                emoji = "➡️"
                                cor = "#FFD93D"
                                texto = "ESTÁVEL"

                            # Card de resultado
                            st.markdown(f"""
                            <div style="background: rgba(0,0,0,0.2); padding: 1.5rem; border-radius: 12px; border-left: 5px solid {cor}; margin-top: 1rem;">
                                <h3 style="margin: 0; color: {cor}; font-size: 1.2rem;">{emoji} {texto}</h3>
                                <h2 style="margin: 0.5rem 0; color: {cor}; font-size: 2.5rem;">{variacao_pct:+.1f}%</h2>
                                <p style="margin: 0; color: #FAFAFA; font-size: 0.9rem;">
                                    {label_1}: <strong>{count_1}</strong><br>
                                    {label_2}: <strong>{count_2}</strong><br>
                                    Variação: <strong>{variacao_abs:+d}</strong>
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                            # Criar figura para download
                            fig_variacao_semanal = go.Figure()
                            fig_variacao_semanal.add_annotation(
                                text=f"<b>{emoji} {texto}</b>", x=0.5, y=0.75,
                                xref="paper", yref="paper", showarrow=False,
                                font=dict(size=28, color=cor, family="Arial Black"), xanchor="center"
                            )
                            fig_variacao_semanal.add_annotation(
                                text=f"<b>{variacao_pct:+.1f}%</b>", x=0.5, y=0.5,
                                xref="paper", yref="paper", showarrow=False,
                                font=dict(size=60, color=cor, family="Arial Black"), xanchor="center"
                            )
                            fig_variacao_semanal.add_annotation(
                                text=f"{label_1}: <b>{count_1}</b> reclamações<br>{label_2}: <b>{count_2}</b> reclamações<br>Variação: <b>{variacao_abs:+d}</b>",
                                x=0.5, y=0.2, xref="paper", yref="paper", showarrow=False,
                                font=dict(size=16, color="#FAFAFA"), xanchor="center"
                            )
                            fig_variacao_semanal.update_layout(
                                width=800, height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='#0E1117',
                                xaxis=dict(visible=False), yaxis=dict(visible=False),
                                margin=dict(t=20, b=20, l=20, r=20),
                                shapes=[dict(type="rect", x0=0, y0=0, x1=1, y1=1, xref="paper", yref="paper",
                                           line=dict(color=cor, width=4), fillcolor="rgba(0,0,0,0.2)")]
                            )

                            img_bytes_semanal = pio.to_image(fig_variacao_semanal, format='png', width=1600, height=800, scale=2)
                            st.download_button(
                                label="📸 Baixar", data=img_bytes_semanal,
                                file_name=f"variacao_semanal_{periodo_1_label.replace('/', '-')}_{periodo_2_label.replace('/', '-')}.png",
                                mime="image/png", use_container_width=True, key="dl_var_semanal"
                            )

                st.markdown("<br>", unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(monthly_chart, use_container_width=True, key="exec_monthly",
                                   config={'displayModeBar': True, 'scrollZoom': True})
                    img_bytes = pio.to_image(monthly_chart, format='png', width=1920, height=1080, scale=2)
                    st.download_button(
                        label="📸 Baixar Gráfico (HD)",
                        data=img_bytes,
                        file_name="grafico_mensal.png",
                        mime="image/png",
                        use_container_width=True
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(weekly_chart, use_container_width=True, key="exec_weekly",
                                   config={'displayModeBar': True, 'scrollZoom': True})
                    img_bytes = pio.to_image(weekly_chart, format='png', width=1920, height=1080, scale=2)
                    st.download_button(
                        label="📸 Baixar Gráfico (HD)",
                        data=img_bytes,
                        file_name="grafico_semanal.png",
                        mime="image/png",
                        use_container_width=True
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                col3, col4 = st.columns(2)
                with col3:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(category_chart, use_container_width=True, key="exec_category",
                                   config={'displayModeBar': True, 'scrollZoom': True})
                    img_bytes = pio.to_image(category_chart, format='png', width=1920, height=1080, scale=2)
                    st.download_button(
                        label="📸 Baixar Gráfico (HD)",
                        data=img_bytes,
                        file_name="grafico_categorias.png",
                        mime="image/png",
                        use_container_width=True
                    )

                    if operator_chart:
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        st.plotly_chart(operator_chart, use_container_width=True, key="exec_operator",
                                       config={'displayModeBar': True, 'scrollZoom': True})
                        img_bytes = pio.to_image(operator_chart, format='png', width=1920, height=1080, scale=2)
                        st.download_button(
                            label="📸 Baixar Gráfico (HD)",
                            data=img_bytes,
                            file_name="grafico_operadoras.png",
                            mime="image/png",
                            use_container_width=True
                        )
                        st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)

                with col4:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(subcategory_chart, use_container_width=True, key="exec_subcategory",
                                   config={'displayModeBar': True, 'scrollZoom': True})
                    img_bytes = pio.to_image(subcategory_chart, format='png', width=1920, height=1080, scale=2)
                    st.download_button(
                        label="📸 Baixar Gráfico (HD)",
                        data=img_bytes,
                        file_name="grafico_subcategorias.png",
                        mime="image/png",
                        use_container_width=True
                    )

                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(branch_chart, use_container_width=True, key="exec_branch",
                                   config={'displayModeBar': True, 'scrollZoom': True})
                    img_bytes = pio.to_image(branch_chart, format='png', width=1920, height=1080, scale=2)
                    st.download_button(
                        label="📸 Baixar Gráfico (HD)",
                        data=img_bytes,
                        file_name="grafico_filiais.png",
                        mime="image/png",
                        use_container_width=True
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(monthly_chart, use_container_width=True, key="temp_monthly",
                                   config={'displayModeBar': True, 'scrollZoom': True})
                    img_bytes = pio.to_image(monthly_chart, format='png', width=1920, height=1080, scale=2)
                    st.download_button(
                        label="📸 Baixar Gráfico (HD)",
                        data=img_bytes,
                        file_name="temporal_mensal.png",
                        mime="image/png",
                        use_container_width=True,
                        key="dl_temp_monthly"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(weekly_chart, use_container_width=True, key="temp_weekly",
                                   config={'displayModeBar': True, 'scrollZoom': True})
                    img_bytes = pio.to_image(weekly_chart, format='png', width=1920, height=1080, scale=2)
                    st.download_button(
                        label="📸 Baixar Gráfico (HD)",
                        data=img_bytes,
                        file_name="temporal_semanal.png",
                        mime="image/png",
                        use_container_width=True,
                        key="dl_temp_weekly"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                if 'DT_REGISTRO_ATENDIMENTO' in filtered_df.columns:
                    st.markdown("#### 📋 Detalhamento por Período")
                    temporal_df = filtered_df.copy()
                    temporal_df['Mês'] = pd.to_datetime(temporal_df['DT_REGISTRO_ATENDIMENTO']).dt.to_period('M').astype(str)
                    summary = temporal_df.groupby('Mês').size().reset_index(name='Quantidade')
                    st.dataframe(summary, use_container_width=True)

            with tab2:
                st.markdown("### Análise por Categorias")

                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.plotly_chart(category_chart, use_container_width=True, key="cat_main",
                               config={'displayModeBar': True, 'scrollZoom': True})
                img_bytes = pio.to_image(category_chart, format='png', width=1920, height=1080, scale=2)
                st.download_button(
                    label="📸 Baixar Gráfico (HD)",
                    data=img_bytes,
                    file_name="categorias.png",
                    mime="image/png",
                    use_container_width=True,
                    key="dl_cat_main"
                )
                st.markdown('</div>', unsafe_allow_html=True)

                if subcategory_chart:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(subcategory_chart, use_container_width=True, key="cat_sub",
                                   config={'displayModeBar': True, 'scrollZoom': True})
                    img_bytes = pio.to_image(subcategory_chart, format='png', width=1920, height=1080, scale=2)
                    st.download_button(
                        label="📸 Baixar Gráfico (HD)",
                        data=img_bytes,
                        file_name="subcategorias.png",
                        mime="image/png",
                        use_container_width=True,
                        key="dl_cat_sub"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                if 'DS_ASSUNTO' in filtered_df.columns:
                    st.markdown("#### 📋 Ranking de Categorias")
                    cat_summary = filtered_df['DS_ASSUNTO'].value_counts().reset_index()
                    cat_summary.columns = ['Categoria', 'Quantidade']
                    cat_summary['Percentual'] = (cat_summary['Quantidade'] / cat_summary['Quantidade'].sum() * 100).round(2)
                    st.dataframe(cat_summary, use_container_width=True)

            with tab3:
                st.markdown("### Análise por Filiais")

                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.plotly_chart(branch_chart, use_container_width=True, key="branch_main",
                               config={'displayModeBar': True, 'scrollZoom': True})
                img_bytes = pio.to_image(branch_chart, format='png', width=1920, height=1080, scale=2)
                st.download_button(
                    label="📸 Baixar Gráfico (HD)",
                    data=img_bytes,
                    file_name="filiais.png",
                    mime="image/png",
                    use_container_width=True,
                    key="dl_branch_main"
                )
                st.markdown('</div>', unsafe_allow_html=True)

                if 'DS_FILIAL' in filtered_df.columns:
                    st.markdown("#### 📋 Ranking de Filiais")
                    branch_summary = filtered_df['DS_FILIAL'].value_counts().reset_index()
                    branch_summary.columns = ['Filial', 'Quantidade']
                    branch_summary['Percentual'] = (branch_summary['Quantidade'] / branch_summary['Quantidade'].sum() * 100).round(2)
                    st.dataframe(branch_summary, use_container_width=True)

            with tab4:
                st.markdown("### Análise por Operadoras")

                if operator_chart:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(operator_chart, use_container_width=True, key="operator_main",
                                   config={'displayModeBar': True, 'scrollZoom': True})
                    img_bytes = pio.to_image(operator_chart, format='png', width=1920, height=1080, scale=2)
                    st.download_button(
                        label="📸 Baixar Gráfico (HD)",
                        data=img_bytes,
                        file_name="operadoras.png",
                        mime="image/png",
                        use_container_width=True,
                        key="dl_operator_main"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                    if 'OPERADORA' in filtered_df.columns:
                        st.markdown("#### 📋 Ranking de Operadoras")
                        operator_summary = filtered_df['OPERADORA'].value_counts().reset_index()
                        operator_summary.columns = ['Operadora', 'Quantidade']
                        operator_summary['Percentual'] = (operator_summary['Quantidade'] / operator_summary['Quantidade'].sum() * 100).round(2)
                        st.dataframe(operator_summary, use_container_width=True)
                else:
                    st.info("⚠️ Coluna 'OPERADORA' não encontrada nos dados")

            with tab5:
                st.markdown("### 📊 Auditoria")

                cache = st.session_state.cache_service

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 📁 Histórico de Uploads")
                    history = cache.get_upload_history(limit=10)
                    if history:
                        df_history = pd.DataFrame(history)
                        st.dataframe(df_history[['filename', 'num_records', 'uploaded_at']], use_container_width=True)
                    else:
                        st.info("Nenhum histórico ainda")

                with col2:
                    st.markdown("#### 📈 Estatísticas do Sistema")
                    stats = cache.get_query_stats(hours=24)

                    st.metric("Total de Operações (24h)", stats.get('total_queries', 0))

                    if stats.get('query_types'):
                        st.markdown("**Operações por Tipo:**")
                        for qtype, count in stats['query_types'].items():
                            st.write(f"- {qtype}: {count}")

        with st.expander("📄 Ver Dados Brutos"):
            st.dataframe(filtered_df, use_container_width=True)

            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name="dados_filtrados.csv",
                mime="text/csv",
                use_container_width=True
            )

    else:
        st.info("👈 Faça upload de um arquivo Excel na barra lateral para começar a análise")


if __name__ == "__main__":
    main()
