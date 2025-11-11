import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Any, Tuple
from ..interfaces.IDashboardGenerator import IDashboardGenerator


class DashboardService(IDashboardGenerator):
    """Implementação do gerador de dashboards"""

    COLORS = {
        'primary': '#00D9FF',      # Ciano vibrante
        'secondary': '#FF6B9D',    # Rosa vibrante
        'accent': '#FFD93D',       # Amarelo vibrante
        'success': '#6BCB77',      # Verde vibrante
        'warning': '#FF8C42',      # Laranja vibrante
        'danger': '#FF5C5C',       # Vermelho vibrante
        'purple': '#BD00FF',       # Roxo vibrante
        'grid': '#2d2d2d',         # Grid sutil
        'text': '#FAFAFA',         # Texto claro
        'background': 'rgba(0,0,0,0)'  # Fundo transparente
    }

    def generate_weekly_chart(self, df: pd.DataFrame) -> Any:
        """Gera gráfico de reclamações semanais com picos anotados"""
        df['DT_REGISTRO_ATENDIMENTO'] = pd.to_datetime(df['DT_REGISTRO_ATENDIMENTO'], errors='coerce')

        df_clean = df.dropna(subset=['DT_REGISTRO_ATENDIMENTO'])

        df_clean['week_start'] = df_clean['DT_REGISTRO_ATENDIMENTO'].dt.to_period('W-SUN').dt.start_time
        df_clean['week_end'] = df_clean['week_start'] + pd.Timedelta(days=6)
        df_clean['week_label'] = df_clean['week_start'].dt.strftime('%d/%m') + '-' + df_clean['week_end'].dt.strftime('%d/%m')

        weekly_counts = df_clean.groupby(['week_label', 'week_start']).size().reset_index(name='count')
        weekly_counts = weekly_counts.sort_values('week_start')

        hover_texts = []
        for week_label in weekly_counts['week_label']:
            week_data = df_clean[df_clean['week_label'] == week_label]
            top_5 = week_data['DS_ASSUNTO'].value_counts().head(5)
            hover_text = f"<b>{week_label}</b><br><br><b>Top 5 Categorias:</b><br>"
            hover_text += "<br>".join([f"• {cat}: {count}" for cat, count in top_5.items()])
            hover_texts.append(hover_text)

        weekly_counts['hover_text'] = hover_texts

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=weekly_counts['week_label'],
            y=weekly_counts['count'],
            mode='lines+markers+text',
            name='Reclamações',
            text=weekly_counts['count'],
            textposition='top center',
            textfont=dict(size=9, color=self.COLORS['text'], family='Arial Black'),
            line=dict(color=self.COLORS['primary'], width=3),
            marker=dict(size=10, color=self.COLORS['primary'],
                       line=dict(color=self.COLORS['text'], width=2)),
            fill='tozeroy',
            fillcolor=f"rgba(0, 217, 255, 0.2)",
            hovertext=weekly_counts['hover_text'],
            hoverinfo='text'
        ))

        fig.update_layout(
            title={
                'text': "📅 Reclamações por Semana",
                'font': {'size': 24, 'color': self.COLORS['text'], 'family': 'Arial Black'}
            },
            xaxis_title="Período",
            yaxis_title="Quantidade de Reclamações",
            hovermode='closest',
            height=550,
            width=1400,
            plot_bgcolor=self.COLORS['background'],
            paper_bgcolor=self.COLORS['background'],
            font=dict(color=self.COLORS['text'], size=14),
            xaxis=dict(
                showgrid=True,
                gridcolor=self.COLORS['grid'],
                title_font=dict(size=16, color=self.COLORS['text']),
                range=[-0.5, len(weekly_counts) - 0.5]
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=self.COLORS['grid'],
                title_font=dict(size=16, color=self.COLORS['text']),
                range=[0, weekly_counts['count'].max() * 1.25]
            ),
            margin=dict(t=120, b=80, l=120, r=120),
            showlegend=False,
            dragmode='pan'
        )

        return fig

    def generate_monthly_chart(self, df: pd.DataFrame) -> Any:
        """Gera gráfico de reclamações mensais com picos anotados"""
        df['DT_REGISTRO_ATENDIMENTO'] = pd.to_datetime(df['DT_REGISTRO_ATENDIMENTO'], errors='coerce')

        df_clean = df.dropna(subset=['DT_REGISTRO_ATENDIMENTO'])

        df_clean['month'] = df_clean['DT_REGISTRO_ATENDIMENTO'].dt.to_period('M')
        df_clean['month_label'] = df_clean['month'].dt.strftime('%b/%Y')
        df_clean['month_sort'] = df_clean['DT_REGISTRO_ATENDIMENTO'].dt.to_period('M')

        monthly_counts = df_clean.groupby(['month_label', 'month_sort']).size().reset_index(name='count')
        monthly_counts = monthly_counts.sort_values('month_sort')

        hover_texts = []
        for month_label in monthly_counts['month_label']:
            month_data = df_clean[df_clean['month_label'] == month_label]
            top_5 = month_data['DS_ASSUNTO'].value_counts().head(5)
            hover_text = f"<b>{month_label}</b><br><br><b>Top 5 Categorias:</b><br>"

            # Converter para lista de tuplas para iterar com índice
            top_5_items = list(top_5.items())
            lines = []

            for i, (cat, count) in enumerate(top_5_items):
                line = f"• {cat}: {count}"

                # Adicionar variação percentual se não for a primeira linha
                if i > 0:
                    prev_count = top_5_items[i-1][1]
                    variation = ((count - prev_count) / prev_count) * 100

                    # Determinar cor baseado se aumentou ou diminuiu
                    if variation > 0:
                        color = "red"
                        sign = "+"
                    elif variation < 0:
                        color = "green"
                        sign = ""
                    else:
                        color = "gray"
                        sign = ""

                    # Adicionar a variação colorida
                    line += f" <span style='color:{color}'>({sign}{variation:.1f}%)</span>"

                lines.append(line)

            hover_text += "<br>".join(lines)
            hover_texts.append(hover_text)

        monthly_counts['hover_text'] = hover_texts

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=monthly_counts['month_label'],
            y=monthly_counts['count'],
            mode='lines+markers+text',
            name='Reclamações',
            text=monthly_counts['count'],
            textposition='top center',
            textfont=dict(size=11, color=self.COLORS['text'], family='Arial Black'),
            line=dict(color=self.COLORS['success'], width=3),
            marker=dict(size=10, color=self.COLORS['success'],
                       line=dict(color=self.COLORS['text'], width=2)),
            fill='tozeroy',
            fillcolor=f"rgba(107, 203, 119, 0.2)",
            hovertext=monthly_counts['hover_text'],
            hoverinfo='text'
        ))

        # Adicionar anotações de variação percentual nas linhas
        annotations = []
        for i in range(1, len(monthly_counts)):
            prev_count = monthly_counts.iloc[i-1]['count']
            curr_count = monthly_counts.iloc[i]['count']
            variation = ((curr_count - prev_count) / prev_count) * 100

            # Determinar cor e sinal
            if variation > 0:
                color = self.COLORS['danger']  # Vermelho para aumento
                sign = "+"
            elif variation < 0:
                color = self.COLORS['success']  # Verde para diminuição
                sign = ""
            else:
                continue  # Não mostrar se não houver variação

            # Posição da anotação (no meio da linha entre dois pontos)
            x_pos = i - 0.5  # Meio do caminho entre os dois meses
            y_pos = (prev_count + curr_count) / 2  # Meio da altura

            annotations.append(dict(
                x=x_pos,
                y=y_pos,
                text=f"{sign}{variation:.1f}%",
                showarrow=False,
                font=dict(
                    size=10,
                    color=color,
                    family='Arial Black'
                ),
                bgcolor='rgba(0,0,0,0.7)',
                bordercolor=color,
                borderwidth=1,
                borderpad=3
            ))

        fig.update_layout(
            title={
                'text': "📆 Reclamações por Mês",
                'font': {'size': 24, 'color': self.COLORS['text'], 'family': 'Arial Black'}
            },
            xaxis_title="Período",
            yaxis_title="Quantidade de Reclamações",
            hovermode='closest',
            height=550,
            width=1400,
            plot_bgcolor=self.COLORS['background'],
            paper_bgcolor=self.COLORS['background'],
            font=dict(color=self.COLORS['text'], size=14),
            xaxis=dict(
                showgrid=True,
                gridcolor=self.COLORS['grid'],
                title_font=dict(size=16, color=self.COLORS['text']),
                range=[-0.5, len(monthly_counts) - 0.5]
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=self.COLORS['grid'],
                title_font=dict(size=16, color=self.COLORS['text']),
                range=[0, monthly_counts['count'].max() * 1.25]
            ),
            margin=dict(t=120, b=80, l=120, r=120),
            showlegend=False,
            dragmode='pan',
            annotations=annotations
        )

        return fig

    def generate_category_chart(self, df: pd.DataFrame, filter_outros: bool = False) -> Any:
        """Gera gráfico de barras de categorias

        Args:
            df: DataFrame com os dados
            filter_outros: Se True, remove "Outros" do gráfico (após classificação)
        """
        if filter_outros:
            df = df[
                (~df['DS_ASSUNTO'].str.upper().isin(['OUTROS', 'OUTRO'])) &
                (df['DS_ASSUNTO'].notna()) &
                (df['DS_ASSUNTO'].str.strip() != '')
            ].copy()

        category_counts = df['DS_ASSUNTO'].value_counts().reset_index()
        category_counts.columns = ['Categoria', 'Quantidade']

        fig = px.bar(
            category_counts,
            x='Quantidade',
            y='Categoria',
            orientation='h',
            title='📋 Categorias de Reclamações',
            text='Quantidade',
            color='Quantidade',
            color_continuous_scale=[
                [0, self.COLORS['success']],
                [0.5, self.COLORS['accent']],
                [1, self.COLORS['danger']]
            ]
        )

        fig.update_traces(
            textposition='outside',
            textfont=dict(size=11, color=self.COLORS['text'])
        )
        fig.update_layout(
            height=max(400, len(category_counts) * 30),
            plot_bgcolor=self.COLORS['background'],
            paper_bgcolor=self.COLORS['background'],
            font=dict(color=self.COLORS['text'], size=12),
            title_font=dict(size=22, color=self.COLORS['text'], family='Arial Black'),
            xaxis=dict(
                showgrid=True,
                gridcolor=self.COLORS['grid'],
                title_font=dict(size=14, color=self.COLORS['text'])
            ),
            yaxis=dict(
                categoryorder='total ascending',
                showgrid=False,
                title_font=dict(size=14, color=self.COLORS['text'])
            ),
            showlegend=False
        )

        return fig

    def generate_subcategory_chart(self, df: pd.DataFrame, filter_outros: bool = False) -> Any:
        """Gera gráfico de barras de subcategorias

        Args:
            df: DataFrame com os dados
            filter_outros: Se True, remove "Outros" do gráfico (após classificação)
        """
        print(f"\n[DEBUG SUBCATEGORY] filter_outros={filter_outros}")
        print(f"[DEBUG SUBCATEGORY] Total registros ANTES filtro: {len(df)}")

        if 'SUB_ASSUNTO' in df.columns:
            outros_antes = df['SUB_ASSUNTO'].str.upper().str.contains('OUTRO', na=False).sum()
            print(f"[DEBUG SUBCATEGORY] Registros com 'OUTRO' ANTES filtro: {outros_antes}")

            variacoes_outro = df[df['SUB_ASSUNTO'].str.upper().str.contains('OUTRO', na=False)]['SUB_ASSUNTO'].unique()
            print(f"[DEBUG SUBCATEGORY] Variacoes de 'OUTRO': {variacoes_outro[:5]}")

        if filter_outros:
            df = df[
                (~df['SUB_ASSUNTO'].str.upper().str.contains('OUTRO', na=False)) &
                (df['SUB_ASSUNTO'].notna()) &
                (df['SUB_ASSUNTO'].str.strip() != '')
            ].copy()
            print(f"[DEBUG SUBCATEGORY] Total registros DEPOIS filtro: {len(df)}")

        subcategory_counts = df['SUB_ASSUNTO'].value_counts().reset_index()
        subcategory_counts.columns = ['Subcategoria', 'Quantidade']

        print(f"[DEBUG SUBCATEGORY] Top 5 no grafico:")
        print(subcategory_counts.head())

        fig = px.bar(
            subcategory_counts,
            x='Quantidade',
            y='Subcategoria',
            orientation='h',
            title='📑 Subcategorias de Reclamações',
            text='Quantidade',
            color='Quantidade',
            color_continuous_scale=[
                [0, self.COLORS['primary']],
                [0.5, self.COLORS['purple']],
                [1, self.COLORS['secondary']]
            ]
        )

        fig.update_traces(
            textposition='outside',
            textfont=dict(size=11, color=self.COLORS['text'])
        )
        fig.update_layout(
            height=max(400, len(subcategory_counts) * 30),
            plot_bgcolor=self.COLORS['background'],
            paper_bgcolor=self.COLORS['background'],
            font=dict(color=self.COLORS['text'], size=12),
            title_font=dict(size=22, color=self.COLORS['text'], family='Arial Black'),
            xaxis=dict(
                showgrid=True,
                gridcolor=self.COLORS['grid'],
                title_font=dict(size=14, color=self.COLORS['text'])
            ),
            yaxis=dict(
                categoryorder='total ascending',
                showgrid=False,
                title_font=dict(size=14, color=self.COLORS['text'])
            ),
            showlegend=False
        )

        return fig

    def generate_branch_ranking(self, df: pd.DataFrame) -> Any:
        """Gera ranking de filiais"""
        branch_counts = df['DS_FILIAL'].value_counts().reset_index()
        branch_counts.columns = ['Filial', 'Quantidade']

        fig = px.bar(
            branch_counts,
            x='Quantidade',
            y='Filial',
            orientation='h',
            title='🏆 Ranking de Filiais por Reclamações',
            text='Quantidade',
            color='Quantidade',
            color_continuous_scale=[
                [0, self.COLORS['accent']],
                [0.5, self.COLORS['warning']],
                [1, self.COLORS['danger']]
            ]
        )

        fig.update_traces(
            textposition='outside',
            textfont=dict(size=11, color=self.COLORS['text'])
        )
        fig.update_layout(
            height=max(400, len(branch_counts) * 30),
            plot_bgcolor=self.COLORS['background'],
            paper_bgcolor=self.COLORS['background'],
            font=dict(color=self.COLORS['text'], size=12),
            title_font=dict(size=22, color=self.COLORS['text'], family='Arial Black'),
            xaxis=dict(
                showgrid=True,
                gridcolor=self.COLORS['grid'],
                title_font=dict(size=14, color=self.COLORS['text'])
            ),
            yaxis=dict(
                categoryorder='total ascending',
                showgrid=False,
                title_font=dict(size=14, color=self.COLORS['text'])
            ),
            showlegend=False
        )

        return fig

    def generate_operator_ranking(self, df: pd.DataFrame) -> Any:
        """Gera ranking de operadoras"""
        if 'OPERADORA' not in df.columns:
            return None

        operator_counts = df['OPERADORA'].value_counts().reset_index()
        operator_counts.columns = ['Operadora', 'Quantidade']

        fig = px.bar(
            operator_counts,
            x='Quantidade',
            y='Operadora',
            orientation='h',
            title='📱 Ranking de Operadoras por Reclamações',
            text='Quantidade',
            color='Quantidade',
            color_continuous_scale=[
                [0, self.COLORS['primary']],
                [0.5, self.COLORS['purple']],
                [1, self.COLORS['secondary']]
            ]
        )

        fig.update_traces(
            textposition='outside',
            textfont=dict(size=11, color=self.COLORS['text'])
        )
        fig.update_layout(
            height=max(400, len(operator_counts) * 30),
            plot_bgcolor=self.COLORS['background'],
            paper_bgcolor=self.COLORS['background'],
            font=dict(color=self.COLORS['text'], size=12),
            title_font=dict(size=22, color=self.COLORS['text'], family='Arial Black'),
            xaxis=dict(
                showgrid=True,
                gridcolor=self.COLORS['grid'],
                title_font=dict(size=14, color=self.COLORS['text'])
            ),
            yaxis=dict(
                categoryorder='total ascending',
                showgrid=False,
                title_font=dict(size=14, color=self.COLORS['text'])
            ),
            showlegend=False
        )

        return fig
