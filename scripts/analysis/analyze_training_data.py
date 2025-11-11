"""
Script para análise da base de treinamento SACS_FEITOS
"""
import pandas as pd

print("Carregando dados de treinamento...")
df = pd.read_csv(
    r"C:\Users\Usuário\Desktop\projetos_ia\monitoramento_nip\SACS_FEITOS_01_07_AO_03_11.CSV",
    sep=';',
    encoding='latin1',
    low_memory=False
)

print(f"\nTotal de registros: {len(df):,}")

print(f"\nColunas disponiveis:")
print(df.columns.tolist())

print("\n" + "="*80)
print("ANALISE DE DS_ASSUNTO")
print("="*80)

assuntos_unicos = df['DS_ASSUNTO'].dropna().unique()
print(f"\nTotal de assuntos unicos: {len(assuntos_unicos)}")

assunto_counts = df['DS_ASSUNTO'].value_counts()
print(f"\nTop 20 DS_ASSUNTO mais frequentes:")
print(assunto_counts.head(20))

total_vazios = df['DS_ASSUNTO'].isna().sum()
total_outros = df['DS_ASSUNTO'].str.upper().isin(['OUTROS', 'OUTRO']).sum()
total_validos = len(df) - total_vazios - total_outros

print(f"\nEstatisticas:")
print(f"   - Registros com DS_ASSUNTO valido: {total_validos:,} ({total_validos/len(df)*100:.2f}%)")
print(f"   - Registros vazios: {total_vazios:,} ({total_vazios/len(df)*100:.2f}%)")
print(f"   - Registros 'Outros': {total_outros:,} ({total_outros/len(df)*100:.2f}%)")

print("\n" + "="*80)
print("COLUNAS DE TEXTO DISPONIVEIS PARA EMBEDDING")
print("="*80)

text_cols = ['DS_OBSERVACAO', 'DS_TRATATIVA', 'DS_RETORNO', 'DS_MOTIVO']
for col in text_cols:
    if col in df.columns:
        non_null = df[col].notna().sum()
        print(f"   - {col}: {non_null:,} registros ({non_null/len(df)*100:.2f}%)")

print("\n" + "="*80)
print("EXEMPLOS DE REGISTROS VALIDOS")
print("="*80)

sample_validos = df[df['DS_ASSUNTO'].notna()].head(3)
for idx, row in sample_validos.iterrows():
    print(f"\nRegistro {idx}:")
    print(f"   DS_ASSUNTO: {row['DS_ASSUNTO']}")
    if 'DS_OBSERVACAO' in df.columns and pd.notna(row['DS_OBSERVACAO']):
        print(f"   DS_OBSERVACAO: {row['DS_OBSERVACAO'][:200]}...")

print("\nAnalise concluida!")
