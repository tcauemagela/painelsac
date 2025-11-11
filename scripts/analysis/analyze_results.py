import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'src'))
from src.services.FuzzyColumnMapper import FuzzyColumnMapper

print("Analisando arquivo classificado...")
print("="*80)

df = pd.read_csv(
    'planilhas_total_classificada/SACS_TOTAL_CLASSIFICADO_20251106_110245.csv',
    sep=';',
    encoding='latin-1',
    engine='python',
    on_bad_lines='warn'
)

print(f"\nTotal de registros carregados: {len(df):,}")
print(f"\nPrimeiras 10 colunas: {list(df.columns)[:10]}")

mapper = FuzzyColumnMapper()
df = mapper.map_columns(df)

print(f"\nApos mapeamento - colunas mapeadas: {list(df.columns)[:10]}")

if 'SUB_ASSUNTO' in df.columns:
    print(f"\n{'='*80}")
    print("ANALISE DE SUB_ASSUNTO")
    print(f"{'='*80}")

    mask_outros = df['SUB_ASSUNTO'].str.upper().str.contains('OUTRO', na=False)
    total_outros = mask_outros.sum()

    print(f"\nTotal de registros com 'OUTROS': {total_outros:,}")

    print(f"\nDistribuicao detalhada dos 'OUTROS':")
    outros_dist = df[mask_outros]['SUB_ASSUNTO'].value_counts()
    for idx, (valor, count) in enumerate(outros_dist.items(), 1):
        print(f"  {idx}. '{valor}': {count:,}")
        if idx >= 10:
            break

    print(f"\nTotal de SUB_ASSUNTO unicos: {df['SUB_ASSUNTO'].nunique():,}")

    print(f"\nTop 10 SUB_ASSUNTO (incluindo OUTROS):")
    for idx, (valor, count) in enumerate(df['SUB_ASSUNTO'].value_counts().head(10).items(), 1):
        print(f"  {idx}. '{valor}': {count:,}")

    vazios = df['SUB_ASSUNTO'].isna().sum()
    print(f"\nRegistros vazios: {vazios:,}")
    print(f"Registros preenchidos: {len(df) - vazios:,}")

else:
    print("\n[ERRO] Coluna SUB_ASSUNTO nao encontrada!")
    print(f"Colunas disponiveis: {list(df.columns)}")
