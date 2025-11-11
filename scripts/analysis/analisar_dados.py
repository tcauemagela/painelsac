import pandas as pd
import re

df = pd.read_csv(r'C:\Users\Usuário\Desktop\projetos_ia\monitoramento_nip\SACS_FEITOS_01_07_AO_03_11.CSV',
                 encoding='latin-1', sep=';', on_bad_lines='skip', low_memory=False)

print(f"Shape: {df.shape}")
print(f"\nColunas: {list(df.columns)}")

print(f"\nValores únicos em SUB_ASSUNTO: {df['SUB_ASSUNTO'].nunique()}")
print(f"\nDistribuição de SUB_ASSUNTO (top 20):")
print(df['SUB_ASSUNTO'].value_counts().head(20))

def tem_outros(valor):
    if pd.isna(valor):
        return False
    valor_str = str(valor).lower()
    return bool(re.search(r'outro', valor_str))

df['tem_outros'] = df['SUB_ASSUNTO'].apply(tem_outros)
total_outros = df['tem_outros'].sum()
total_nao_outros = (~df['tem_outros']).sum()

print(f"\n\nREGISTROS COM 'OUTROS': {total_outros}")
print(f"REGISTROS SEM 'OUTROS': {total_nao_outros}")

print(f"\n\nExemplos de valores com 'outros':")
print(df[df['tem_outros']]['SUB_ASSUNTO'].value_counts().head(10))

print(f"\n\nCategorias válidas (sem 'outros'):")
categorias_validas = df[~df['tem_outros']]['SUB_ASSUNTO'].value_counts()
print(categorias_validas)
print(f"\nTotal de categorias válidas: {len(categorias_validas)}")
