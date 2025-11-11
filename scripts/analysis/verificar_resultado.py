import pandas as pd
import re

df_total = pd.read_csv(r'C:\Users\Usuário\Desktop\projetos_ia\monitoramento_nip\planilhas_total_classificada\SACS_TOTAL_CLASSIFICADO_20251106_141528.csv',
                       encoding='latin-1', sep=';', on_bad_lines='skip', low_memory=False)

df_parcial = pd.read_csv(r'C:\Users\Usuário\Desktop\projetos_ia\monitoramento_nip\planilhas_parcialmente_classificada\SACS_PARCIAL_8500_20251106_141528.csv',
                         encoding='latin-1', sep=';', on_bad_lines='skip', low_memory=False)

def tem_outros(valor):
    if pd.isna(valor):
        return False
    return bool(re.search(r'outro', str(valor).lower()))

print("="*80)
print("VERIFICACAO DAS PLANILHAS CRIADAS")
print("="*80)

print("\n=== PLANILHA TOTAL CLASSIFICADA ===")
print(f"Total de linhas: {len(df_total)}")
total_outros = df_total['SUB_ASSUNTO'].apply(tem_outros).sum()
print(f"Registros ainda com 'outros': {total_outros}")
print(f"Registros classificados corretamente: {len(df_total) - total_outros}")

print("\n=== PLANILHA PARCIALMENTE CLASSIFICADA ===")
print(f"Total de linhas: {len(df_parcial)}")
parcial_outros = df_parcial['SUB_ASSUNTO'].apply(tem_outros).sum()
print(f"Registros ainda com 'outros': {parcial_outros}")
print(f"Registros classificados corretamente: {len(df_parcial) - parcial_outros}")

print("\n=== DISTRIBUICAO DE CATEGORIAS (PLANILHA TOTAL) ===")
print(df_total['SUB_ASSUNTO'].value_counts().head(15))

print("\n" + "="*80)
