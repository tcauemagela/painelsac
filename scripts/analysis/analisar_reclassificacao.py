import pandas as pd
import re

print("Carregando planilhas...")
df_original = pd.read_csv(r'C:\Users\Usuário\Desktop\projetos_ia\monitoramento_nip\SACS_FEITOS_01_07_AO_03_11.CSV',
                          encoding='latin-1', sep=';', on_bad_lines='skip', low_memory=False)

df_classificado = pd.read_csv(r'C:\Users\Usuário\Desktop\projetos_ia\monitoramento_nip\planilhas_total_classificada\SACS_TOTAL_CLASSIFICADO_20251106_141528.csv',
                              encoding='latin-1', sep=';', on_bad_lines='skip', low_memory=False)

def tem_outros(valor):
    if pd.isna(valor):
        return False
    return bool(re.search(r'outro', str(valor).lower()))

df_original['era_outros'] = df_original['SUB_ASSUNTO'].apply(tem_outros)

indices_outros = df_original[df_original['era_outros']].index
novas_categorias = df_classificado.loc[indices_outros, 'SUB_ASSUNTO']

print("\n" + "="*80)
print("RECLASSIFICACAO DOS REGISTROS QUE ERAM 'OUTROS'")
print("="*80)
print(f"\nTotal de registros que eram 'outros': {len(novas_categorias)}")
print(f"\nDistribuicao das novas categorias:\n")

distribuicao = novas_categorias.value_counts()
for categoria, quantidade in distribuicao.items():
    percentual = (quantidade / len(novas_categorias)) * 100
    print(f"{categoria:<65} {quantidade:>5} ({percentual:>5.2f}%)")

print("\n" + "="*80)
print(f"Total: {distribuicao.sum()} registros reclassificados")
print(f"Categorias unicas utilizadas: {len(distribuicao)}")
print("="*80)
