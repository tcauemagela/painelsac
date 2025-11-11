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

antes = df_original[~df_original['era_outros']]['SUB_ASSUNTO'].value_counts()

depois = df_classificado['SUB_ASSUNTO'].value_counts()

todas_categorias = sorted(set(antes.index) | set(depois.index))

comparacao = []
for cat in todas_categorias:
    antes_val = antes.get(cat, 0)
    depois_val = depois.get(cat, 0)
    diferenca = depois_val - antes_val
    crescimento = ((diferenca / antes_val) * 100) if antes_val > 0 else float('inf')

    comparacao.append({
        'Categoria': cat,
        'Antes': antes_val,
        'Depois': depois_val,
        'Diferenca': diferenca,
        'Crescimento_%': crescimento
    })

df_comp = pd.DataFrame(comparacao)

df_comp = df_comp.sort_values('Depois', ascending=False)

print("\n" + "="*120)
print("COMPARACAO ANTES vs DEPOIS DA RECLASSIFICACAO")
print("="*120)
print(f"\n{'SUBCATEGORIA':<65} {'ANTES':>8} {'DEPOIS':>8} {'DIFERENCA':>10} {'CRESC %':>10}")
print("-"*120)

for _, row in df_comp.iterrows():
    cat = row['Categoria'][:63]  # Limitar tamanho
    antes_val = int(row['Antes'])
    depois_val = int(row['Depois'])
    dif = int(row['Diferenca'])

    if row['Crescimento_%'] == float('inf'):
        cresc = "NOVO"
    else:
        cresc = f"+{row['Crescimento_%']:.1f}%"

    sinal = "+" if dif > 0 else ""
    print(f"{cat:<65} {antes_val:>8} {depois_val:>8} {sinal}{dif:>9} {cresc:>10}")

print("-"*120)
print(f"{'TOTAL (sem outros)':<65} {int(antes.sum()):>8} {int(depois.sum()):>8} {'+'+str(int(depois.sum() - antes.sum())):>9} {'+'+str(((depois.sum() - antes.sum())/antes.sum()*100))[:5]+'%':>10}")
print("="*120)

print("\nESTATISTICAS:")
print(f"  - Registros com 'outros' originalmente: {df_original['era_outros'].sum()}")
print(f"  - Registros reclassificados: {int(depois.sum() - antes.sum())}")
print(f"  - Categorias utilizadas para reclassificacao: {len(df_comp[df_comp['Diferenca'] > 0])}")
print(f"  - Maior crescimento absoluto: {df_comp.iloc[0]['Categoria'][:50]} (+{int(df_comp.iloc[0]['Diferenca'])})")

print("\n" + "="*120)
