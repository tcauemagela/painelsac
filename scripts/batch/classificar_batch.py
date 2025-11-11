import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("INICIANDO CLASSIFICAÇÃO DE SUB_ASSUNTOS")
print("="*80)

print("\n[1/7] Carregando dados...")
df = pd.read_csv(r'C:\Users\Usuário\Desktop\projetos_ia\monitoramento_nip\SACS_FEITOS_01_07_AO_03_11.CSV',
                 encoding='latin-1', sep=';', on_bad_lines='skip', low_memory=False)
print(f"   OK - {len(df)} registros carregados")

def tem_outros(valor):
    if pd.isna(valor):
        return False
    valor_str = str(valor).lower()
    return bool(re.search(r'outro', valor_str))

df['tem_outros'] = df['SUB_ASSUNTO'].apply(tem_outros)
df_treino = df[~df['tem_outros']].copy()
df_classificar = df[df['tem_outros']].copy()

print(f"\n[2/7] Separando dados...")
print(f"   OK - Dados de treinamento (sem 'outros'): {len(df_treino)}")
print(f"   OK - Dados para classificar (com 'outros'): {len(df_classificar)}")

print("\n[3/7] Preparando features...")
def preparar_texto(row):
    """Combina campos relevantes para criar o texto de entrada"""
    textos = []

    campos = ['DS_ASSUNTO', 'DS_OBSERVACAO', 'DS_TRATATIVA', 'DS_RETORNO', 'DS_MOTIVO']

    for campo in campos:
        if campo in row.index and pd.notna(row[campo]):
            textos.append(str(row[campo]))

    return ' '.join(textos)

df_treino['texto_features'] = df_treino.apply(preparar_texto, axis=1)
df_classificar['texto_features'] = df_classificar.apply(preparar_texto, axis=1)

print(f"   OK - Features criadas")

print("\n[4/7] Vetorizando textos...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    stop_words=None  # Mantém todas as palavras
)

X_treino = vectorizer.fit_transform(df_treino['texto_features'])
X_classificar = vectorizer.transform(df_classificar['texto_features'])
print(f"   OK - Vetorizacao concluida ({X_treino.shape[1]} features)")

print("\n[5/7] Treinando modelo...")
le = LabelEncoder()
y_treino = le.fit_transform(df_treino['SUB_ASSUNTO'])

clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=50,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
    verbose=0
)

clf.fit(X_treino, y_treino)
print(f"   OK - Modelo treinado")

print("\n[6/7] Classificando registros...")
print(f"   Classificando {len(df_classificar)} registros...")

y_pred = clf.predict(X_classificar)
y_pred_labels = le.inverse_transform(y_pred)

y_pred_proba = clf.predict_proba(X_classificar)
max_proba = y_pred_proba.max(axis=1)

df_classificar['SUB_ASSUNTO_NOVO'] = y_pred_labels
df_classificar['CONFIANCA'] = max_proba

print(f"   OK - Classificacao concluida!")
print(f"   Confianca media: {max_proba.mean():.2%}")
print(f"   Confianca minima: {max_proba.min():.2%}")
print(f"   Confianca maxima: {max_proba.max():.2%}")

print("\n[7/7] Criando planilhas finais...")

df_total = df.copy()
df_total.loc[df_total['tem_outros'], 'SUB_ASSUNTO'] = df_classificar['SUB_ASSUNTO_NOVO'].values
df_total = df_total.drop(columns=['tem_outros'])

df_parcial = df.copy()
indices_classificar_parcial = df_classificar.nlargest(8500, 'CONFIANCA').index
df_parcial.loc[indices_classificar_parcial, 'SUB_ASSUNTO'] = df_classificar.loc[indices_classificar_parcial, 'SUB_ASSUNTO_NOVO'].values
df_parcial = df_parcial.drop(columns=['tem_outros'])

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

import os
pasta_total = r'C:\Users\Usuário\Desktop\projetos_ia\monitoramento_nip\planilhas_total_classificada'
pasta_parcial = r'C:\Users\Usuário\Desktop\projetos_ia\monitoramento_nip\planilhas_parcialmente_classificada'

os.makedirs(pasta_total, exist_ok=True)
os.makedirs(pasta_parcial, exist_ok=True)

arquivo_total = os.path.join(pasta_total, f'SACS_TOTAL_CLASSIFICADO_{timestamp}.csv')
arquivo_parcial = os.path.join(pasta_parcial, f'SACS_PARCIAL_8500_{timestamp}.csv')

df_total.to_csv(arquivo_total, index=False, sep=';', encoding='latin-1')
df_parcial.to_csv(arquivo_parcial, index=False, sep=';', encoding='latin-1')

print(f"   OK - Planilha total salva: {arquivo_total}")
print(f"   OK - Planilha parcial salva: {arquivo_parcial}")

print("\n" + "="*80)
print("CLASSIFICAÇÃO CONCLUÍDA COM SUCESSO!")
print("="*80)
print(f"\nDistribuição das classificações (total):")
print(df_total['SUB_ASSUNTO'].value_counts().head(10))

print("\n" + "="*80)
