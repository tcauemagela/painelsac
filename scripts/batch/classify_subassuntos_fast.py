import pandas as pd
import sys
from pathlib import Path
import numpy as np
from datetime import datetime

sys.path.append(str(Path(__file__).parent / 'src'))

from src.services.EmbeddingService import EmbeddingService
from src.services.SubAssuntoClassifierService import SubAssuntoClassifierService
from src.services.FuzzyColumnMapper import FuzzyColumnMapper

def main():
    print("="*80)
    print("CLASSIFICAÇÃO RÁPIDA DE SUB_ASSUNTOS")
    print("="*80)

    print("\n[1/5] Carregando planilha...")
    encodings = ['latin-1', 'iso-8859-1', 'cp1252', 'utf-8-sig', 'utf-8']
    df = None
    for encoding in encodings:
        try:
            df = pd.read_csv(
                "SACS_FEITOS_01_07_AO_03_11.CSV",
                encoding=encoding,
                sep=';',              # Delimitador é ponto-e-vírgula
                on_bad_lines='warn',  # Avisar mas continuar se houver linhas ruins
                engine='python',      # Engine mais tolerante
                quoting=1,            # QUOTE_MINIMAL
                escapechar='\\'
            )
            print(f"-> Arquivo carregado com encoding: {encoding}")
            break
        except (UnicodeDecodeError, Exception) as e:
            if "encoding" not in str(e).lower():
                print(f"[AVISO] Erro com {encoding}: {str(e)[:50]}")
            continue

    if df is None:
        raise Exception("Nao foi possivel detectar o encoding do arquivo")

    print(f"-> Total de registros: {len(df):,}")

    print("\n[2/5] Mapeando colunas...")
    print(f"-> Colunas originais: {list(df.columns)[:10]}...")  # Mostrar primeiras 10 colunas
    mapper = FuzzyColumnMapper()
    df = mapper.map_columns(df)
    print(f"-> Colunas mapeadas: {list(df.columns)[:10]}...")

    if 'SUB_ASSUNTO' not in df.columns:
        print(f"[ERRO] Coluna SUB_ASSUNTO nao encontrada!")
        print(f"-> Colunas disponiveis: {list(df.columns)}")
        return

    mask_needs_class = (
        df['SUB_ASSUNTO'].isna() |
        (df['SUB_ASSUNTO'].str.strip() == '') |
        df['SUB_ASSUNTO'].str.upper().str.contains('OUTRO', na=False)
    )
    num_needs_class = mask_needs_class.sum()
    print(f"-> Registros que precisam classificacao: {num_needs_class:,}")

    print("\n[3/5] Inicializando classificador...")
    embedding_service = EmbeddingService(
        model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    )
    classifier = SubAssuntoClassifierService(
        embedding_service=embedding_service,
        threshold=0.50,  # Threshold maior = menos processamento
        k_neighbors=3    # Menos vizinhos = mais rápido
    )
    print("-> Classificador carregado (modo rapido: k=3, threshold=0.50)")

    print("\n[4/5] Classificando TODOS os registros...")
    df_total = df.copy()

    def progress_callback(progress):
        if int(progress * 100) % 10 == 0:
            print(f"   Progresso: {int(progress * 100)}%")

    df_total_classified = classifier.classify_dataframe(df_total, progress_callback=progress_callback)

    mask_outros_depois_total = df_total_classified['SUB_ASSUNTO'].str.upper().str.contains('OUTRO', na=False)
    outros_depois_total = mask_outros_depois_total.sum()
    classificados_total = num_needs_class - outros_depois_total

    print(f"-> Classificacao total concluida")
    print(f"   - Antes: {num_needs_class:,} 'outros'")
    print(f"   - Depois: {outros_depois_total:,} 'outros'")
    print(f"   - Classificados: {classificados_total:,}")

    print("\n[5/5] Criando versão parcial (8500 classificações)...")
    df_partial = df.copy()

    indices_to_classify = df_partial[mask_needs_class].index.tolist()

    indices_limited = indices_to_classify[:8500]

    mask_classify_limited = pd.Series(False, index=df_partial.index)
    mask_classify_limited.loc[indices_limited] = True

    temp_df = df_partial[mask_classify_limited].copy()
    temp_df_classified = classifier.classify_dataframe(temp_df, progress_callback=None)

    df_partial.loc[mask_classify_limited, 'SUB_ASSUNTO'] = temp_df_classified['SUB_ASSUNTO'].values

    mask_outros_depois_partial = df_partial['SUB_ASSUNTO'].str.upper().str.contains('OUTRO', na=False)
    outros_depois_partial = mask_outros_depois_partial.sum()
    classificados_partial = 8500 - (df_partial.loc[indices_limited, 'SUB_ASSUNTO'].str.upper().str.contains('OUTRO', na=False).sum())

    print(f"-> Classificacao parcial concluida")
    print(f"   - Classificados: ~{classificados_partial:,} de 8500 tentativas")
    print(f"   - Restantes 'outros': {num_needs_class - 8500:,}")

    print("\n[6/6] Salvando resultados...")
    Path("planilhas_total_classificada").mkdir(exist_ok=True)
    Path("planilhas_parcialmente_classificada").mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_total = f"planilhas_total_classificada/SACS_TOTAL_CLASSIFICADO_{timestamp}.csv"
    df_total_classified.to_csv(path_total, index=False, encoding='utf-8-sig', sep=';')
    print(f"-> Salvo: {path_total}")

    path_partial = f"planilhas_parcialmente_classificada/SACS_PARCIAL_8500_{timestamp}.csv"
    df_partial.to_csv(path_partial, index=False, encoding='utf-8-sig', sep=';')
    print(f"-> Salvo: {path_partial}")

    print("\n" + "="*80)
    print("CONCLUIDO COM SUCESSO!")
    print("="*80)
    print(f"\nRESUMO:")
    print(f"   Versao Total: {classificados_total:,} sub_assuntos classificados")
    print(f"   Versao Parcial: ~{classificados_partial:,} sub_assuntos classificados")
    print(f"\nArquivos gerados:")
    print(f"   - {path_total}")
    print(f"   - {path_partial}")

if __name__ == "__main__":
    main()
