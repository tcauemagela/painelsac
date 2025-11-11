"""
Script de treinamento RÁPIDO para classificação automática de SUB_ASSUNTO.
Usa embeddings pré-computados e K-NN para classificação.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sentence_transformers import SentenceTransformer
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.services.TextBuilderService import TextBuilderService


def main():
    print("="*80)
    print("TREINAMENTO RÁPIDO - CLASSIFICADOR DE SUB_ASSUNTO")
    print("="*80)

    print("\n[1/4] Carregando dados de treinamento...")

    training_file = Path(__file__).parent.parent / "SACS_FEITOS_01_07_AO_03_11.CSV"
    if not training_file.exists():
        print(f"Erro: Arquivo de treinamento nao encontrado: {training_file}")
        print("   Por favor, coloque o arquivo CSV na pasta raiz do projeto.")
        return

    df = pd.read_csv(
        training_file,
        sep=';',
        encoding='latin1',
        low_memory=False
    )

    print(f"   OK {len(df):,} registros carregados")

    print("\n[2/4] Filtrando registros válidos para treinamento...")

    text_builder = TextBuilderService()

    mask_valid = (
        df['SUB_ASSUNTO'].notna() &
        (df['SUB_ASSUNTO'].str.strip() != '') &
        ~df['SUB_ASSUNTO'].str.upper().str.contains('OUTRO', na=False)  # Remove QUALQUER "OUTRO"
    )

    df_valid = df[mask_valid].copy()
    print(f"   OK {len(df_valid):,} registros com SUB_ASSUNTO valido")

    MAX_TRAINING_SAMPLES = 5000
    if len(df_valid) > MAX_TRAINING_SAMPLES:
        print(f"\n   OTIMIZACAO: Limitando a {MAX_TRAINING_SAMPLES:,} registros (amostragem estratificada)")
        df_valid = df_valid.groupby('SUB_ASSUNTO', group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, int(MAX_TRAINING_SAMPLES * len(x) / len(df_valid)))))
        ).reset_index(drop=True)
        print(f"   OK {len(df_valid):,} registros selecionados")

    print("\n   Top 10 SUB_ASSUNTO:")
    subassunto_counts = df_valid['SUB_ASSUNTO'].value_counts()
    for subassunto, count in subassunto_counts.head(10).items():
        print(f"      - {subassunto}: {count:,}")

    print("\n[3/4] Gerando embeddings (isso pode demorar alguns minutos)...")

    print("   > Construindo textos de referência...")
    reference_texts = []
    for idx, row in df_valid.iterrows():
        text = text_builder.build_text_from_row(row)
        reference_texts.append(text)

    print(f"   OK {len(reference_texts):,} textos construidos")

    print("   > Carregando modelo de embeddings...")
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    print("   OK Modelo carregado")

    print("   > Gerando embeddings em mini-batches...")
    BATCH_SIZE = 100
    all_embeddings = []

    for i in range(0, len(reference_texts), BATCH_SIZE):
        batch = reference_texts[i:i+BATCH_SIZE]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.extend(batch_embeddings)

        if (i // BATCH_SIZE + 1) % 10 == 0:
            print(f"      Processados {i + len(batch):,}/{len(reference_texts):,} textos...")

    embeddings_array = np.array(all_embeddings)
    print(f"   OK Embeddings gerados: shape={embeddings_array.shape}")

    print("\n[4/4] Salvando modelo treinado...")

    output_dir = Path(__file__).parent.parent / "data" / "ml"
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_df = df_valid[['SUB_ASSUNTO']].copy()
    reference_df['texto_referencia'] = reference_texts

    reference_path = output_dir / "subassunto_reference.pkl"
    with open(reference_path, 'wb') as f:
        pickle.dump(reference_df, f)
    print(f"   OK Referencias salvas: {reference_path}")

    embeddings_path = output_dir / "subassunto_embeddings.npy"
    np.save(embeddings_path, embeddings_array)
    print(f"   OK Embeddings salvos: {embeddings_path}")

    print("\n" + "="*80)
    print("OK TREINAMENTO CONCLUIDO COM SUCESSO!")
    print("="*80)
    print(f"\nEstatisticas do Modelo:")
    print(f"   - Total de referencias: {len(reference_df):,}")
    print(f"   - Dimensao dos embeddings: {embeddings_array.shape[1]}")
    print(f"   - Total de subcategorias unicas: {reference_df['SUB_ASSUNTO'].nunique()}")
    print(f"   - Tamanho do modelo: {embeddings_array.nbytes / (1024*1024):.2f} MB")

    print(f"\nArquivos gerados:")
    print(f"   - {reference_path}")
    print(f"   - {embeddings_path}")

    print(f"\nProximos passos:")
    print(f"   1. Execute a aplicacao: streamlit run app.py")
    print(f"   2. Faca upload de um arquivo")
    print(f"   3. Clique em 'Classificar SUB_ASSUNTO' na sidebar")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
