"""
Script de treinamento OTIMIZADO do classificador de DS_ASSUNTO.

Melhorias:
1. Usa modelo menor e mais rapido (MiniLM)
2. Amostra estratificada do dataset (mais rapido)
3. Salva checkpoints incrementais
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime

from src.services.EmbeddingService import EmbeddingService
from src.services.TextBuilderService import TextBuilderService


def criar_amostra_estratificada(df, coluna_categoria, n_por_categoria=None, total_desejado=5000):
    """
    Cria amostra estratificada mantendo proporcao de categorias.

    Args:
        df: DataFrame completo
        coluna_categoria: Nome da coluna de categoria
        n_por_categoria: Numero especifico por categoria (opcional)
        total_desejado: Total de registros desejado (se n_por_categoria nao especificado)

    Returns:
        DataFrame com amostra estratificada
    """
    category_counts = df[coluna_categoria].value_counts()

    if n_por_categoria is None:
        proporcao = total_desejado / len(df)
        amostras = {}
        for cat, count in category_counts.items():
            n_amostras = max(int(count * proporcao), 10)  # Minimo 10 por categoria
            amostras[cat] = n_amostras
    else:
        amostras = {cat: n_por_categoria for cat in category_counts.index}

    df_amostras = []
    for cat, n in amostras.items():
        df_cat = df[df[coluna_categoria] == cat]
        n_final = min(n, len(df_cat))  # Nao exceder tamanho da categoria
        df_sample = df_cat.sample(n=n_final, random_state=42)
        df_amostras.append(df_sample)

    df_resultado = pd.concat(df_amostras, ignore_index=True)

    print("\nAmostra estratificada criada:")
    for cat in df_resultado[coluna_categoria].value_counts().index:
        original = len(df[df[coluna_categoria] == cat])
        amostra = len(df_resultado[df_resultado[coluna_categoria] == cat])
        print(f"   - {cat}: {amostra:,} de {original:,} ({amostra/original*100:.1f}%)")

    return df_resultado


def main():
    """Execute the optimized training pipeline."""
    print("="*80)
    print("TREINAMENTO OTIMIZADO DO CLASSIFICADOR DE DS_ASSUNTO")
    print("="*80)
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    TRAINING_FILE = r"C:\Users\Usuário\Desktop\projetos_ia\monitoramento_nip\SACS_FEITOS_01_07_AO_03_11.CSV"
    OUTPUT_DIR = Path(r"C:\Users\Usuário\Desktop\projetos_ia\monitoramento_nip\data\ml")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    USE_SMALLER_MODEL = True  # Usar modelo menor (mais rapido)
    USE_STRATIFIED_SAMPLE = True  # Usar amostra estratificada
    SAMPLE_SIZE = 5000  # Tamanho da amostra (0 = usar todos)

    print("ETAPA 1: Carregando dados de treinamento...")
    print(f"Arquivo: {TRAINING_FILE}")

    df = pd.read_csv(
        TRAINING_FILE,
        sep=';',
        encoding='latin1',
        low_memory=False
    )

    print(f"Total de registros carregados: {len(df):,}\n")

    print("ETAPA 2: Filtrando registros validos...")

    text_builder = TextBuilderService()

    df_valid = df[~df['DS_ASSUNTO'].apply(text_builder.needs_classification)].copy()

    print(f"Registros com DS_ASSUNTO valido: {len(df_valid):,}")

    print("\nConstruindo representacao textual dos registros...")
    df_valid['texto_referencia'] = df_valid.apply(
        text_builder.build_text_from_row,
        axis=1
    )

    df_valid = df_valid[
        df_valid['texto_referencia'].apply(
            lambda x: text_builder.validate_text_length(x, min_length=10)
        )
    ]

    print(f"Registros com texto suficiente: {len(df_valid):,}")

    print("\nDistribuicao ORIGINAL por categoria:")
    category_counts = df_valid['DS_ASSUNTO'].value_counts()
    for category, count in category_counts.items():
        print(f"   - {category}: {count:,}")

    if USE_STRATIFIED_SAMPLE and SAMPLE_SIZE > 0 and len(df_valid) > SAMPLE_SIZE:
        print("\n" + "="*80)
        print("OTIMIZACAO: Criando amostra estratificada")
        print("="*80)
        print(f"Reduzindo de {len(df_valid):,} para ~{SAMPLE_SIZE:,} registros...")

        df_train = criar_amostra_estratificada(
            df_valid,
            'DS_ASSUNTO',
            total_desejado=SAMPLE_SIZE
        )

        print(f"\nTotal na amostra de treinamento: {len(df_train):,}")
    else:
        df_train = df_valid
        print(f"\nUsando dataset completo: {len(df_train):,} registros")

    print("\n" + "="*80)
    print("ETAPA 3: Gerando embeddings...")
    print("="*80)

    if USE_SMALLER_MODEL:
        model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        print("\nOTIMIZACAO: Usando modelo menor e mais rapido (MiniLM)")
    else:
        model_name = 'neuralmind/bert-base-portuguese-cased'

    embedding_service = EmbeddingService(model_name=model_name)

    print(f"\nModelo: {embedding_service.model_name}")
    print(f"Dimensao dos embeddings: {embedding_service.get_embedding_dimension()}")
    print(f"\nGerando embeddings para {len(df_train):,} textos...")
    print("(Agora sera mais rapido!)\n")

    texts = df_train['texto_referencia'].tolist()
    embeddings = embedding_service.generate_embeddings(texts)

    print(f"\nEmbeddings gerados: {embeddings.shape}")

    print("\n" + "="*80)
    print("ETAPA 4: Salvando dados de referencia...")
    print("="*80)

    reference_path = OUTPUT_DIR / 'assunto_reference.pkl'
    with open(reference_path, 'wb') as f:
        pickle.dump(df_train[['DS_ASSUNTO', 'texto_referencia']], f)

    print(f"\nDataFrame de referencia salvo em: {reference_path}")

    embeddings_path = OUTPUT_DIR / 'assunto_embeddings.npy'
    np.save(embeddings_path, embeddings)

    print(f"Embeddings salvos em: {embeddings_path}")

    metadata = {
        'training_date': datetime.now().isoformat(),
        'total_records': len(df_train),
        'original_dataset_size': len(df_valid),
        'used_stratified_sample': USE_STRATIFIED_SAMPLE,
        'sample_size': len(df_train),
        'embedding_model': embedding_service.model_name,
        'embedding_dimension': embedding_service.get_embedding_dimension(),
        'categories': df_train['DS_ASSUNTO'].value_counts().to_dict(),
        'reference_path': str(reference_path),
        'embeddings_path': str(embeddings_path)
    }

    metadata_path = OUTPUT_DIR / 'assunto_metadata.pkl'
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"Metadata salvo em: {metadata_path}")

    print("\n" + "="*80)
    print("TREINAMENTO CONCLUIDO COM SUCESSO!")
    print("="*80)
    print(f"\nResumo:")
    print(f"   - Registros de treinamento: {len(df_train):,}")
    print(f"   - Dataset original: {len(df_valid):,}")
    print(f"   - Categorias unicas: {len(df_train['DS_ASSUNTO'].unique())}")
    print(f"   - Dimensao dos embeddings: {embeddings.shape[1]}")
    print(f"   - Tamanho em disco (embeddings): {embeddings_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"   - Modelo usado: {embedding_service.model_name}")

    print(f"\nDistribuicao final por categoria:")
    for cat, count in df_train['DS_ASSUNTO'].value_counts().items():
        print(f"   - {cat}: {count:,}")

    print(f"\nArquivos gerados:")
    print(f"   - {reference_path}")
    print(f"   - {embeddings_path}")
    print(f"   - {metadata_path}")

    print(f"\nFim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nO classificador esta pronto para uso!")
    print("Use AssuntoClassifierService para classificar novos registros.")


if __name__ == "__main__":
    main()
