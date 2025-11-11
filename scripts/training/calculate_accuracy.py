"""
Calcula a taxa de acerto REAL do classificador usando validação cruzada.

Método:
1. Carrega dados de treinamento
2. Separa 20% para teste (com DS_ASSUNTO conhecido)
3. Simula que DS_ASSUNTO está vazio
4. Classifica usando os 80% restantes
5. Compara previsão vs realidade
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path

from src.services.EmbeddingService import EmbeddingService
from src.services.TextBuilderService import TextBuilderService


def normalizar_categoria(cat):
    """Normaliza categoria para comparação."""
    if pd.isna(cat):
        return None
    return cat.upper().replace('É', 'E').replace('Í', 'I').replace('Ó', 'O').strip()


def main():
    print("="*80)
    print("CALCULO DE ACURACIA REAL - VALIDACAO CRUZADA")
    print("="*80)

    print("\nCarregando dados de treinamento...")
    TRAINING_FILE = r"C:\Users\Usuário\Desktop\projetos_ia\monitoramento_nip\SACS_FEITOS_01_07_AO_03_11.CSV"

    df = pd.read_csv(
        TRAINING_FILE,
        sep=';',
        encoding='latin1',
        low_memory=False
    )

    print(f"Total de registros: {len(df):,}")

    text_builder = TextBuilderService()

    df_valid = df[~df['DS_ASSUNTO'].apply(text_builder.needs_classification)].copy()
    df_valid['texto_referencia'] = df_valid.apply(text_builder.build_text_from_row, axis=1)
    df_valid = df_valid[df_valid['texto_referencia'].apply(lambda x: text_builder.validate_text_length(x, 10))]

    print(f"Registros validos: {len(df_valid):,}")

    print("\nDividindo em treino (80%) e teste (20%)...")

    train_df, test_df = train_test_split(
        df_valid,
        test_size=0.2,
        random_state=42,
        stratify=df_valid['DS_ASSUNTO']
    )

    print(f"Treino: {len(train_df):,} registros")
    print(f"Teste: {len(test_df):,} registros")

    print("\nGerando embeddings do conjunto de TREINO...")

    embedding_service = EmbeddingService(
        model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    )

    train_embeddings = embedding_service.generate_embeddings(train_df['texto_referencia'].tolist())

    print(f"Embeddings gerados: {train_embeddings.shape}")

    temp_dir = Path("data/ml/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    with open(temp_dir / 'train_reference.pkl', 'wb') as f:
        pickle.dump(train_df[['DS_ASSUNTO', 'texto_referencia']], f)

    np.save(temp_dir / 'train_embeddings.npy', train_embeddings)

    print("\n" + "="*80)
    print("TESTANDO DIFERENTES THRESHOLDS")
    print("="*80)

    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    results = []

    for threshold in thresholds:
        print(f"\nThreshold: {threshold}")

        from src.services.AssuntoClassifierService import AssuntoClassifierService

        classifier = AssuntoClassifierService(
            embedding_service=embedding_service,
            reference_data_path=str(temp_dir / 'train_reference.pkl'),
            reference_embeddings_path=str(temp_dir / 'train_embeddings.npy'),
            threshold=threshold,
            k_neighbors=5
        )

        correct = 0
        classified = 0
        total = len(test_df)

        confusion = {}  # Para matriz de confusão simplificada

        for idx, row in test_df.iterrows():
            texto = row['texto_referencia']
            real_category = normalizar_categoria(row['DS_ASSUNTO'])

            result = classifier.classify_assunto(texto)
            predicted_category = normalizar_categoria(result['categoria'])

            if predicted_category:
                classified += 1

                if predicted_category == real_category:
                    correct += 1
                else:
                    key = f"{real_category} -> {predicted_category}"
                    confusion[key] = confusion.get(key, 0) + 1

        taxa_classificacao = (classified / total) * 100
        acuracia = (correct / classified * 100) if classified > 0 else 0
        acuracia_total = (correct / total) * 100

        print(f"   Total testado: {total}")
        print(f"   Classificados: {classified} ({taxa_classificacao:.1f}%)")
        print(f"   Acertos: {correct}")
        print(f"   Acuracia (dos classificados): {acuracia:.1f}%")
        print(f"   Acuracia total: {acuracia_total:.1f}%")

        results.append({
            'threshold': threshold,
            'total': total,
            'classified': classified,
            'correct': correct,
            'taxa_classificacao': taxa_classificacao,
            'acuracia': acuracia,
            'acuracia_total': acuracia_total
        })

    print("\n" + "="*80)
    print("RESUMO DOS RESULTADOS")
    print("="*80)

    print("\n{:<12} {:<15} {:<20} {:<15}".format(
        "Threshold", "Taxa Classif.", "Acuracia (classif.)", "Acuracia Total"
    ))
    print("-" * 80)

    for r in results:
        print("{:<12.2f} {:<15.1f}% {:<20.1f}% {:<15.1f}%".format(
            r['threshold'],
            r['taxa_classificacao'],
            r['acuracia'],
            r['acuracia_total']
        ))

    best = max(results, key=lambda x: x['acuracia_total'])
    print("\n" + "="*80)
    print("MELHOR CONFIGURACAO")
    print("="*80)
    print(f"Threshold: {best['threshold']}")
    print(f"Taxa de classificacao: {best['taxa_classificacao']:.1f}%")
    print(f"Acuracia (dos classificados): {best['acuracia']:.1f}%")
    print(f"Acuracia total: {best['acuracia_total']:.1f}%")

    import shutil
    shutil.rmtree(temp_dir)
    print("\nArquivos temporarios removidos.")


if __name__ == "__main__":
    main()
