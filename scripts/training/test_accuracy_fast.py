"""
Teste RAPIDO de acuracia usando amostra pequena.
"""
import pandas as pd
import numpy as np
from src.services.EmbeddingService import EmbeddingService
from src.services.AssuntoClassifierService import AssuntoClassifierService
from src.services.TextBuilderService import TextBuilderService


def normalizar_categoria(cat):
    """Normaliza categoria para comparação."""
    if pd.isna(cat):
        return None
    return cat.upper().replace('É', 'E').replace('Í', 'I').replace('Ó', 'O').strip()


def main():
    print("="*80)
    print("TESTE RAPIDO DE ACURACIA")
    print("="*80)

    print("\nCarregando dados...")
    TRAINING_FILE = r"C:\Users\Usuário\Desktop\projetos_ia\monitoramento_nip\SACS_FEITOS_01_07_AO_03_11.CSV"

    df = pd.read_csv(
        TRAINING_FILE,
        sep=';',
        encoding='latin1',
        low_memory=False
    )

    text_builder = TextBuilderService()
    df_valid = df[~df['DS_ASSUNTO'].apply(text_builder.needs_classification)].copy()
    df_valid['texto_referencia'] = df_valid.apply(text_builder.build_text_from_row, axis=1)
    df_valid = df_valid[df_valid['texto_referencia'].apply(lambda x: text_builder.validate_text_length(x, 10))]

    print(f"Total validos: {len(df_valid):,}")
    print("\nCriando amostra de teste (200 registros)...")

    test_sample = df_valid.sample(min(200, len(df_valid)), random_state=42)

    print(f"Amostra de teste: {len(test_sample)} registros")

    print(f"Amostra de teste: {len(test_sample)} registros")
    print("\nDistribuicao:")
    for cat, count in test_sample['DS_ASSUNTO'].value_counts().items():
        print(f"   - {cat}: {count}")

    print("\nCarregando classificador pre-treinado...")

    embedding_service = EmbeddingService(
        model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    )

    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]

    print("\n" + "="*80)
    print("TESTANDO DIFERENTES THRESHOLDS")
    print("="*80)

    results = []

    for threshold in thresholds:
        print(f"\n[Threshold: {threshold}]")

        classifier = AssuntoClassifierService(
            embedding_service=embedding_service,
            threshold=threshold,
            k_neighbors=5
        )

        correct = 0
        classified = 0
        total = len(test_sample)

        for idx, row in test_sample.iterrows():
            real_category = normalizar_categoria(row['DS_ASSUNTO'])
            texto = row['texto_referencia']

            result = classifier.classify_assunto(texto)
            predicted_category = normalizar_categoria(result['categoria'])

            if predicted_category:
                classified += 1
                if predicted_category == real_category:
                    correct += 1

        taxa_classificacao = (classified / total) * 100
        acuracia = (correct / classified * 100) if classified > 0 else 0
        acuracia_total = (correct / total) * 100

        print(f"   Classificados: {classified}/{total} ({taxa_classificacao:.1f}%)")
        print(f"   Acertos: {correct}/{classified}")
        print(f"   Acuracia dos classificados: {acuracia:.1f}%")
        print(f"   Acuracia total: {acuracia_total:.1f}%")

        results.append({
            'threshold': threshold,
            'classified': classified,
            'correct': correct,
            'taxa_classificacao': taxa_classificacao,
            'acuracia': acuracia,
            'acuracia_total': acuracia_total
        })

    print("\n" + "="*80)
    print("RESUMO")
    print("="*80)

    print("\n{:<12} {:<18} {:<22} {:<15}".format(
        "Threshold", "Taxa Classif.", "Acuracia (classif.)", "Acuracia Total"
    ))
    print("-" * 80)

    for r in results:
        print("{:<12.2f} {:<18.1f}% {:<22.1f}% {:<15.1f}%".format(
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
    print(f"Acuracia dos classificados: {best['acuracia']:.1f}%")
    print(f"Acuracia total: {best['acuracia_total']:.1f}%")

    print("\n" + "="*80)
    print("INTERPRETACAO")
    print("="*80)
    print(f"Com threshold {best['threshold']}:")
    print(f"  - {best['taxa_classificacao']:.0f}% dos registros SAO classificados automaticamente")
    print(f"  - Desses, {best['acuracia']:.0f}% estao CORRETOS")
    print(f"  - {100-best['taxa_classificacao']:.0f}% ficam vazios (precisam classificacao manual)")


if __name__ == "__main__":
    main()
