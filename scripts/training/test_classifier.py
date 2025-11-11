"""
Script de teste do classificador de DS_ASSUNTO.

Testa o classificador com exemplos de cada categoria.
"""
import pandas as pd
from src.services.EmbeddingService import EmbeddingService
from src.services.AssuntoClassifierService import AssuntoClassifierService


def main():
    print("="*80)
    print("TESTE DO CLASSIFICADOR DE DS_ASSUNTO")
    print("="*80)

    print("\nCarregando classificador...")

    embedding_service = EmbeddingService(
        model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    )

    classifier = AssuntoClassifierService(
        embedding_service=embedding_service,
        threshold=0.45,  # Threshold mais baixo para melhor recall
        k_neighbors=5
    )

    print("Classificador carregado com sucesso!\n")

    test_cases = [
        {
            'texto': 'BENEFICIARIO INFORMA QUE NAO CONSEGUE BAIXAR SUAS RECEITAS NEM PEDIDOS DE EXAMES PELO APLICATIVO. JA TENTOU VARIAS VEZES E CONTINUA DANDO ERRO. SOLICITA AJUDA PARA RESOLVER O PROBLEMA TECNICO NO APP.',
            'categoria_esperada': 'PROBLEMAS TECNICOS SITE/APP'
        },
        {
            'texto': 'CLIENTE RELATA QUE AGENDOU TELECONSULTA MAS QUANDO ENTRA NA SALA FICA TUDO ESCURO E NAO CONSEGUE VER O MEDICO. JA TENTOU DUAS VEZES E NAS DUAS NAO CONSEGUIU ATENDIMENTO. PROBLEMA DE ACESSO AO PORTAL.',
            'categoria_esperada': 'TELECONSULTA ELETIVA_ACESSO PORTAL'
        },
        {
            'texto': 'FEZ TELECONSULTA MAS O DOCUMENTO ELETRONICO NAO FOI GERADO. PRECISA DA RECEITA E DO ATESTADO MEDICO. SOLICITOU VIA EMAIL MAS AINDA NAO RECEBEU O DOCUMENTO ELETRONICO DA CONSULTA.',
            'categoria_esperada': 'TELECONSULTA ELETIVA_DOC. ELETRONICO'
        },
        {
            'texto': 'BENEFICIARIO RELATA PROBLEMA NA BIOMETRIA FACIAL PARA CONSULTAS DE TELEMEDICINA. O SISTEMA NAO RECONHECE O ROSTO E NAO PERMITE ACESSO. JA TENTOU VARIAS VEZES COM BOA ILUMINACAO.',
            'categoria_esperada': 'TELECONSULTA ELETIVA_BIOMETRIA FACIAL'
        },
        {
            'texto': 'LIGOU VARIAS VEZES PARA CENTRAL MAS A URA NAO FUNCIONA E CAI A LIGACAO. TELEFONE APRESENTA PROBLEMAS TECNICOS E NAO CONSEGUE COMPLETAR O ATENDIMENTO. SOLICITA VERIFICACAO DO SISTEMA.',
            'categoria_esperada': 'PROBLEMAS TECNICOS DE TELEFONIA/URA'
        },
        {
            'texto': 'SOLICITA AUDITORIA E ATUALIZACAO DO EMAIL E TELEFONE NO CADASTRO. DADOS ESTAO DESATUALIZADOS NO SISTEMA. PRECISA CORRIGIR AS INFORMACOES DE CONTATO.',
            'categoria_esperada': 'AUDITORIA DE EMAIL/TELEFONE - T.I'
        }
    ]

    print("="*80)
    print("TESTANDO CLASSIFICACAO")
    print("="*80)

    acertos = 0
    total = len(test_cases)

    for i, test in enumerate(test_cases, 1):
        print(f"\n[TESTE {i}/{total}]")
        print(f"Texto: {test['texto'][:80]}...")

        resultado = classifier.classify_assunto(test['texto'])

        print(f"\nResultado:")
        print(f"   Categoria prevista: {resultado['categoria']}")
        print(f"   Confianca: {resultado['confianca']:.4f}")
        print(f"   Metodo: {resultado['metodo']}")

        if 'categoria_esperada' in test:
            esperada = test['categoria_esperada']
            prevista = resultado['categoria']

            esperada_norm = esperada.upper().replace('�', 'E').replace('Í', 'I')
            prevista_norm = prevista.upper().replace('�', 'E').replace('Í', 'I') if prevista else ''

            acertou = esperada_norm == prevista_norm

            if acertou:
                print(f"   Status: ACERTOU")
                acertos += 1
            else:
                print(f"   Status: ERROU")
                print(f"   Esperado: {esperada}")

        print(f"\nTextos similares usados:")
        for j, similar in enumerate(resultado['top_similares'][:2], 1):
            print(f"   {j}. {similar[:80]}...")

    print("\n" + "="*80)
    print("RESUMO DOS TESTES")
    print("="*80)
    print(f"Total de testes: {total}")
    print(f"Acertos: {acertos}")
    print(f"Erros: {total - acertos}")
    print(f"Acuracia: {acertos/total*100:.1f}%")

    print("\n" + "="*80)
    print("TESTE COM DATAFRAME")
    print("="*80)

    test_df = pd.DataFrame([
        {
            'DS_ASSUNTO': '',
            'DS_OBSERVACAO': 'O aplicativo nao abre e fica travado',
            'DS_MOTIVO': 'Problemas tecnicos',
            'SUB_ASSUNTO': ''
        },
        {
            'DS_ASSUNTO': 'Outros',
            'DS_OBSERVACAO': 'Fiz teleconsulta mas o documento nao chegou',
            'DS_MOTIVO': 'Documento eletronico',
            'SUB_ASSUNTO': ''
        },
        {
            'DS_ASSUNTO': 'PROBLEMAS TECNICOS SITE/APP',  # Ja classificado - nao deve mudar
            'DS_OBSERVACAO': 'Site lento',
            'DS_MOTIVO': '',
            'SUB_ASSUNTO': ''
        }
    ])

    print(f"\nDataFrame de teste com {len(test_df)} registros:")
    print(f"   - Registros vazios/Outros: 2")
    print(f"   - Registros ja classificados: 1")

    df_classified = classifier.classify_dataframe(test_df)

    print("\nResultados:")
    for idx, row in df_classified.iterrows():
        print(f"\nRegistro {idx + 1}:")
        print(f"   DS_ASSUNTO: {row['DS_ASSUNTO']}")
        print(f"   Auto-categorizado: {row['AUTO_CATEGORIZADO_ASSUNTO']}")
        print(f"   Confianca: {row['CONFIANCA_ASSUNTO']:.4f}")
        print(f"   Requer revisao: {row['REQUER_REVISAO_ASSUNTO']}")

    stats = classifier.get_classification_stats(df_classified)
    print("\nEstatisticas:")
    print(f"   Total de registros: {stats['total_registros']}")
    print(f"   Auto-classificados: {stats['total_auto_classificados']}")
    print(f"   Requerem revisao: {stats['total_requer_revisao']}")

    if stats['total_auto_classificados'] > 0:
        print(f"   Confianca media: {stats['confianca_media']:.4f}")

    print("\n" + "="*80)
    print("TESTES CONCLUIDOS!")
    print("="*80)


if __name__ == "__main__":
    main()
