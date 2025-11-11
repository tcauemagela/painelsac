"""
Service for automatic SUB_ASSUNTO classification using K-NN with embeddings.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import pickle

from ...domain.interfaces.IEmbeddingService import IEmbeddingService
from ....shared.builders.TextBuilderService import TextBuilderService


class SubAssuntoClassifierService:
    """
    Classifier for SUB_ASSUNTO using K-NN with semantic embeddings.
    Uses pre-trained embeddings loaded from disk.
    """

    def __init__(
        self,
        embedding_service: IEmbeddingService,
        reference_data_path: str = 'data/ml/subassunto_reference.pkl',
        reference_embeddings_path: str = 'data/ml/subassunto_embeddings.npy',
        threshold: float = 0.45,
        k_neighbors: int = 5
    ):
        """
        Initialize the classifier with pre-trained data.

        Args:
            embedding_service: Service for generating embeddings
            reference_data_path: Path to reference data (pickle)
            reference_embeddings_path: Path to reference embeddings (numpy)
            threshold: Minimum confidence threshold for auto-classification (0-1)
            k_neighbors: Number of neighbors for K-NN voting
        """
        self.embedding_service = embedding_service
        self.text_builder = TextBuilderService()
        self.threshold = threshold
        self.k_neighbors = k_neighbors

        self.reference_df = self._load_reference_data(reference_data_path)
        self.reference_embeddings = self._load_embeddings(reference_embeddings_path)

        try:
            print(f"Classificador SUB_ASSUNTO carregado com {len(self.reference_df)} registros de referencia")
            print(f"Threshold: {self.threshold}, K-neighbors: {self.k_neighbors}")
        except (OSError, IOError):
            pass  # Stdout not available, skip logging

    def _load_reference_data(self, path: str) -> pd.DataFrame:
        """Load reference DataFrame from disk."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(
                f"Reference data not found: {path}\n"
                "Please run train_subassunto_classifier_fast.py first to generate training data."
            )

        with open(file_path, 'rb') as f:
            df = pickle.load(f)

        return df

    def _load_embeddings(self, path: str) -> np.ndarray:
        """Load reference embeddings from disk."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(
                f"Reference embeddings not found: {path}\n"
                "Please run train_subassunto_classifier_fast.py first to generate embeddings."
            )

        embeddings = np.load(file_path)
        return embeddings

    def needs_classification(self, sub_assunto: str) -> bool:
        """
        Check if a SUB_ASSUNTO value needs classification.
        Identifica: vazios, "OUTRO", "OUTROS", e qualquer variação como "OUTROS (DETALHAR...)"

        Args:
            sub_assunto: Value to check

        Returns:
            bool: True if needs classification
        """
        if pd.isna(sub_assunto):
            return True

        sub_assunto_str = str(sub_assunto).strip().upper()

        if sub_assunto_str == '':
            return True

        if 'OUTRO' in sub_assunto_str:
            return True

        return False

    def classify_dataframe(self, df: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
        """
        Classify all records in a DataFrame that need classification.
        APENAS PREENCHE SUB_ASSUNTO quando vazio ou "Outros".

        Args:
            df: DataFrame with text columns
            progress_callback: Function to call with progress (0.0 to 1.0)

        Returns:
            DataFrame: DataFrame com SUB_ASSUNTO preenchido
        """
        df_result = df.copy()

        mask_needs_classification = df_result['SUB_ASSUNTO'].apply(
            self.needs_classification
        )
        indices_to_classify = df_result[mask_needs_classification].index.tolist()

        total_to_classify = len(indices_to_classify)

        try:
            print(f"\nClassificando {total_to_classify} SUB_ASSUNTO em mini-batches...")
        except (OSError, IOError):
            pass

        if total_to_classify == 0:
            return df_result

        classified_count = 0
        not_classified_count = 0

        BATCH_SIZE = 1000

        for batch_start in range(0, total_to_classify, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_to_classify)
            batch_indices = indices_to_classify[batch_start:batch_end]

            try:
                print(f"   Batch {batch_start//BATCH_SIZE + 1}/{(total_to_classify + BATCH_SIZE - 1)//BATCH_SIZE}: registros {batch_start+1}-{batch_end}")
            except (OSError, IOError):
                pass

            batch_texts = [
                self.text_builder.build_text_from_row(df_result.loc[idx])
                for idx in batch_indices
            ]

            batch_embeddings = self.embedding_service.generate_embeddings(batch_texts)

            all_similarities = cosine_similarity(batch_embeddings, self.reference_embeddings)

            for i, idx in enumerate(batch_indices):
                similarities = all_similarities[i]

                top_k_indices = np.argsort(similarities)[-self.k_neighbors:][::-1]
                top_k_scores = similarities[top_k_indices]

                best_category = self.reference_df.iloc[top_k_indices[0]]['SUB_ASSUNTO']
                avg_score = top_k_scores[0]  # Usar apenas o melhor score para velocidade

                df_result.loc[idx, 'SUB_ASSUNTO'] = best_category
                if avg_score >= self.threshold:
                    classified_count += 1
                else:
                    not_classified_count += 1

            if progress_callback:
                progress = batch_end / total_to_classify
                progress_callback(progress)

        try:
            print(f"\nResultado da Classificacao SUB_ASSUNTO:")
            print(f"   - Classificados: {classified_count}")
            print(f"   - Nao classificados (confianca baixa): {not_classified_count}")
        except (OSError, IOError):
            pass

        return df_result

    def get_classification_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about SUB_ASSUNTO distribution.

        Args:
            df: DataFrame

        Returns:
            dict: Classification statistics
        """
        stats = {
            'total_registros': len(df),
            'total_vazios': df['SUB_ASSUNTO'].apply(self.needs_classification).sum(),
        }

        if 'SUB_ASSUNTO' in df.columns:
            stats['distribuicao_categorias'] = df['SUB_ASSUNTO'].value_counts().to_dict()

        return stats
