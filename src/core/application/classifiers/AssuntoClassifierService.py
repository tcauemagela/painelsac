"""
Service for automatic DS_ASSUNTO classification using K-NN with embeddings.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import pickle

from ...domain.interfaces.IAssuntoClassifier import IAssuntoClassifier
from ...domain.interfaces.IEmbeddingService import IEmbeddingService
from ...domain.entities.ClassificationResult import ClassificationResult
from ....shared.builders.TextBuilderService import TextBuilderService


class AssuntoClassifierService(IAssuntoClassifier):
    """
    Concrete implementation of IAssuntoClassifier using K-NN with semantic embeddings.
    Uses pre-trained embeddings loaded from disk.
    """

    def __init__(
        self,
        embedding_service: IEmbeddingService,
        reference_data_path: str = 'data/ml/assunto_reference.pkl',
        reference_embeddings_path: str = 'data/ml/assunto_embeddings.npy',
        threshold: float = 0.65,
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
            print(f"Classificador carregado com {len(self.reference_df)} registros de referencia")
            print(f"Threshold: {self.threshold}, K-neighbors: {self.k_neighbors}")
        except (OSError, IOError):
            pass  # Stdout not available, skip logging

    def _load_reference_data(self, path: str) -> pd.DataFrame:
        """Load reference DataFrame from disk."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(
                f"Reference data not found: {path}\n"
                "Please run train_assunto_classifier.py first to generate training data."
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
                "Please run train_assunto_classifier.py first to generate embeddings."
            )

        embeddings = np.load(file_path)
        return embeddings

    def classify_assunto(self, texto: str) -> Dict[str, Any]:
        """
        Classify a single text into a DS_ASSUNTO category.

        Args:
            texto: Text to classify

        Returns:
            dict: Classification result with category, confidence, method, and similar texts
        """
        if not self.text_builder.validate_text_length(texto, min_length=10):
            return {
                'categoria': None,
                'confianca': 0.0,
                'metodo': 'manual_review',
                'top_similares': []
            }

        texto_embedding = self.embedding_service.generate_embeddings([texto])

        similarities = cosine_similarity(texto_embedding, self.reference_embeddings)[0]

        top_k_indices = np.argsort(similarities)[-self.k_neighbors:][::-1]
        top_k_scores = similarities[top_k_indices]

        top_k_categories = self.reference_df.iloc[top_k_indices]['DS_ASSUNTO'].tolist()
        top_k_texts = self.reference_df.iloc[top_k_indices]['texto_referencia'].tolist()

        category_scores = {}
        for cat, score in zip(top_k_categories, top_k_scores):
            category_scores[cat] = category_scores.get(cat, 0) + score

        best_category = max(category_scores, key=category_scores.get)

        category_count = sum(1 for c in top_k_categories if c == best_category)
        avg_score = category_scores[best_category] / category_count

        if avg_score < self.threshold:
            return {
                'categoria': None,
                'confianca': avg_score,
                'metodo': 'manual_review',
                'top_similares': [t[:100] + '...' for t in top_k_texts[:3]]
            }

        return {
            'categoria': best_category,
            'confianca': avg_score,
            'metodo': 'auto',
            'top_similares': [t[:100] + '...' for t in top_k_texts[:3]]
        }

    def classify_dataframe(self, df: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
        """
        Classify all records in a DataFrame that need classification.
        APENAS PREENCHE DS_ASSUNTO quando vazio ou "Outros".
        Usa MINI-BATCHES para processar mais rápido.

        Args:
            df: DataFrame with text columns
            progress_callback: Function to call with progress (0.0 to 1.0)

        Returns:
            DataFrame: DataFrame com DS_ASSUNTO preenchido
        """
        df_result = df.copy()

        mask_needs_classification = df_result['DS_ASSUNTO'].apply(
            self.text_builder.needs_classification
        )
        indices_to_classify = df_result[mask_needs_classification].index.tolist()

        total_to_classify = len(indices_to_classify)

        try:
            print(f"\nClassificando {total_to_classify} registros em mini-batches...")
        except (OSError, IOError):
            pass

        if total_to_classify == 0:
            return df_result

        classified_count = 0
        not_classified_count = 0

        BATCH_SIZE = 500

        for batch_start in range(0, total_to_classify, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_to_classify)
            batch_indices = indices_to_classify[batch_start:batch_end]

            try:
                print(f"   Processando batch {batch_start//BATCH_SIZE + 1}/{(total_to_classify + BATCH_SIZE - 1)//BATCH_SIZE}: registros {batch_start+1}-{batch_end}")
            except (OSError, IOError):
                pass

            batch_texts = [
                self.text_builder.build_text_from_row(df_result.loc[idx])
                for idx in batch_indices
            ]

            batch_embeddings = self.embedding_service.generate_embeddings(batch_texts)

            for i, idx in enumerate(batch_indices):
                texto_embedding = batch_embeddings[i:i+1]  # Pegar embedding individual

                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(texto_embedding, self.reference_embeddings)[0]

                top_k_indices = np.argsort(similarities)[-self.k_neighbors:][::-1]
                top_k_scores = similarities[top_k_indices]
                top_k_categories = self.reference_df.iloc[top_k_indices]['DS_ASSUNTO'].tolist()

                category_scores = {}
                for cat, score in zip(top_k_categories, top_k_scores):
                    category_scores[cat] = category_scores.get(cat, 0) + score

                best_category = max(category_scores, key=category_scores.get)
                category_count = sum(1 for c in top_k_categories if c == best_category)
                avg_score = category_scores[best_category] / category_count

                if avg_score >= self.threshold:
                    df_result.loc[idx, 'DS_ASSUNTO'] = best_category
                    classified_count += 1
                else:
                    not_classified_count += 1

            if progress_callback:
                progress = batch_end / total_to_classify
                progress_callback(progress)

        try:
            print(f"\nResultado da Classificacao:")
            print(f"   - Classificados: {classified_count}")
            print(f"   - Nao classificados (confianca baixa): {not_classified_count}")
        except (OSError, IOError):
            pass

        return df_result

    def get_classification_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about DS_ASSUNTO distribution.

        Args:
            df: DataFrame

        Returns:
            dict: Classification statistics
        """
        stats = {
            'total_registros': len(df),
            'total_vazios': df['DS_ASSUNTO'].apply(self.text_builder.needs_classification).sum(),
        }

        if 'DS_ASSUNTO' in df.columns:
            stats['distribuicao_categorias'] = df['DS_ASSUNTO'].value_counts().to_dict()

        return stats
