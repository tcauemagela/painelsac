"""
Interface para serviço de geração de embeddings.
"""
from abc import ABC, abstractmethod
from typing import List
import numpy as np


class IEmbeddingService(ABC):
    """
    Interface para serviço de geração de embeddings de texto usando Sentence Transformers.
    """

    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Gera embeddings para uma lista de textos.

        Args:
            texts: Lista de strings para gerar embeddings

        Returns:
            numpy.ndarray: Matriz de embeddings (len(texts), embedding_dimension)
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Retorna a dimensão dos embeddings gerados pelo modelo.

        Returns:
            int: Dimensão dos vetores de embedding
        """
        pass
