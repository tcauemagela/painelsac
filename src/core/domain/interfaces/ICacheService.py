from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List


class ICacheService(ABC):
    """Interface para serviço de cache e auditoria"""

    @abstractmethod
    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Armazena um valor no cache com TTL"""
        pass

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Recupera um valor do cache"""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Remove um valor do cache"""
        pass

    @abstractmethod
    def clear_expired(self) -> int:
        """Remove itens expirados do cache"""
        pass

    @abstractmethod
    def log_upload(self, filename: str, num_records: int, user: str = "system", metadata: Dict = None) -> bool:
        """Registra upload de arquivo"""
        pass

    @abstractmethod
    def log_query(self, query_text: str, query_type: str, user: str = "system", response_time_ms: float = 0) -> bool:
        """Registra query executada"""
        pass

    @abstractmethod
    def get_upload_history(self, limit: int = 10) -> List[Dict]:
        """Retorna histórico de uploads"""
        pass

    @abstractmethod
    def get_query_stats(self, hours: int = 24) -> Dict:
        """Retorna estatísticas de queries"""
        pass
