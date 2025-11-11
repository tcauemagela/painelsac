import sqlite3
import pickle
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List
from pathlib import Path
from ...core.domain.interfaces.ICacheService import ICacheService


class SQLiteCacheService(ICacheService):
    """Implementação de cache e auditoria usando SQLite"""

    def __init__(self, db_path: str = "./data/cache.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Inicializa o banco de dados"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS upload_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                num_records INTEGER,
                user TEXT,
                metadata TEXT,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT,
                query_type TEXT,
                user TEXT,
                response_time_ms REAL,
                executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Armazena um valor no cache"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            pickled_value = pickle.dumps(value)
            expires_at = datetime.now() + timedelta(seconds=ttl_seconds)

            cursor.execute("""
                INSERT OR REPLACE INTO cache (key, value, expires_at)
                VALUES (?, ?, ?)
            """, (key, pickled_value, expires_at))

            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    def get(self, key: str) -> Optional[Any]:
        """Recupera um valor do cache"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT value, expires_at FROM cache WHERE key = ?
            """, (key,))

            result = cursor.fetchone()
            conn.close()

            if result:
                value, expires_at = result
                expires_at = datetime.fromisoformat(expires_at)

                if expires_at > datetime.now():
                    return pickle.loads(value)
                else:
                    self.delete(key)
                    return None

            return None
        except Exception:
            return None

    def delete(self, key: str) -> bool:
        """Remove um valor do cache"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    def clear_expired(self) -> int:
        """Remove itens expirados"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                DELETE FROM cache WHERE expires_at < ?
            """, (datetime.now(),))

            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            return deleted
        except Exception:
            return 0

    def log_upload(self, filename: str, num_records: int, user: str = "system", metadata: Dict = None) -> bool:
        """Registra upload de arquivo"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            metadata_str = str(metadata) if metadata else ""

            cursor.execute("""
                INSERT INTO upload_audit (filename, num_records, user, metadata)
                VALUES (?, ?, ?, ?)
            """, (filename, num_records, user, metadata_str))

            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    def log_query(self, query_text: str, query_type: str, user: str = "system", response_time_ms: float = 0) -> bool:
        """Registra query executada"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO query_audit (query_text, query_type, user, response_time_ms)
                VALUES (?, ?, ?, ?)
            """, (query_text, query_type, user, response_time_ms))

            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    def get_upload_history(self, limit: int = 10) -> List[Dict]:
        """Retorna histórico de uploads"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT filename, num_records, user, uploaded_at
                FROM upload_audit
                ORDER BY uploaded_at DESC
                LIMIT ?
            """, (limit,))

            results = cursor.fetchall()
            conn.close()

            return [
                {
                    'filename': row[0],
                    'num_records': row[1],
                    'user': row[2],
                    'uploaded_at': row[3]
                }
                for row in results
            ]
        except Exception:
            return []

    def get_query_stats(self, hours: int = 24) -> Dict:
        """Retorna estatísticas de queries"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            since = datetime.now() - timedelta(hours=hours)

            cursor.execute("""
                SELECT
                    COUNT(*) as total_queries,
                    AVG(response_time_ms) as avg_response_time_ms,
                    query_type,
                    COUNT(*) as count
                FROM query_audit
                WHERE executed_at > ?
                GROUP BY query_type
            """, (since,))

            results = cursor.fetchall()
            conn.close()

            if not results:
                return {
                    'total_queries': 0,
                    'avg_response_time_ms': 0,
                    'query_types': {}
                }

            total = sum(row[3] for row in results)
            avg_time = sum(row[1] for row in results if row[1]) / len(results) if results else 0

            query_types = {row[2]: row[3] for row in results}

            return {
                'total_queries': total,
                'avg_response_time_ms': avg_time,
                'query_types': query_types
            }
        except Exception:
            return {
                'total_queries': 0,
                'avg_response_time_ms': 0,
                'query_types': {}
            }
