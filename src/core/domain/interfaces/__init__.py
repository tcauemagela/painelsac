"""
Interfaces module - Abstractions for dependency inversion
"""
from .IExcelReader import IExcelReader
from .IDataValidator import IDataValidator
from .IColumnMapper import IColumnMapper
from .IDashboardGenerator import IDashboardGenerator
from .IReportExporter import IReportExporter
from .ICacheService import ICacheService
from .IEmbeddingService import IEmbeddingService
from .IAssuntoClassifier import IAssuntoClassifier

__all__ = [
    'IExcelReader',
    'IDataValidator',
    'IColumnMapper',
    'IDashboardGenerator',
    'IReportExporter',
    'ICacheService',
    'IEmbeddingService',
    'IAssuntoClassifier',
]
