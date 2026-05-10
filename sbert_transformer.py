"""
Опциональные семантические признаки (Sentence-BERT) для обучения классификатора блоков.

Подключение: установите зависимости `pip install -r requirements-ml.txt`,
затем при обучении задайте USE_SBERT=1.

Модель по умолчанию лёгкая и мультиязычная (подходит для рус/анг текста блоков).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

LOGGER = logging.getLogger(__name__)

# Размер вектора для модели по умолчанию (paraphrase-multilingual-MiniLM-L12-v2).
_DEFAULT_EMB_DIM = 384
_SBERT_MISSING_WARNED = False

DEFAULT_MODEL = os.getenv(
    "SBERT_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)


class LazySentenceEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """Векторизация текста блока через предобученный SentenceTransformer (без дообучения)."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or DEFAULT_MODEL
        self._model = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            # Пайплайн обучен с USE_SBERT=1; в контейнере часто нет torch/sentence-transformers.
            # Нули той же размерности, что у эмбеддинга модели по умолчанию — предсказание хуже, но не падает.
            global _SBERT_MISSING_WARNED
            if not _SBERT_MISSING_WARNED:
                LOGGER.warning(
                    "Пакет sentence_transformers недоступен; SBERT-признаки заменены нулями "
                    "(качество классификатора ниже). Установите: pip install -r requirements-ml.txt"
                )
                _SBERT_MISSING_WARNED = True
            if hasattr(X, "fillna"):
                n = len(X)
            else:
                n = len(list(X))
            dim = int(os.getenv("SBERT_FALLBACK_DIM", str(_DEFAULT_EMB_DIM)))
            return np.zeros((n, dim), dtype=np.float32)

        if hasattr(X, "fillna"):
            texts = X.fillna("").astype(str).tolist()
        else:
            texts = [str(x) for x in X]

        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

        emb = self._model.encode(texts, batch_size=int(os.getenv("SBERT_BATCH", "48")), show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_model"] = None
        return d
