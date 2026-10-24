"""LangChain vector store using OriginChainDB's official Python client."""

from __future__ import annotations

import math
import uuid
from collections.abc import Callable, Iterable
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from ._support import (
    delete_ids,
    equality_filter,
    read_row,
    validate_config,
    validate_id,
    validate_k,
    validate_vector,
)


class OriginChainVectorStore(VectorStore):
    """Store vectors and optionally their document content in OriginChainDB.

    The caller owns ``client``. Create ``text_schema`` with string primary key
    ``id``, string ``text`` and JSON ``metadata`` before adding documents.
    Without it, search returns ID-only Documents. Writes span two endpoints;
    failures propagate and callers may retry with the same explicit IDs.
    """

    def __init__(
        self,
        *,
        client: Any,
        table: str,
        dim: int,
        embedding: Embeddings,
        metric: str = "cosine",
        text_schema: str | None = None,
        mode: str | None = None,
    ) -> None:
        validate_config(table, text_schema, dim, metric, mode)
        self._client = client
        self._table = table
        self._dim = dim
        self._embedding = embedding
        self._metric = metric
        self._text_schema = text_schema
        self._mode = mode

    @property
    def embeddings(self) -> Embeddings | None:
        return self._embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        texts_list = list(texts)
        if ids is None:
            ids = [uuid.uuid4().hex for _ in texts_list]
        if metadatas is None:
            metadatas = [{} for _ in texts_list]
        if len(ids) != len(texts_list) or len(metadatas) != len(texts_list):
            raise ValueError("texts, ids and metadatas must have equal lengths")
        for row_id in ids:
            validate_id(row_id)
        if any(not isinstance(text, str) for text in texts_list):
            raise ValueError("texts must contain strings")
        if any(not isinstance(meta, dict) for meta in metadatas):
            raise ValueError("metadatas must contain dictionaries")
        if not texts_list:
            return []
        vectors = self._embedding.embed_documents(texts_list)
        if len(vectors) != len(texts_list):
            raise ValueError("embedding returned wrong number of vectors")
        for vector in vectors:
            validate_vector(vector, self._dim)
        for row_id, text, meta, vector in zip(ids, texts_list, metadatas, vectors):
            if self._text_schema is not None:
                self._client.rows.put(
                    self._text_schema,
                    {
                        "id": row_id,
                        "text": text,
                        "metadata": dict(meta),
                    },
                )
            self._client.vector_put(
                self._table,
                id=row_id,
                embedding=list(vector),
                dim=self._dim,
                metric=self._metric,
                metadata=dict(meta) or None,
            )
        return list(ids)

    def similarity_search(
        self, query: str, k: int = 4, filter: dict[str, Any] | None = None, **kwargs: Any
    ) -> list[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k, filter, **kwargs)]

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        return [doc for doc, _ in self._topk_with_score(embedding, k=k, filter=filter)]

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: dict[str, Any] | None = None, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        validate_k(k)
        equality_filter(filter)
        return self._topk_with_score(self._embedding.embed_query(query), k=k, filter=filter)

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Delete vectors and configured rows; raise on any partial failure.

        ``ids=None`` is rejected to prevent an ambiguous delete-all operation.
        An empty list is a successful no-op. Missing IDs are idempotent.
        """
        if ids is None:
            raise ValueError("delete requires explicit ids; delete-all is not supported")
        delete_ids(self._client, self._table, self._text_schema, ids)
        return True

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        *,
        client: Any = None,
        table: str = "langchain",
        dim: int | None = None,
        metric: str = "cosine",
        text_schema: str | None = None,
        ids: list[str] | None = None,
        mode: str | None = None,
        **kwargs: Any,
    ) -> OriginChainVectorStore:
        if client is None:
            raise ValueError("from_texts requires client= (an OriginChain instance)")
        if dim is None:
            if not texts:
                raise ValueError("dim= required when texts is empty")
            dim = len(embedding.embed_query(texts[0]))
        store = cls(
            client=client,
            table=table,
            dim=dim,
            embedding=embedding,
            metric=metric,
            text_schema=text_schema,
            mode=mode,
        )
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store

    def _topk_with_score(
        self, vector: list[float], *, k: int, filter: dict[str, Any] | None
    ) -> list[tuple[Document, float]]:
        validate_k(k)
        validate_vector(vector, self._dim)
        hits = self._client.vector_topk(
            self._table,
            query=list(vector),
            k=k,
            dim=self._dim,
            metric=self._metric,
            filter=equality_filter(filter),
            mode=self._mode,
        )
        out: list[tuple[Document, float]] = []
        for hit in hits:
            if self._text_schema is None:
                doc = Document(id=hit.id, page_content="", metadata={"id": hit.id})
            else:
                row = read_row(self._client, self._text_schema, hit.id)
                if row is None:
                    continue  # Stale vector hit; never return fabricated empty content.
                meta = dict(row.get("metadata", {}))
                meta.setdefault("id", hit.id)
                doc = Document(id=hit.id, page_content=row["text"], metadata=meta)
            out.append((doc, float(hit.score)))
        return out

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        if self._metric == "cosine":
            return lambda score: max(0.0, min(1.0, (score + 1.0) / 2.0))
        if self._metric == "dot":

            def sigmoid(score: float) -> float:
                if score >= 0:
                    return 1.0 / (1.0 + math.exp(-score))
                exp_score = math.exp(score)
                return exp_score / (1.0 + exp_score)

            return sigmoid
        return lambda score: 1.0 / (1.0 + max(0.0, score))
