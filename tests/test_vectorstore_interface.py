import httpx
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from originchain import OCAuthError, OCServerError

from langchain_originchain import OriginChainVectorStore


class Embedding(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def store(client, **kwargs):
    return OriginChainVectorStore(
        client=client, table="docs", dim=2, embedding=Embedding(), **kwargs
    )


def test_real_framework_round_trip_and_delete(api):
    client, state = api
    vector_store = store(client, text_schema="doc_rows")
    vector_store.add_documents([Document(id="one", page_content="hello", metadata={"tag": "x"})])
    documents = vector_store.as_retriever(search_kwargs={"filter": {"tag": "x"}}).invoke("hello")
    assert documents[0].page_content == "hello"
    assert documents[0].id == "one"
    assert vector_store.delete(["one"]) is True
    assert vector_store.similarity_search("hello") == []
    assert state.connection.execute("SELECT COUNT(*) FROM doc_rows").fetchone()[0] == 0
    assert vector_store.delete(["one"]) is True


@pytest.mark.parametrize(
    "row_id",
    ["o'brien", "x' OR '1'='1", "x'; DELETE FROM doc_rows; --", "a/b?c#d", "unicode-你好", "a\\b"],
)
def test_ids_are_values_not_sql_or_url_syntax(api, row_id):
    client, state = api
    vector_store = store(client, text_schema="doc_rows")
    vector_store.add_texts(["target", "untouched"], ids=[row_id, "safe"])
    assert vector_store.similarity_search("q")[0].page_content == "target"
    vector_store.delete([row_id])
    assert list(state.vectors) == ["safe"]
    assert vector_store.similarity_search("q")[0].page_content == "untouched"
    sql_requests = [body for _, path, body in state.requests if path.endswith("/sql")]
    assert all(row_id not in body["sql"] for body in sql_requests)
    assert any(body.get("params") == [row_id] for body in sql_requests)


def test_vector_only_delete(api):
    client, state = api
    vector_store = store(client)
    vector_store.add_texts(["hello"], ids=["a"])
    assert vector_store.similarity_search("hello")[0].id == "a"
    assert vector_store.delete(["a"])
    assert state.vectors == {}
    assert not any(request[1].endswith("/sql") for request in state.requests)


def test_delete_requires_explicit_ids(api):
    client, state = api
    vector_store = store(client)
    with pytest.raises(ValueError, match="explicit ids"):
        vector_store.delete()
    with pytest.raises(ValueError, match="explicit list"):
        vector_store.delete("abc")
    assert vector_store.delete([])
    assert state.requests == []


def test_translation_is_not_success(api):
    client, state = api
    vector_store = store(client, text_schema="doc_rows")
    vector_store.add_texts(["retained"], ids=["a"])
    state.translate_only = True
    with pytest.raises(RuntimeError, match="row still exists"):
        vector_store.delete(["a"])
    state.translate_only = False
    assert vector_store.delete(["a"])  # retry converges after partial vector deletion


@pytest.mark.parametrize("endpoint", ["/delete-bulk", "/sql"])
def test_delete_failure_propagates(api, endpoint):
    client, state = api
    vector_store = store(client, text_schema="doc_rows")
    vector_store.add_texts(["hello"], ids=["a"])
    state.failure = lambda req, body: (
        httpx.Response(500, json={"error": "outage"}) if req.url.path.endswith(endpoint) else None
    )
    with pytest.raises(OCServerError):
        vector_store.delete(["a"])
    assert state.connection.execute("SELECT COUNT(*) FROM doc_rows").fetchone()[0] == 1


def test_stale_hits_are_skipped(api):
    client, state = api
    vector_store = store(client, text_schema="doc_rows")
    vector_store.add_texts(["real"], ids=["a"])
    state.stale_hits = [{"id": "missing", "score": 1.0}]
    pairs = vector_store.similarity_search_with_score("q")
    assert [(doc.id, doc.page_content, score) for doc, score in pairs] == [("a", "real", 0.9)]


def test_hydration_failure_is_not_empty_content(api):
    client, state = api
    vector_store = store(client, text_schema="doc_rows")
    vector_store.add_texts(["real"], ids=["a"])
    state.failure = lambda req, body: (
        httpx.Response(401, json={"error": "denied"}) if req.url.path.endswith("/sql") else None
    )
    with pytest.raises(OCAuthError):
        vector_store.similarity_search("q")


@pytest.mark.parametrize(
    "schema", ["doc_rows; DELETE FROM docs", "doc_rows --", "a/b", '"doc_rows"', "", "foo.bar.baz"]
)
def test_schema_syntax_rejected_before_network(api, schema):
    client, state = api
    with pytest.raises(ValueError, match="identifier"):
        store(client, text_schema=schema)
    assert state.requests == []


@pytest.mark.parametrize(
    "value", [{"tag": {"$ne": "secret"}}, {"$or": [1, 2]}, {"tag": None}, {"n": float("nan")}]
)
def test_unsupported_filter_fails_closed(api, value):
    client, state = api
    with pytest.raises(ValueError, match="equality"):
        store(client).similarity_search("q", filter=value)
    assert state.requests == []


def test_batch_prevalidation_prevents_partial_invalid_write(api):
    client, state = api

    class BadEmbedding(Embedding):
        def embed_documents(self, texts):
            return [[1.0, 0.0], [float("nan"), 1.0]]

    vector_store = OriginChainVectorStore(
        client=client, table="docs", dim=2, embedding=BadEmbedding()
    )
    with pytest.raises(ValueError, match="finite"):
        vector_store.add_texts(["good", "invalid"])
    assert state.requests == []


@pytest.mark.parametrize("score, expected", [(-1000, 0), (1000, 1), (0, 0.5)])
def test_dot_relevance_does_not_overflow(api, score, expected):
    client, _ = api
    assert store(client, metric="dot")._select_relevance_score_fn()(score) == expected


def test_from_texts_preserves_mode(api):
    client, state = api
    vector_store = OriginChainVectorStore.from_texts(
        ["one"], Embedding(), client=client, mode="fast"
    )
    vector_store.similarity_search("q")
    assert state.requests[-1][2]["mode"] == "fast"
