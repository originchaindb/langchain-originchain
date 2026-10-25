# LangChain for OriginChainDB

Store and retrieve LangChain documents using an OriginChainDB vector table and
an optional table for document text and metadata.

**Status:** `0.1.0rc1` preview. **License:** Proprietary; the existing license
declaration is unchanged. No open-source license is granted by this release.

## Install

Requires Python 3.10+, LangChain Core 0.3 or 1.x, and the OriginChain Python SDK
0.10.0rc1 or newer (below 1.0). The commands below install a pinned development version of the core SDK and
the existing adapter preview. Git must be installed; the adapter is not on PyPI.

```bash
python -m pip install "originchain @ git+https://github.com/originchaindb/originchaindb-python.git@944cd410ffb11110c00ce925c0c609e11862916b"
python -m pip install https://github.com/originchaindb/langchain-originchain/releases/download/v0.1.0-rc.1/langchain_originchain-0.1.0rc1-py3-none-any.whl
```

The example uses OpenAI embeddings. Install that provider separately:

```bash
python -m pip install langchain-openai
```

The core SDK remains imported as `originchain`; its maintained repository is
`originchaindb-python`. This source pin is `0.11.0.dev0`, which is unreleased.
The adapter's `0.1.0rc1` release and proprietary license are unchanged. See the
[SDK support matrix](https://github.com/originchaindb/.github/blob/main/SDK-SUPPORT.md) for the distinction between source verification and
published packages.

## Add, search and delete documents

Set `OC_BASE_URL`, `OC_TENANT` and `OC_BEARER` to your database connection values,
and `OPENAI_API_KEY` for the embedding provider. Load credentials through your
environment or secrets manager.

This example creates the content schema, stores two documents, searches with a
metadata filter, and deletes the example documents. Embedding requests go to
OpenAI. Use a separate table for each embedding model and dimension.

```python
from langchain_openai import OpenAIEmbeddings
from langchain_originchain import OriginChainVectorStore
from originchain import OriginChain

with OriginChain.from_env() as db:
    db.schemas.register("""
namespace = "sdk_examples"
table = "langchain_documents"
primary_key = ["id"]
[[columns]]
name = "id"
ty = "str"
required = true
[[columns]]
name = "text"
ty = "str"
[[columns]]
name = "metadata"
ty = "json"
""")

    store = OriginChainVectorStore(
        client=db,
        table="langchain_vectors",
        text_schema="sdk_examples.langchain_documents",
        dim=1536,
        embedding=OpenAIEmbeddings(
            model="text-embedding-3-small", dimensions=1536
        ),
    )
    ids = store.add_texts(
        ["Returns are accepted within 30 days.", "Express delivery takes two days."],
        ids=["example-returns", "example-delivery"],
        metadatas=[{"topic": "returns"}, {"topic": "delivery"}],
    )
    documents = store.similarity_search(
        "When can I return an order?", k=2, filter={"topic": "returns"}
    )
    for document in documents:
        print(document.id, document.page_content, document.metadata)

    retriever = store.as_retriever(search_kwargs={"k": 2})
    print(retriever.invoke("How long does delivery take?"))
    store.delete(ids=ids)
```

`table` names the vector table; `text_schema` names the separate content schema.
Schema registration belongs in application setup. When reusing a schema, its
columns and primary key must match the definition above. Tables and schema names
must be ASCII identifiers, optionally qualified as `namespace.table`.

## Supported operations

| Operation | Behavior |
| --- | --- |
| `add_texts`, `add_documents` | Embed and store content. Supply stable IDs to retry an interrupted write. |
| `from_texts` | Construct and populate a store; requires `client=`. Pass `dim=` to avoid an extra embedding call to infer it. |
| `similarity_search` | Return hydrated LangChain `Document` objects. |
| `similarity_search_by_vector` | Search using an existing vector of exactly `dim` finite numbers. |
| `similarity_search_with_score` | Return documents with raw server scores. |
| `as_retriever` | Use similarity search through LangChain's retriever interface. MMR is not implemented. |
| `delete(ids=[...])` | Delete the named vectors and their content rows. An empty list is a no-op; omitting IDs raises `ValueError`. |

Filters are flat equality dictionaries: `{"topic": "returns", "published": True}`
means both conditions must match. Values can be strings, finite numbers or
booleans. Nested operators, lists and `None` are rejected.

`metric` accepts `cosine` (default), `dot` or `l2`. LangChain relevance-score
search maps these to `[0, 1]` using `(score + 1) / 2`, a sigmoid, or
`1 / (1 + distance)` respectively. `mode` optionally selects `fast` or
`high_recall` on the server.

## Storage, failures and client ownership

The caller owns the synchronous `OriginChain` client. Keep it open while using
the store or retriever; the context manager in the example closes it afterward.

Without `text_schema`, the adapter stores only vectors. Search then returns
ID-only documents with empty text; your application must resolve the content.
With `text_schema`, a hit whose content row is missing is skipped, so a search
can return fewer than `k` documents.

Writes store a content row before its vector. Deletes remove a vector before its
row, then read the row back to confirm absence. These requests are **not one
transaction**: a failure can leave partial state. Errors propagate to the caller.
Retry with the same IDs, and serialize concurrent writes and deletes for an ID.
Repeated deletion of an absent ID is safe.

The server must execute SQL DELETE and support positional SQL parameters.
Authentication, transport and server errors come from the core SDK; malformed
inputs raise `ValueError`. A server that acknowledges deletion but retains the
row causes `RuntimeError` rather than a successful result.

## Development

After installing the core SDK above:

```bash
python -m pip install -e ".[dev]" build
python -m pytest -q tests
python -m ruff check src tests
python -m build
```

Tests use real LangChain types and mocked HTTP. Separate live database results
and package checksums are attached to the [GitHub release](https://github.com/originchaindb/langchain-originchain/releases/tag/v0.1.0-rc.1).
