# langchain-originchain

LangChain VectorStore adapter for [OriginChainDB](https://originchaindb.com).

This is a **Proprietary** GitHub preview. The existing license declaration is
preserved; this release does not add an open-source license grant.

Install the tested preview directly from its GitHub release assets. These
packages have not been published to PyPI:

```bash
python -m pip install "https://github.com/originchaindb/originchain-python/releases/download/v0.10.0-rc.3/originchain-0.10.0rc3-py3-none-any.whl#sha256=1d6456e7bc51e0dfb4b0d03ef4e0713f2cd3c902e4579a7fb956e5fb6e59c647"
python -m pip install https://github.com/originchaindb/langchain-originchain/releases/download/v0.1.0-rc.1/langchain_originchain-0.1.0rc1-py3-none-any.whl
```

```python
from langchain_originchain import OriginChainVectorStore
from langchain_openai import OpenAIEmbeddings
from originchain import OriginChain

# OpenAIEmbeddings is an optional, separately installed provider integration.
with OriginChain.from_env() as db:
    store = OriginChainVectorStore(
        client=db,
        table="docs",
        dim=1536,
        embedding=OpenAIEmbeddings(),
        text_schema="doc_rows",
    )
    store.add_texts(["hello world"], ids=["greeting"], metadatas=[{"tag": "public"}])
    documents = store.similarity_search("hello", k=2, filter={"tag": "public"})
    store.delete(ids=["greeting"])
```

Supports LangChain Core 0.3 and 1.x. `add_documents`, retriever invocation,
`from_texts`, text/vector search and relevance scores use the framework's normal
VectorStore interface. The supplied embedding implementation owns model calls.
Metadata filters are flat scalar equalities joined with AND; operator dictionaries
are rejected. `delete` requires explicit IDs and also works without text storage.
An empty ID list is a successful no-op. Relevance scores map cosine to `[0,1]`,
dot products through an overflow-safe sigmoid, and L2 through `1/(1+distance)`.

## Correctness and compatibility

Python 3.10 or newer and `originchain>=0.10.0rc1,<1` are required. The server must
support positional SQL parameters and executed SQL DELETE (the deployed test
database was checked on 24 October 2026). Identifiers accept ASCII names with an
optional namespace; IDs and document references are sent as SQL parameters.

Create the content table before ingesting, using SQL or the schema API:

```sql
CREATE TABLE doc_rows (
  id TEXT PRIMARY KEY,
  text TEXT,
  metadata JSON,
  ref_doc_id TEXT
)
```

`ref_doc_id` is used by LlamaIndex. LangChain does not write it. Content storage
is optional, but without it the application/framework must retain the documents
in its own docstore. Vectors are stored in the separate configured vector table.

Writes and deletions touch separate vector and row endpoints; they are not one
transaction. Supply stable IDs and retry after failures. Errors propagate to the
caller. A deletion removes the vector first, deletes the row by its primary key,
and confirms the row is absent. An old server that only translates SQL cannot
report a successful deletion while retaining the row. Concurrent writers of the
same ID must be serialized by the application.

Search skips vector hits whose content rows no longer exist, so it can return
fewer than `k` results. Authentication, schema, transport, and service errors
raise; they never become fabricated empty documents. The caller owns and closes
the OriginChain client.

## Verification

```powershell
python -m pip install -e ".[dev]" build twine
python -m pytest -q tests
python -m ruff check src tests
python -m build
python -m twine check dist/*
```

The regression tests use installed framework classes and the real OriginChain
HTTP client with `httpx.MockTransport`. They verify the wire payloads, parameter
binding, vector plus row deletion, partial failures, missing hits and unsupported
filters. These tests are distinct from live database acceptance. Credentials are
never part of the package. The existing Proprietary license is preserved.

The public acceptance workflow tests Python 3.10, 3.12 and 3.14 on Linux
and Python 3.12 on Windows, including the minimum framework version and a fresh
wheel installation. Release assets include separate mock-test and live-database
verification receipts. Live checks use a separate disposable namespace.
