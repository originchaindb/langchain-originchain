"""Read an existing vector index with a precomputed embedding."""

import argparse
import json
import math
import os

from langchain_core.embeddings import Embeddings
from originchain import OriginChain

from langchain_originchain import OriginChainVectorStore


class PrecomputedOnly(Embeddings):
    """Prevent accidental text embedding; this example takes a vector."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("Supply OC_QUERY_VECTOR; no text embedding is configured")

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError("Supply OC_QUERY_VECTOR; no text embedding is configured")


def search(client, table, vector):
    store = OriginChainVectorStore(
        client=client, table=table, dim=len(vector), embedding=PrecomputedOnly()
    )
    return [
        {"id": doc.metadata["id"], "text": doc.page_content}
        for doc in store.similarity_search_by_vector(vector, k=5)
    ]


def required(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Set {name}")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for name in ["OC_BASE_URL", "OC_TENANT", "OC_BEARER"]:
        required(name)
    table = required("OC_TABLE")
    vector = json.loads(required("OC_QUERY_VECTOR"))
    if (
        not isinstance(vector, list)
        or not vector
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in vector
        )
    ):
        raise ValueError("OC_QUERY_VECTOR must be a nonempty array of finite numbers")
    db_user = os.environ.get("OC_DB_USER")
    db_key = os.environ.get("OC_DB_KEY")
    if bool(db_user) != bool(db_key):
        raise ValueError("Set both OC_DB_USER and OC_DB_KEY, or neither")
    with OriginChain.from_env(db_user=db_user, db_key=db_key, timeout=30, max_retries=0) as client:
        print(json.dumps(search(client, table, vector), indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise SystemExit(
            "Query failed. Check connection, permissions and example inputs."
        ) from None
