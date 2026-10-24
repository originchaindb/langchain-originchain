"""LangChain VectorStore adapter for OriginChain.

Wraps the official ``originchain`` Python SDK so OriginChain can be used
as a drop-in vector store in any LangChain pipeline.

Quick start::

    from langchain_originchain import OriginChainVectorStore
    from langchain_openai import OpenAIEmbeddings
    from originchain import OriginChain

    db = OriginChain.from_env()
    store = OriginChainVectorStore(
        client=db,
        table="docs",
        dim=1536,
        embedding=OpenAIEmbeddings(),
    )
    store.add_texts(["hello world", "vector store"])
    hits = store.similarity_search("hi", k=2)
"""

from .vectorstore import OriginChainVectorStore

__all__ = ["OriginChainVectorStore"]
__version__ = "0.1.0rc1"
