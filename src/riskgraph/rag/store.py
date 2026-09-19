"""Dense policy retrieval (SPEC §9.2, dense baseline only): bge-small-en-v1.5 embeddings on CPU
in a Weaviate collection with bring-your-own vectors. Hybrid search and re-ranking arrive in
phase 05a.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

import numpy as np
import numpy.typing as npt
import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.util import generate_uuid5

from riskgraph.rag.chunking import Chunk

MODEL = "BAAI/bge-small-en-v1.5"
# bge v1.5 retrieval instruction, applied to queries only (model card).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
COLLECTION = "PolicyChunk"
FIELDS = ("doc_id", "section_id", "section_path", "text")
TOP_K = 5
_encode = threading.Lock()  # one encoder shared by parallel evaluation threads


@lru_cache(maxsize=1)
def _model() -> Any:
    from sentence_transformers import SentenceTransformer  # slow import, only when embedding

    try:  # the cached copy, without a network round trip on every load
        return SentenceTransformer(MODEL, device="cpu", local_files_only=True)
    except OSError:  # first use: download it
        return SentenceTransformer(MODEL, device="cpu")


def embed(texts: Sequence[str], query: bool = False) -> npt.NDArray[np.float32]:
    """Unit-length embeddings (cosine = dot product)."""
    items = [QUERY_PREFIX + t for t in texts] if query else list(texts)
    with _encode:
        out: npt.NDArray[np.float32] = _model().encode(
            items, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )
    return out


def connect() -> weaviate.WeaviateClient:
    return weaviate.connect_to_local(host=os.environ.get("WEAVIATE_HOST", "localhost"))


def build_index(client: weaviate.WeaviateClient, chunks: Sequence[Chunk]) -> int:
    """Recreate the collection and load every chunk with its vector. Returns the object count."""
    if client.collections.exists(COLLECTION):
        client.collections.delete(COLLECTION)
    col = client.collections.create(
        COLLECTION,
        vector_config=Configure.Vectors.self_provided(),
        properties=[Property(name=f, data_type=DataType.TEXT) for f in FIELDS],
    )
    vectors = embed([c["text"] for c in chunks])
    with col.batch.fixed_size(batch_size=200) as batch:
        for i, (c, v) in enumerate(zip(chunks, vectors, strict=True)):
            uid = generate_uuid5(f"{c['doc_id']}|{c['section_id']}|{i}")
            batch.add_object(properties={f: c[f] for f in FIELDS}, vector=v.tolist(), uuid=uid)
    if col.batch.failed_objects:
        raise RuntimeError(f"{len(col.batch.failed_objects)} chunks failed to load")
    return len(col)


class Retriever:
    """search_policy backend: top-k chunks by cosine similarity, optional doc_id filter."""

    def __init__(self, client: weaviate.WeaviateClient, k: int = TOP_K) -> None:
        self.col, self.k = client.collections.get(COLLECTION), k

    def __call__(self, query: str, filters: Mapping[str, str]) -> list[dict[str, Any]]:
        unknown = set(filters) - {"doc_id"}
        if unknown:
            raise ValueError(f"unsupported filters {sorted(unknown)}; only doc_id")
        where = Filter.by_property("doc_id").equal(filters["doc_id"]) if filters else None
        res = self.col.query.near_vector(
            near_vector=embed([query], query=True)[0].tolist(),
            limit=self.k,
            filters=where,
            return_metadata=MetadataQuery(distance=True),
        )
        return [
            {f: str(o.properties[f]) for f in FIELDS}
            | {"score": round(1 - float(o.metadata.distance or 0), 4)}
            for o in res.objects
        ]
