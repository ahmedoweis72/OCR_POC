import os
from typing import List, Dict, Any, Optional, Literal, Tuple
import hashlib

import requests
from pymilvus import MilvusClient, FieldSchema, DataType, Function, FunctionType, CollectionSchema, Collection


class MilvusConfig:
    host: str = "62.193.95.102"
    port: str = "19530"


EmbedKind = Literal["query", "document"]


class EmbeddingConfig:
    """Embedding model configuration"""
    model_name: str = os.getenv("EMBEDDING_MODEL", "snowflake-arctic-embed2:latest")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/").rstrip("/")
    # Optional legacy override (full embed URL). If unset, embeddings.py uses base + /api/embed.
    ollama_url: str = os.getenv("OLLAMA_URL", "")
    dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
    # Snowflake Arctic: prefix queries with "query: "; documents have no prefix.
    query_embed_prefix: str = os.getenv("EMBED_QUERY_PREFIX", "query: ")
    document_embed_prefix: str = os.getenv("EMBED_DOCUMENT_PREFIX", "")


class CollectionConfig:
    """Default collection configuration"""
    name: str = "semantic_jupiter_test"
    index_type: str = "HNSW"
    metric_type: str = "COSINE"
    # HNSW parameters for high accuracy
    hnsw_m: int = 32  # Max connections per node
    hnsw_ef_construction: int = 500  # Build depth
    # Hybrid search: candidates per branch before weighted rerank (capped)
    hybrid_branch_limit_multiplier: int = 4
    hybrid_branch_limit_max: int = 256
    # WeightedRanker: one weight per AnnSearchRequest (dense, then BM25 sparse)
    hybrid_dense_weight: float = 0.3
    hybrid_sparse_weight: float = 0.7


def connect(host: str = None, port: str = None) -> MilvusClient | None:
    print("Connecting...")
    try:
        # connections.connect(alias="default", host=host or MilvusConfig.host, port=port or MilvusConfig.port)
        client = MilvusClient(f"http://{MilvusConfig.host}:{MilvusConfig.port}")

        print("Connected to Milvus.")
        return client
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        return None


def disconnect(client: MilvusClient):
    try:
        print("Disconnect...")
        client.close()
    except Exception as e:
        print(f"Failed to disconnect from Milvus: {e}")


def list_collections(client: MilvusClient) -> List[str]:
    return client.list_collections()


def collection_exists(client: MilvusClient, collection_name: str) -> bool:
    return client.has_collection(collection_name)


def create_collection(client: MilvusClient, collection_name: str, dim: int = None) -> bool:
    dim = dim or EmbeddingConfig.dimension

    if collection_exists(client, collection_name):
        print(f"⚠️ Collection '{collection_name}' already exists")
        return False

    fields = [
        # Changed: auto_id=False so we can upsert by ID
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=20000, enable_analyzer=True),
        FieldSchema(name="section_name", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="sparse_embedding", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="metadata", dtype=DataType.JSON, nullable=True),
    ]

    description = "Hybrid search collection with section-based chunking"

    bm25_function = Function(name="content_bm25", function_type=FunctionType.BM25, input_field_names=["content"],
                             output_field_names=["sparse_embedding"], )
    schema = CollectionSchema(fields=fields, description=description, functions=[bm25_function])
    client.create_collection(collection_name, schema=schema)
    index_params = client.prepare_index_params()

    # Dense vector index (HNSW)
    index_params.add_index(
        field_name="embedding",
        index_type=CollectionConfig.index_type,  # "HNSW"
        metric_type=CollectionConfig.metric_type,  # "COSINE"
        params={
            "M": CollectionConfig.hnsw_m,  # 32
            "efConstruction": CollectionConfig.hnsw_ef_construction,  # 500
        }
    )

    # Sparse vector index (for BM25 hybrid search)
    index_params.add_index(
        field_name="sparse_embedding",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25"
    )

    client.create_index(collection_name, index_params)
    print(f"✅ Collection '{collection_name}' created successfully")
    return True

def drop_collection(client: MilvusClient, collection_name: str) -> bool:
    if not collection_exists(client, collection_name):
        print(f"⚠️ Collection '{collection_name}' does not exist")
        return False

    client.drop_collection(collection_name)
    print(f"✅ Collection '{collection_name}' dropped successfully")
    return True

def add_document(client: MilvusClient, content: str, source: str = "unknown", metadata: Optional[Dict[str, Any]] = None,
                 collection_name: str = None, section_name: str = "", chunk_index: int = 0) -> bool:
    collection_name = collection_name or CollectionConfig.name
    try:
        embedding = get_document_embedding(content)
        doc_id = generate_doc_id(source, content)

        data = [{
            "id": doc_id,
            "source": source,
            "content": content,
            "section_name": section_name,
            "chunk_index": chunk_index,
            "embedding": embedding,
            "metadata": metadata or {}
        }]

        res = client.insert(collection_name=collection_name, data=data)
        print(res)
        client.flush(collection_name)
        print(f"✅ Document added from '{source}'")
        return True
    except Exception as e:
        print(f"❌ Error adding document: {e}")
        return False


def get_document_embedding(text: str, model: str = None) -> List[float]:
    return _ollama_embed(_apply_embed_prefix(text, "document"), model)


def _ollama_embed(text: str, model: str = None, timeout_s: float = 120.0) -> List[float]:
    model = model or EmbeddingConfig.model_name
    base = ollama_api_base()

    attempts: List[Tuple[str, dict]] = [(f"{base}/api/embed", {"model": model, "input": text}),
                                        (f"{base}/api/embeddings", {"model": model, "prompt": text}), ]

    last_response: Optional[requests.Response] = None
    last_error: Optional[Exception] = None

    for url, payload in attempts:
        try:
            response = requests.post(url, json=payload, timeout=timeout_s)
            last_response = response
            if response.status_code == 404:
                continue
            response.raise_for_status()
            embedding = _parse_embed_response(response.json())
            if embedding is None:
                raise ValueError(f"Ollama embed response missing vector from {url}")
            return embedding
        except requests.HTTPError as exc:
            last_error = exc
            if exc.response is not None and exc.response.status_code == 404:
                continue
            print(f"HTTP Error generating embedding ({url}): {exc}")
            if exc.response is not None:
                print(f"Status Code: {exc.response.status_code}")
                print(f"Response Body: {exc.response.text}")
            raise
        except requests.RequestException as exc:
            last_error = exc
            print(f"Error generating embedding ({url}): {exc}")
            raise

    detail = ""
    if last_response is not None:
        detail = f" status={last_response.status_code} body={last_response.text[:300]}"
    msg = (f"Ollama embedding failed: neither {base}/api/embed nor "
           f"{base}/api/embeddings is available.{detail}")
    if last_error:
        raise RuntimeError(msg) from last_error
    raise RuntimeError(msg)


def ollama_api_base() -> str:
    """Resolve Ollama server base URL (no /api/embed path)."""
    url = (EmbeddingConfig.ollama_url or "").strip().rstrip("/")
    if url:
        for suffix in ("/api/embeddings", "/api/embed"):
            if url.endswith(suffix):
                return url[: -len(suffix)]
        return url

    explicit = (EmbeddingConfig.ollama_base_url or "").strip().rstrip("/")
    if explicit:
        return explicit

    raise RuntimeError("Ollama not configured: set OLLAMA_BASE_URL (e.g. https://host:11434) "
                       "or OLLAMA_URL")


def _apply_embed_prefix(text: str, kind: EmbedKind) -> str:
    text = text.strip()
    if kind == "query":
        prefix = EmbeddingConfig.query_embed_prefix
    else:
        prefix = EmbeddingConfig.document_embed_prefix
    if not prefix:
        return text
    if text.lower().startswith(prefix.strip().lower()):
        return text
    return f"{prefix}{text}"


def _parse_embed_response(data: dict) -> Optional[List[float]]:
    embedding = data.get("embedding")
    if embedding is not None:
        return embedding
    embeddings = data.get("embeddings")
    if embeddings:
        first = embeddings[0]
        if isinstance(first, list):
            return first
    return None

def generate_doc_id(source: str, content: str) -> str:
    """Generate a deterministic ID from source + content."""
    raw = f"{source}::{content}"
    return hashlib.sha256(raw.encode()).hexdigest()

def upsert_document(client: MilvusClient, content: str,doc_id: str, source: str = "unknown",metadata: Optional[Dict[str, Any]] = None, collection_name: str = None, section_name: str = "",chunk_index: int = 0, ) -> bool:
    """
    Insert or update a document.
    If a document with the same ID exists, it will be fully replaced.
    """
    collection_name = collection_name or CollectionConfig.name
    try:
        embedding = get_document_embedding(content)
        doc_id = doc_id or generate_doc_id(source, content)
        print(f"Upserting document (id={doc_id[:12]}...) from '{source}'...")
        data = [{"id": doc_id, "source": source, "content": content, "section_name": section_name,
            "chunk_index": chunk_index, "embedding": embedding, "metadata": metadata or {}, }]

        res = client.upsert(collection_name=collection_name, data=data)
        print(f"✅ Document upserted (id={doc_id}...) from '{source}'")
        return True
    except Exception as e:
        print(f"❌ Error upserting document: {e}")
        return False

def get_document(client: MilvusClient, doc_id: str, collection_name: str = None) -> Optional[Dict[str, Any]]:
    collection_name = collection_name or CollectionConfig.name
    try:
        results = client.query(
            collection_name=collection_name,
            filter=f'id == "{doc_id}"',
            output_fields=["source", "content", "section_name", "chunk_index", "metadata"],
            limit=1,
        )
        return results[0] if results else None
    except Exception as e:
        print(f"❌ Error fetching document: {e}")
        return None

def update_document(client: MilvusClient, doc_id: str, content: str = None, source: str = None,
        metadata: Optional[Dict[str, Any]] = None, collection_name: str = None, ) -> bool:
    """
    Update specific fields of an existing document.
    You MUST provide doc_id. Only provided fields are changed.
    """
    collection_name = collection_name or CollectionConfig.name
    try:
        # Fetch existing document to merge fields
        results = client.query(
            collection_name=collection_name,
            filter=f'id == "{doc_id}"',
            output_fields=["source", "content", "section_name", "chunk_index", "metadata","embedding"],
            limit=1,
        )
        print(results)
        if not results:
            print(f"❌ Document with id='{doc_id}' not found")
            return False

        existing = results[0]

        # Build updated record
        new_content = content if content is not None else existing["content"]
        print(new_content)
        new_source = source if source is not None else existing["source"]
        new_metadata = metadata if metadata is not None else existing.get("metadata", {})

        # Re-embed if content changed
        if content is not None:
            embedding = get_document_embedding(new_content)
        else:
            embedding = existing["embedding"]

        data = [{
            "id": doc_id,
            "source": new_source,
            "content": new_content,
            "section_name": existing.get("section_name", ""),
            "chunk_index": existing.get("chunk_index", 0),
            "embedding": embedding,
            "metadata": new_metadata,
        }]

        res = client.upsert(collection_name=collection_name, data=data, keyArgs={"partial_update" : True})
        print(f"✅ Document updated (id={doc_id[:12]}...)")
        return True
    except Exception as e:
        print(f"❌ Error updating document: {e}")
        return False

def get_document_by_source(client: MilvusClient,source: str, collection_name: str = None) -> Tuple[bool, str]:
    collection_name = collection_name or CollectionConfig.name
    if not collection_exists(client, collection_name):
        return False, f"Collection '{collection_name}' does not exist"
    client.load_collection(collection_name=collection_name)
    expr = f'source == "{source}"'
    results = client.query(collection_name=collection_name, filter=expr,output_fields=["source", "content"],limit=1)
    if not results:
        return False, f"unregistered"
    return True, f"registered{results}"