"""
Vector embedder and Qdrant uploader for Al-Muhami Al-Zaki.

Handles:
- Text embedding using multilingual-e5-large
- Qdrant collection management
- Batch upsert of legal chunks
"""

from typing import Any, Dict, List, Optional
from uuid import uuid4

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from sentence_transformers import SentenceTransformer

from src.ingest.schemas import LegalChunkPayload
from src.utils.config import get_settings


class LegalEmbedder:
    """
    Embedder and uploader for legal document chunks.

    Uses multilingual-e5-large with proper prefixes:
    - "query: " for search queries
    - "passage: " for document chunks

    Example:
        embedder = LegalEmbedder()
        embedder.create_collection()
        embedder.embed_and_upload(chunks)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        """
        Initialize the embedder.

        Args:
            model_name: HuggingFace model ID (default from settings)
            qdrant_url: Qdrant Cloud URL (default from settings)
            qdrant_api_key: Qdrant API key (default from settings)
            collection_name: Qdrant collection name (default from settings)
        """
        settings = get_settings()

        self.model_name = model_name or settings.embedding_model
        self.collection_name = collection_name or settings.qdrant_collection_name

        # Initialize embedding model with GPU if available
        logger.info(f"Loading embedding model: {self.model_name}")

        # Auto-detect CUDA
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            logger.info(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("⚠️ GPU not available, using CPU")

        self.model = SentenceTransformer(self.model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # Initialize Qdrant client with extended timeout for cloud latency
        self.qdrant = QdrantClient(
            url=qdrant_url or settings.qdrant_url,
            api_key=qdrant_api_key or settings.qdrant_api_key,
            timeout=120,  # 2 minute timeout for large uploads
        )

        logger.info(
            f"Embedder initialized: model={self.model_name}, "
            f"dim={self.embedding_dim}, collection={self.collection_name}"
        )

    def create_collection(self, recreate: bool = False) -> None:
        """
        Create the Qdrant collection for legal documents.

        Uses cosine similarity (standard for E5 models).

        Args:
            recreate: If True, delete existing collection first
        """
        # Check if collection exists
        collections = self.qdrant.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if exists and recreate:
            logger.warning(f"Deleting existing collection: {self.collection_name}")
            self.qdrant.delete_collection(self.collection_name)
            exists = False

        if not exists:
            logger.info(f"Creating collection: {self.collection_name}")
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=self.embedding_dim,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )

            # Create payload indexes for filtering
            self._create_indexes()
        else:
            logger.info(f"Collection already exists: {self.collection_name}")

    def _create_indexes(self) -> None:
        """Create payload indexes for efficient filtering."""
        index_fields = [
            ("source_type", qdrant_models.PayloadSchemaType.KEYWORD),
            ("source_name", qdrant_models.PayloadSchemaType.KEYWORD),
            ("law_year", qdrant_models.PayloadSchemaType.INTEGER),
            ("article_number", qdrant_models.PayloadSchemaType.KEYWORD),
        ]

        for field_name, field_type in index_fields:
            try:
                self.qdrant.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                )
                logger.debug(f"Created index for: {field_name}")
            except Exception as e:
                logger.warning(f"Index creation failed for {field_name}: {e}")

    def embed_text(self, text: str, is_query: bool = False) -> List[float]:
        """
        Embed a single text.

        Uses E5 prefix convention:
        - "query: " for search queries
        - "passage: " for documents

        Args:
            text: Text to embed
            is_query: If True, use query prefix

        Returns:
            Embedding vector as list of floats
        """
        prefix = "query: " if is_query else "passage: "
        prefixed_text = prefix + text

        embedding = self.model.encode(
            prefixed_text,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    def embed_batch(
        self, texts: List[str], is_query: bool = False
    ) -> List[List[float]]:
        """
        Embed a batch of texts.

        Args:
            texts: List of texts to embed
            is_query: If True, use query prefix

        Returns:
            List of embedding vectors
        """
        prefix = "query: " if is_query else "passage: "
        prefixed_texts = [prefix + t for t in texts]

        embeddings = self.model.encode(
            prefixed_texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 10,
        )

        return [e.tolist() for e in embeddings]

    def embed_and_upload(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 20,  # Smaller batches for cloud stability
        max_retries: int = 3,
    ) -> int:
        """
        Embed chunks and upload to Qdrant.

        Args:
            chunks: List of chunk dictionaries with 'text_anonymized' field
            batch_size: Number of chunks to upload per request (smaller = more stable)
            max_retries: Number of retry attempts per batch

        Returns:
            Number of chunks uploaded
        """
        if not chunks:
            logger.warning("No chunks to upload")
            return 0

        total_uploaded = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            # Extract anonymized text for embedding
            texts = [c.get("text_anonymized", c.get("text", "")) for c in batch]

            # Embed batch
            embeddings = self.embed_batch(texts, is_query=False)

            # Prepare points for Qdrant
            points = []
            for chunk, embedding in zip(batch, embeddings):
                # Validate payload
                try:
                    payload = LegalChunkPayload(
                        text=chunk.get("text", ""),
                        text_anonymized=chunk.get(
                            "text_anonymized", chunk.get("text", "")
                        ),
                        source_name=chunk.get("source_name", "Unknown"),
                        source_type=chunk.get("source_type", "law"),
                        law_number=chunk.get("law_number"),
                        law_year=chunk.get("law_year", 1900),
                        article_number=chunk.get("article_number"),
                        chapter=chunk.get("chapter"),
                        chunk_index=chunk.get("chunk_index", 0),
                        total_chunks=chunk.get("total_chunks", 1),
                        is_anonymized=chunk.get("is_anonymized", True),
                    )
                except Exception as e:
                    logger.error(f"Invalid chunk payload: {e}")
                    continue

                point = qdrant_models.PointStruct(
                    id=str(uuid4()),
                    vector=embedding,
                    payload=payload.model_dump(mode="json"),
                )
                points.append(point)

            # Upsert to Qdrant with retry logic
            if points:
                for attempt in range(max_retries):
                    try:
                        self.qdrant.upsert(
                            collection_name=self.collection_name,
                            points=points,
                        )
                        total_uploaded += len(points)
                        logger.info(
                            f"Uploaded batch {i // batch_size + 1}: "
                            f"{len(points)} points (total: {total_uploaded})"
                        )
                        break  # Success, exit retry loop
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.warning(
                                f"Batch upload failed (attempt {attempt + 1}/{max_retries}): {e}"
                            )
                            import time

                            time.sleep(2**attempt)  # Exponential backoff
                        else:
                            logger.error(
                                f"Batch upload failed after {max_retries} attempts: {e}"
                            )
                            raise

        logger.info(f"Upload complete: {total_uploaded} chunks")
        return total_uploaded

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant legal documents.

        Args:
            query: Search query in Arabic
            top_k: Number of results to return
            filters: Optional Qdrant filter conditions

        Returns:
            List of matching documents with scores
        """
        # Embed query with query prefix
        query_embedding = self.embed_text(query, is_query=True)

        # Build filter if provided
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    conditions.append(
                        qdrant_models.FieldCondition(
                            key=key,
                            match=qdrant_models.MatchAny(any=value),
                        )
                    )
                else:
                    conditions.append(
                        qdrant_models.FieldCondition(
                            key=key,
                            match=qdrant_models.MatchValue(value=value),
                        )
                    )
            qdrant_filter = qdrant_models.Filter(must=conditions)

        # Search using query_points (qdrant-client >= 1.10)
        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        # Format results
        documents = []
        for point in results.points:
            doc = {
                "id": point.id,
                "score": point.score,
                **point.payload,
            }
            documents.append(doc)

        return documents
