"""
RAG Knowledge Base for FoodDetector
=====================================
Vector store  : Qdrant Cloud (Free Tier)
Embedding     : Google text-embedding-004 (768-dim)
Knowledge     : VietFood67 nutrition DB + Medical/Nutrition docs

Credentials required in .streamlit/secrets.toml:
  QDRANT_URL     = "https://your-cluster.qdrant.io:6333"
  QDRANT_API_KEY = "your-api-key"

The Gemini API key is read from st.session_state["gemini_api_key"]
(already entered by the user in the sidebar).
"""

import os
import uuid
import time
from typing import List, Optional

COLLECTION_NAME = "fooddetector_knowledge_v1"
EMBEDDING_DIM   = 768        # text-embedding-004 output dimension
EMBEDDING_MODEL = "models/text-embedding-004"
SCORE_THRESHOLD = 0.45       # minimum cosine similarity to return a chunk
EMBED_DELAY_SEC = 0.25       # throttle between embedding API calls during build


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _get_gemini_api_key() -> str:
    """Resolve Gemini API key: session_state → secrets → env."""
    try:
        import streamlit as st
        key = st.session_state.get("gemini_api_key", "")
        if key:
            return key
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "")


def _get_qdrant_client():
    """Build a QdrantClient from Streamlit secrets or env vars. Returns None if unconfigured."""
    try:
        from qdrant_client import QdrantClient

        try:
            import streamlit as st
            qdrant_url     = st.secrets.get("QDRANT_URL", os.environ.get("QDRANT_URL", ""))
            qdrant_api_key = st.secrets.get("QDRANT_API_KEY", os.environ.get("QDRANT_API_KEY", ""))
        except Exception:
            qdrant_url     = os.environ.get("QDRANT_URL", "")
            qdrant_api_key = os.environ.get("QDRANT_API_KEY", "")

        if not qdrant_url:
            return None

        return QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None, timeout=30)
    except ImportError:
        print("[RAG] qdrant-client not installed. Run: pip install qdrant-client")
        return None
    except Exception as e:
        print(f"[RAG] Failed to connect to Qdrant: {e}")
        return None


def _embed(text: str, task_type: str = "retrieval_document") -> Optional[List[float]]:
    """Embed a single text string with Google text-embedding-004."""
    import google.generativeai as genai

    api_key = _get_gemini_api_key()
    if not api_key:
        print("[RAG] No Gemini API key available for embedding.")
        return None

    genai.configure(api_key=api_key)
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type=task_type,
        )
        return result["embedding"]
    except Exception as e:
        print(f"[RAG] Embedding error: {e}")
        return None


# ---------------------------------------------------------------------------
# Document builders
# ---------------------------------------------------------------------------

def _build_food_documents() -> List[dict]:
    """
    Convert every entry in class_names.py into a rich text document
    suitable for semantic retrieval.
    """
    try:
        from class_names import class_names
    except ImportError:
        return []

    docs = []
    for food in class_names:
        name    = food.get("name", "Unknown")
        serving = food.get("serving_type", "N/A")
        n       = food.get("nutrition", {})

        calories   = n.get("Calories",  0)
        protein    = n.get("Protein",   0)
        fat        = n.get("Fat",       0)
        saturates  = n.get("Saturates", 0)
        sugar      = n.get("Sugar",     0)
        salt       = n.get("Salt",      0)

        # Derived values
        sodium_mg     = round(salt * 400, 0)          # Salt (g) → Sodium (mg)  (1g salt ≈ 400mg Na)
        density_score = round(protein / calories * 100, 1) if calories > 0 else 0.0

        # Traffic light labels
        def _fat_label(v):
            return "Thấp (Green)" if v < 3 else ("Trung bình (Yellow)" if v < 17.5 else "Cao (Red)")
        def _salt_label(v):
            return "Thấp (Green)" if v < 0.3 else ("Trung bình (Yellow)" if v < 1.5 else "Cao (Red)")
        def _sugar_label(v):
            return "Thấp (Green)" if v < 5 else ("Trung bình (Yellow)" if v < 22.5 else "Cao (Red)")
        def _sat_label(v):
            return "Thấp (Green)" if v < 1.5 else ("Trung bình (Yellow)" if v < 5 else "Cao (Red)")

        text = (
            f"Món ăn: {name}\n"
            f"Khẩu phần: {serving}\n"
            f"\n"
            f"Thông tin dinh dưỡng:\n"
            f"- Calories: {calories} kcal\n"
            f"- Protein: {protein} g\n"
            f"- Chất béo (Fat): {fat} g  [{_fat_label(fat)}]\n"
            f"- Chất béo bão hòa (Saturates): {saturates} g  [{_sat_label(saturates)}]\n"
            f"- Đường (Sugar): {sugar} g  [{_sugar_label(sugar)}]\n"
            f"- Muối (Salt): {salt} g  [{_salt_label(salt)}]\n"
            f"- Natri ước tính (Sodium): {sodium_mg:.0f} mg\n"
            f"- Chỉ số mật độ Protein (Protein Density): {density_score}% (protein/calorie ratio)\n"
        )

        docs.append({
            "id":   str(uuid.uuid5(uuid.NAMESPACE_DNS, f"food::{name}")),
            "text": text,
            "metadata": {
                "source":    "VietFood67",
                "category":  "food_nutrition",
                "food_name": name,
                "serving":   serving,
            },
        })

    return docs


def _load_knowledge_docs() -> List[dict]:
    """
    Load all .txt files from rag/knowledge_docs/, split into paragraphs,
    and return as a list of document dicts.
    """
    docs_dir = os.path.join(os.path.dirname(__file__), "knowledge_docs")
    if not os.path.isdir(docs_dir):
        return []

    docs = []
    for filename in sorted(os.listdir(docs_dir)):
        if not filename.endswith(".txt"):
            continue
        source_name = filename.replace(".txt", "")
        filepath    = os.path.join(docs_dir, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Split on double-newlines (paragraph level chunking)
        paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) >= 80]

        for i, para in enumerate(paragraphs):
            docs.append({
                "id":   str(uuid.uuid5(uuid.NAMESPACE_DNS, f"doc::{source_name}::{i}")),
                "text": para,
                "metadata": {
                    "source":    source_name,
                    "category":  "medical_nutrition",
                    "food_name": "",
                    "serving":   "",
                },
            })

    return docs


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FoodKnowledgeBase:
    """
    Manages the RAG vector store on Qdrant Cloud.

    Usage:
        kb = FoodKnowledgeBase()
        chunks = kb.retrieve("phở bò nhiều muối", top_k=3)
    """

    def __init__(self):
        self.client = _get_qdrant_client()
        self._ready = False
        self._ensure_ready()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _collection_populated(self) -> bool:
        if self.client is None:
            return False
        try:
            info = self.client.get_collection(COLLECTION_NAME)
            return (info.points_count or 0) > 0
        except Exception:
            return False

    def _ensure_ready(self):
        """Build the KB on first run; skip if already populated."""
        if self._collection_populated():
            self._ready = True
            return
        self._build()

    def _build(self):
        """
        Embed all documents and upsert to Qdrant.
        Called once per collection lifetime (or when force-rebuilding).
        """
        if self.client is None:
            print("[RAG] Qdrant client unavailable — skipping build.")
            return

        from qdrant_client.models import Distance, VectorParams, PointStruct

        # Re-create collection (idempotent)
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

        all_docs = _build_food_documents() + _load_knowledge_docs()
        print(f"[RAG] Building KB with {len(all_docs)} documents …")

        points = []
        for i, doc in enumerate(all_docs):
            vector = _embed(doc["text"], task_type="retrieval_document")
            if vector is None:
                print(f"[RAG]   Skipped doc {i} (embedding failed)")
                continue
            points.append(
                PointStruct(
                    id=doc["id"],
                    vector=vector,
                    payload={"text": doc["text"], **doc["metadata"]},
                )
            )
            # Throttle to avoid rate limits
            time.sleep(EMBED_DELAY_SEC)
            if (i + 1) % 10 == 0:
                print(f"[RAG]   Embedded {i + 1}/{len(all_docs)} …")

        # Upload in batches of 50
        batch_size = 50
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=points[i : i + batch_size],
            )

        print(f"[RAG] Build complete — {len(points)} vectors stored in Qdrant.")
        self._ready = True

    def rebuild(self):
        """Force a full rebuild of the knowledge base."""
        self._build()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve the top-k most relevant text chunks for a query.

        Args:
            query:  Natural language question or context string.
            top_k:  Number of chunks to return.

        Returns:
            List of text strings (empty if Qdrant unavailable or no match).
        """
        if not self._ready or self.client is None:
            return []

        query_vector = _embed(query, task_type="retrieval_query")
        if query_vector is None:
            return []

        try:
            results = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=SCORE_THRESHOLD,
                with_payload=True,
            )
            return [r.payload.get("text", "") for r in results if r.payload]
        except Exception as e:
            print(f"[RAG] Retrieval error: {e}")
            return []


# ---------------------------------------------------------------------------
# Streamlit cached singleton
# ---------------------------------------------------------------------------

def get_knowledge_base() -> FoodKnowledgeBase:
    """
    Return a cached FoodKnowledgeBase instance.
    Safe to call from multiple Streamlit reruns.
    Import and use this function inside Streamlit code.
    """
    try:
        import streamlit as st

        # Use session-level cache so the object persists across reruns
        # but is rebuilt if the page is refreshed (acceptable for cloud deploy)
        if "_rag_kb" not in st.session_state:
            st.session_state["_rag_kb"] = FoodKnowledgeBase()
        return st.session_state["_rag_kb"]
    except Exception:
        # Fallback for non-Streamlit contexts (e.g., build script)
        return FoodKnowledgeBase()
