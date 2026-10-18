"""
Cambridge Support RAG Engine
Handles document loading, vector embeddings (Azure OpenAI / OpenAI),
cosine similarity retrieval, and LLM answer generation with citations.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

# Try importing SDKs gracefully
try:
    from google import genai
    from google.genai import types as genai_types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

try:
    from openai import AzureOpenAI, OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Load environment variables
load_dotenv()


class CambridgeRAGEngine:
    """
    RAG Engine that coordinates document indexing, semantic embedding retrieval,
    and contextual generation for Cambridge student support.
    Supports Gemini, Azure OpenAI, and OpenAI.
    """

    def __init__(self, data_path: Optional[str] = None):
        if data_path is None:
            base_dir = Path(__file__).resolve().parent
            self.data_path = base_dir / "data" / "cambridge_knowledge_base.json"
        else:
            self.data_path = Path(data_path)

        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.provider = os.getenv("MODEL_PROVIDER", "gemini").lower()

        # Initialize clients & model configs
        self._init_clients()
        self.load_and_index_documents()

    def _init_clients(self) -> None:
        """Initializes Gemini, OpenAI, or AzureOpenAI client based on environment variables."""
        self.client = None
        self.chat_model = None
        self.embed_model = None
        self.is_configured = False

        if self.provider == "gemini":
            api_key = os.getenv("GEMINI_KEY") or os.getenv("GEMINI_API_KEY", "")
            # In google-genai SDK, use gemini-2.5-flash or gemini-2.0-flash
            self.chat_model = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
            self.embed_model = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")

            if api_key and not api_key.startswith("your_") and HAS_GENAI:
                try:
                    self.client = genai.Client(api_key=api_key)
                    self.is_configured = True
                except Exception as e:
                    print(f"Warning: Failed to initialize Gemini client: {e}")

        elif self.provider == "azure" and HAS_OPENAI:
            api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
            self.chat_model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")
            self.embed_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

            if api_key and endpoint and not api_key.startswith("your_"):
                try:
                    self.client = AzureOpenAI(
                        api_key=api_key,
                        azure_endpoint=endpoint,
                        api_version=api_version,
                    )
                    self.is_configured = True
                except Exception as e:
                    print(f"Warning: Failed to initialize Azure OpenAI client: {e}")
        elif HAS_OPENAI:
            api_key = os.getenv("OPENAI_API_KEY", "")
            self.chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
            self.embed_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

            if api_key and not api_key.startswith("sk-your_"):
                try:
                    self.client = OpenAI(api_key=api_key)
                    self.is_configured = True
                except Exception as e:
                    print(f"Warning: Failed to initialize OpenAI client: {e}")

    def load_and_index_documents(self) -> None:
        """Loads JSON knowledge base and precomputes or caches embeddings."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Knowledge base file not found at: {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        # Prepare composite text for embedding
        for doc in self.documents:
            tags_str = ", ".join(doc.get("tags", []))
            doc["search_text"] = (
                f"Title: {doc.get('title', '')}\n"
                f"Category: {doc.get('category', '')}\n"
                f"Tags: {tags_str}\n"
                f"Content: {doc.get('content', '')}"
            )

        # Compute embeddings if client is configured
        if self.is_configured and self.client:
            try:
                texts = [d["search_text"] for d in self.documents]
                if self.provider == "gemini":
                    # Google GenAI embeddings
                    vectors = []
                    for t in texts:
                        res = self.client.models.embed_content(
                            model=self.embed_model,
                            contents=t,
                        )
                        vectors.append(res.embeddings[0].values)
                    self.embeddings = np.array(vectors, dtype=np.float32)
                else:
                    response = self.client.embeddings.create(
                        input=texts,
                        model=self.embed_model,
                    )
                    vectors = [item.embedding for item in response.data]
                    self.embeddings = np.array(vectors, dtype=np.float32)

                # Normalize for fast dot-product cosine similarity
                norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                self.embeddings = self.embeddings / norms
            except Exception as e:
                print(f"Embedding computation error (falling back to keyword matching): {e}")
                self.embeddings = None

    def _get_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Generates a normalized embedding vector for the user query."""
        if not self.is_configured or not self.client:
            return None
        try:
            if self.provider == "gemini":
                res = self.client.models.embed_content(
                    model=self.embed_model,
                    contents=query,
                )
                vec = np.array(res.embeddings[0].values, dtype=np.float32)
            else:
                response = self.client.embeddings.create(
                    input=[query],
                    model=self.embed_model,
                )
                vec = np.array(response.data[0].embedding, dtype=np.float32)

            norm = np.linalg.norm(vec)
            return vec / (norm if norm > 0 else 1.0)
        except Exception as e:
            print(f"Query embedding generation failed: {e}")
            return None

    def _lexical_search(self, query: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """Fallback keyword-based scoring when embeddings are unavailable."""
        query_words = set(re.findall(r"\w+", query.lower()))
        scores = []

        for doc in self.documents:
            text = (
                doc.get("title", "") + " " +
                doc.get("content", "") + " " +
                " ".join(doc.get("tags", []))
            ).lower()
            doc_words = set(re.findall(r"\w+", text))
            if not doc_words or not query_words:
                score = 0.0
            else:
                intersection = query_words.intersection(doc_words)
                # Boost if in title or tags
                title_words = set(re.findall(r"\w+", doc.get("title", "").lower()))
                title_matches = query_words.intersection(title_words)
                score = (len(intersection) + len(title_matches) * 2.0) / (len(query_words) + 1.0)
            scores.append((doc, float(score)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def retrieve(self, query: str, top_k: int = 3, threshold: float = 0.25) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieves the top-k most relevant knowledge articles for a query.
        Uses vector semantic similarity if available; falls back to lexical search.
        """
        if self.embeddings is not None and self.is_configured:
            query_vec = self._get_query_embedding(query)
            if query_vec is not None:
                # Cosine similarities
                similarities = np.dot(self.embeddings, query_vec)
                indices = np.argsort(similarities)[::-1]

                results = []
                for idx in indices[:top_k]:
                    sim = float(similarities[idx])
                    if sim >= threshold or len(results) == 0:
                        results.append((self.documents[idx], sim))
                return results

        # Fallback to lexical search
        return self._lexical_search(query, top_k=top_k)

    def generate_response(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Grounded answer generation. Combines retrieved context with LLM generation.
        Returns a dictionary containing 'answer', 'sources', and 'retrieved_docs'.
        """
        retrieved_items = self.retrieve(query, top_k=top_k)
        retrieved_docs = [item[0] for item in retrieved_items]

        # Extract unique sources
        sources = []
        for doc in retrieved_docs:
            source_info = {
                "title": doc.get("title", "Untitled Document"),
                "category": doc.get("category", "General"),
                "url": doc.get("source_url", "#")
            }
            if source_info not in sources:
                sources.append(source_info)

        # Build context prompt
        context_blocks = []
        for i, doc in enumerate(retrieved_docs, start=1):
            context_blocks.append(
                f"[Source {i} - {doc.get('title')} ({doc.get('category')})]\n"
                f"{doc.get('content')}"
            )
        context_str = "\n\n".join(context_blocks)

        system_prompt = (
            "You are the Cambridge Support RAG Bot, an official, supportive, and precise AI assistant "
            "for University of Cambridge applicants, students, and staff.\n\n"
            "Guidelines:\n"
            "1. Answer the question accurately using ONLY the retrieved context below.\n"
            "2. If the context does not contain sufficient information, politely state that you do not "
            "have that specific information and suggest contacting the relevant Cambridge department or college.\n"
            "3. Be welcoming, clear, professional, and collegiate in tone.\n"
            "4. Mention relevant source titles when helpful.\n\n"
            f"--- RETRIEVED CONTEXT ---\n{context_str}\n--- END CONTEXT ---"
        )

        # If LLM client is configured, generate with Gemini / OpenAI / Azure
        if self.is_configured and self.client:
            try:
                if self.provider == "gemini":
                    # Construct prompt for Gemini
                    convo_history = ""
                    if chat_history:
                        for msg in chat_history[-4:]:
                            speaker = "User" if msg["role"] == "user" else "Assistant"
                            convo_history += f"{speaker}: {msg['content']}\n"

                    full_prompt = (
                        f"{system_prompt}\n\n"
                        f"Previous Conversation:\n{convo_history}\n"
                        f"User Question: {query}\n"
                        "Answer:"
                    )
                    try:
                        res = self.client.models.generate_content(
                            model=self.chat_model,
                            contents=full_prompt,
                        )
                        answer = res.text or ""
                    except Exception as inner_err:
                        # Fallback to gemini-2.0-flash or gemini-2.5-flash
                        fallback_model = "gemini-2.0-flash" if self.chat_model != "gemini-2.0-flash" else "gemini-2.5-flash"
                        res = self.client.models.generate_content(
                            model=fallback_model,
                            contents=full_prompt,
                        )
                        answer = res.text or ""
                    return {
                        "answer": answer,
                        "sources": sources,
                        "retrieved_docs": retrieved_docs,
                        "mode": f"Online (Google Gemini: {self.chat_model})",
                    }
                else:
                    messages = [{"role": "system", "content": system_prompt}]
                    if chat_history:
                        for msg in chat_history[-4:]:
                            messages.append({"role": msg["role"], "content": msg["content"]})
                    messages.append({"role": "user", "content": query})

                    response = self.client.chat.completions.create(
                        model=self.chat_model,
                        messages=messages,
                        temperature=0.2,
                        max_tokens=600,
                    )
                    answer = response.choices[0].message.content or ""
                    return {
                        "answer": answer,
                        "sources": sources,
                        "retrieved_docs": retrieved_docs,
                        "mode": f"Online ({self.provider.capitalize()} OpenAI: {self.chat_model})",
                    }
            except Exception as e:
                return {
                    "answer": (
                        f"*(API Call Error: {str(e)})*\n\n"
                        f"**Here is the relevant information retrieved from the Cambridge Knowledge Base:**\n\n"
                        + "\n\n".join([f"- **{d['title']}**: {d['content']}" for d in retrieved_docs])
                    ),
                    "sources": sources,
                    "retrieved_docs": retrieved_docs,
                    "mode": "Offline Knowledge Retrieval",
                }

        # Offline / Demo Mode when no API keys are configured yet
        fallback_answer = (
            "👋 **Welcome to Cambridge Support!** *(Demo Mode: API keys not detected in `.env`)*\n\n"
            "Based on our knowledge base, here is the relevant guidance on your inquiry:\n\n"
        )
        for doc in retrieved_docs:
            fallback_answer += f"### 📌 {doc.get('title')} ({doc.get('category')})\n{doc.get('content')}\n\n"

        fallback_answer += (
            "\n> 💡 *To activate live AI generation, configure your Azure OpenAI or standard OpenAI credentials in `.env`.*"
        )

        return {
            "answer": fallback_answer,
            "sources": sources,
            "retrieved_docs": retrieved_docs,
            "mode": "Knowledge Base Direct Retrieval (Offline)",
        }
