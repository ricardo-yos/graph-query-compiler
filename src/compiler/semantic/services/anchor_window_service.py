"""
Semantic Local Retriever
========================

Dense retriever for detecting attribute anchors and extracting
contextual windows in natural language questions within a
semantic compiler pipeline.

This component generates n-grams from the question text, computes
embeddings, identifies the most relevant segment (anchor) related
to a target attribute, and extracts a right-oriented context window
for downstream semantic tasks such as operator resolution or value extraction.

Resolution Strategy
-------------------
1. Tokenize the question and generate n-grams.
2. Compute embeddings for n-grams and the target attribute description.
3. Select the n-gram with the highest embedding similarity to the target.
4. Extract a right-oriented semantic window around the anchor for further analysis.

Input
-----
question : str
    Natural language user query.
target_text : str
    Target attribute or keyword to locate in the question.

Output
------
dict
    Anchor information containing:
        - anchor_text : Detected n-gram text
        - similarity  : Cosine similarity score (0 to 1)
        - window_text : Contextual window around the anchor
"""

from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SemanticLocalRetriever:
    """
    Mini dense retriever for semantic compiler questions.
    Detects attribute/operator anchors and extracts contextual windows.
    """

    def __init__(
        self,
        model: Optional[SentenceTransformer] = None,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ngram_range: tuple = (1, 3),
        window_size: int = 5,
        max_right_span: Optional[int] = None
    ):
        """
        Initialize retriever with model and window parameters.

        Parameters
        ----------
        model : SentenceTransformer, optional
            Pre-initialized embedding model (dependency injection).
        model_name : str
            Model name if model is not provided.
        ngram_range : tuple
            Range of n-grams (min_n, max_n).
        window_size : int
            General contextual window size.
        max_right_span : int, optional
            Maximum tokens to capture to the right of the anchor.
            Defaults to window_size if None.
        """
        self.model = model if model is not None else SentenceTransformer(model_name)
        self.ngram_range = ngram_range
        self.window_size = window_size
        self.max_right_span = max_right_span if max_right_span is not None else window_size

        # Caches to avoid recomputation
        self._ngram_cache = {}
        self._embedding_cache = {}

    # -----------------------------
    # Embedding utility
    # -----------------------------
    def _embed(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed a list of texts using the SentenceTransformer model.
        Caches embeddings to reduce redundant computations.

        Parameters
        ----------
        texts : List[str]
            Texts to embed.

        Returns
        -------
        List[np.ndarray]
            Normalized embeddings for each text.
        """
        embeddings = []
        for t in texts:
            if t not in self._embedding_cache:
                self._embedding_cache[t] = self.model.encode([t], normalize_embeddings=True)[0]
            embeddings.append(self._embedding_cache[t])
        return embeddings

    # -----------------------------
    # N-gram generator
    # -----------------------------
    def _generate_ngrams(self, tokens: List[str]) -> List[Dict[str, int]]:
        """
        Generate n-grams for a list of tokens, with caching.

        Parameters
        ----------
        tokens : List[str]
            Tokenized question.

        Returns
        -------
        List[Dict[str, int]]
            List of n-grams with start/end indices.
        """
        key = tuple(tokens)
        if key in self._ngram_cache:
            return self._ngram_cache[key]

        ngrams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                ngrams.append({
                    "text": " ".join(tokens[i:i+n]),
                    "start": i,
                    "end": i + n - 1
                })

        self._ngram_cache[key] = ngrams
        return ngrams

    # -----------------------------
    # Window extraction
    # -----------------------------
    def _get_window(self, tokens: List[str], start: int, end: int) -> str:
        """
        Extract a right-oriented semantic window around a given n-gram.

        Parameters
        ----------
        tokens : List[str]
            Tokenized question.
        start : int
            Start index of n-gram.
        end : int
            End index of n-gram.

        Returns
        -------
        str
            Right-oriented window text including the anchor.
        """
        right = min(len(tokens), end + self.max_right_span + 1)
        return " ".join(tokens[start:right])

    # -----------------------------
    # Main anchor detection
    # -----------------------------
    def detect_anchor(self, question: str, target_text: str) -> Dict[str, str]:
        """
        Detect the most similar n-gram to the target_text and extract a
        semantic window around it.

        Parameters
        ----------
        question : str
            Natural language question.
        target_text : str
            Attribute keyword to detect.

        Returns
        -------
        Dict[str, str]
            Anchor information including:
                - anchor_text : detected n-gram
                - similarity  : cosine similarity score
                - window_text : contextual window
        """
        question = question.lower()
        tokens = question.split()
        if not tokens:
            return {"anchor_text": None, "similarity": 0.0, "window_text": None}

        ngrams = self._generate_ngrams(tokens)
        ngram_texts = [ng["text"] for ng in ngrams]

        ngram_embeddings = self._embed(ngram_texts)
        target_embedding = self._embed([target_text])[0]

        max_score = -1.0
        best_ngram = None
        for idx, ng_emb in enumerate(ngram_embeddings):
            score = float(cosine_similarity([target_embedding], [ng_emb])[0][0])
            if score > max_score:
                max_score = score
                best_ngram = ngrams[idx]

        window = self._get_window(tokens, best_ngram["start"], best_ngram["end"]) if best_ngram else None

        return {
            "anchor_text": best_ngram["text"] if best_ngram else None,
            "similarity": round(max_score, 3),
            "window_text": window
        }

    # -----------------------------
    # Multi-target detection
    # -----------------------------
    def detect_multiple(self, question: str, targets: List[str]) -> List[Dict[str, str]]:
        """
        Detect anchors for multiple target terms in a question.

        Parameters
        ----------
        question : str
            Natural language question.
        targets : List[str]
            List of attribute keywords to detect.

        Returns
        -------
        List[Dict[str, str]]
            List of anchor detection results for each target.
        """
        return [{"target": t, **self.detect_anchor(question, t)} for t in targets]
