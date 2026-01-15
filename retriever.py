import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict, Any

class DenseRetriever:
    """
    Production-grade wrapper for Dense Retrieval.
    Uses 'sentence-transformers' for encoding and 'faiss' for exact search.
    """
    
    def __init__(self, model_name: str = "codesage/codesage-small-v2", device: str = None):
        """
        Args:
            model_name: HuggingFace model ID. 
            device: 'cuda', 'mps', or 'cpu'. Auto-detected if None.
        """
        self._detect_device(device)
        print(f"[DenseRetriever] Loading model {model_name} on {self.device}...")
        
        # Load model. trust_remote_code needed for some code models
        self.model = SentenceTransformer(model_name, device=self.device, trust_remote_code=True)
        
        # We use Inner Product (IP) index. 
        # Since embeddings will be normalized, IP == Cosine Similarity.
        self.index = None
        self.candidates: List[str] = []
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def _detect_device(self, device):
        if device:
            self.device = device
            return
        
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps" # For macOS M-chips
        else:
            self.device = "cpu"

    def index_candidates(self, candidates: List[str]):
        """
        Encodes and indexes a list of code snippets.
        Crucial: Normalizes embeddings so Inner Product equals Cosine Similarity.
        """
        if not candidates:
            # print("[DenseRetriever] Warning: No candidates provided to index.")
            self.candidates = []
            self.index = None
            return

        self.candidates = candidates
        
        # Encode with normalization for Cosine Similarity
        # batch_size can be adjusted based on VRAM
        embeddings = self.model.encode(
            candidates, 
            convert_to_numpy=True, 
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32 
        )
        
        # Initialize FAISS Index (IndexFlatIP is exact search)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieves top_k candidates for a given query.
        Returns: List of dicts with 'content', 'score' (Cosine), 'original_idx'
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        # Encode query
        query_emb = self.model.encode(
            [query], 
            convert_to_numpy=True, 
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # FAISS Search
        # D: Distances (Scores), I: Indices
        # k cannot be larger than the number of candidates
        k_search = min(top_k, len(self.candidates))
        D, I = self.index.search(query_emb, k=k_search)
        
        results = []
        # I[0] because we only have 1 query
        for rank, (score, idx) in enumerate(zip(D[0], I[0])):
            if idx == -1: continue 
            
            results.append({
                'content': self.candidates[idx],
                'score': float(score), # Raw Cosine Similarity [-1, 1]
                'rank': rank,
                'original_idx': int(idx)
            })
            
        return results

    def clear_index(self):
        """Resets the index to free memory."""
        if self.index:
            self.index.reset()
        self.candidates = []