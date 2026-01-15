import numpy as np
import re
from typing import List, Dict, Set

class BudgetAwareSelector:
    """
    [Phase B Updated]
    - Removed Min-Max Normalization (Uses Raw Cosine Scores).
    - Optimized for Dense Retrieval interactions.
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, token_budget: int = 2048):
        self.alpha = alpha
        self.beta = beta
        self.budget = token_budget

    def select(self, 
               query_context: str, 
               candidates: List[Dict], 
               tokenizer) -> List[Dict]:
        
        # 1. Pre-computation (NO Min-Max Normalization)
        # ---------------------------------------------------------
        # 直接使用 Retriever 传来的 Raw Cosine Scores
        scores = np.array([c['score'] for c in candidates])
        
        # Tokenize candidates once for Jaccard
        candidate_sets = [self._tokenize_to_set(c['code']) for c in candidates]
        
        # Calculate Adaptive Gamma based on Raw Stats
        gamma = self._calculate_adaptive_gamma(scores, candidate_sets)
        
        # Structure (Currently placeholder 1.0)
        structure_scores = [self._evaluate_structure(c['code']) for c in candidates]

        # 2. Greedy Selection Loop
        # ---------------------------------------------------------
        selected_indices = []
        current_tokens = len(tokenizer.encode(query_context))
        remaining_budget = self.budget - current_tokens
        
        available_mask = [True] * len(candidates)
        
        while remaining_budget > 0 and sum(available_mask) > 0:
            best_idx = -1
            best_gain = -float('inf')
            
            for i, is_available in enumerate(available_mask):
                if not is_available: continue
                
                # Check budget (Strict)
                # Ensure we don't select chunks that individually exceed remaining budget
                code_len = len(tokenizer.encode(candidates[i]['code']))
                if code_len > remaining_budget:
                    available_mask[i] = False
                    continue
                
                # --- GAIN CALCULATION (Raw Scores) ---
                # Term 1: Relevance (Raw Cosine ~0.7-0.9)
                rel_term = self.alpha * scores[i]
                
                # Term 2: Structure
                struct_term = self.beta * structure_scores[i]
                
                # Term 3: Redundancy
                if not selected_indices:
                    red_term = 0.0
                else:
                    max_jaccard = 0.0
                    for sel_idx in selected_indices:
                        j_val = self._jaccard_similarity(candidate_sets[i], candidate_sets[sel_idx])
                        if j_val > max_jaccard:
                            max_jaccard = j_val
                    
                    # Penalty Logic: 
                    # Raw Score (0.8) vs Penalty (Gamma * 1.0 * Scale)
                    # No extra scaling needed if Gamma is derived from scores.
                    red_term = gamma * max_jaccard
                
                gain = rel_term + struct_term - red_term
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            # [Gatekeeper Logic]
            # 如果最佳增益为负（说明冗余惩罚超过了相关性收益），立即停止。
            # 这是 "宁缺毋滥" 的核心机制。
            if best_gain <= 0:
                break

            if best_idx != -1:
                selected_indices.append(best_idx)
                available_mask[best_idx] = False
                remaining_budget -= len(tokenizer.encode(candidates[best_idx]['code']))
            else:
                break
                
        return [candidates[i] for i in selected_indices]

    def _calculate_adaptive_gamma(self, scores: np.ndarray, candidate_sets: List[Set]) -> float:
        """
        Adaptive Penalty Coefficient.
        Logic: Gamma = Avg(Relevance) * (1 + Pool_Redundancy)
        
        Why: 
        - If Pool is diverse (Red~0.1), Gamma ~= Avg_Rel. 
          Gain ~= Rel - Rel * Jaccard. A 100% clone cancels out relevance. Perfect.
        - If Pool is repetitive (Red~0.8), Gamma ~= 1.8 * Avg_Rel.
          We punish clones HARDER to force diversity.
        """
        if len(scores) == 0: return 1.0
        
        avg_relevance = np.mean(scores)
        
        # Estimate pool redundancy (Sample top 5 for speed)
        limit = min(len(candidate_sets), 5)
        if limit < 2: return 1.0 
            
        j_sum = 0
        count = 0
        for i in range(limit):
            for j in range(i + 1, limit):
                j_sum += self._jaccard_similarity(candidate_sets[i], candidate_sets[j])
                count += 1
        
        avg_redundancy = j_sum / count if count > 0 else 0.0
        
        # Raw Score Gamma Formula
        gamma = avg_relevance * (1.0 + avg_redundancy)
        return float(gamma)

    def _tokenize_to_set(self, text: str, n: int = 4) -> Set[tuple]:
        tokens = re.findall(r'\w+|[^\w\s]', text)
        if len(tokens) < n: return set([tuple(tokens)])
        return set(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

    def _jaccard_similarity(self, set_a: Set, set_b: Set) -> float:
        if not set_a and not set_b: return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def _evaluate_structure(self, code: str) -> float:
        return 1.0 # Placeholder