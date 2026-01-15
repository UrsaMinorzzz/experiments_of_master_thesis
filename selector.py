import numpy as np
import re
from typing import List, Dict, Set

class BudgetAwareSelector:
    """
    Implements the Budget-Aware Greedy MMR with Adaptive Gamma.
    Reference: AdaGReS (2025) adapted for Code Jaccard.
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, token_budget: int = 4096):
        self.alpha = alpha  # Weight for Relevance
        self.beta = beta    # Weight for Structure
        self.budget = token_budget

    def select(self, 
               query_context: str, 
               candidates: List[Dict], 
               tokenizer) -> List[Dict]:
        """
        Args:
            query_context: The incomplete code context.
            candidates: List of dicts, each containing {'code': str, 'score': float, 'id': str}.
            tokenizer: HuggingFace tokenizer (or similar) with .encode() method.
            
        Returns:
            Selected candidates fitting the budget.
        """
        
        # 1. Pre-computation & Normalization
        # ---------------------------------------------------------
        raw_scores = [c['score'] for c in candidates]
        norm_scores = self._normalize_scores(raw_scores)
        
        # Pre-compute Jaccard representations (sets of tokens) for speed
        # O(N) once, instead of O(N^2) inside loops
        candidate_sets = [self._tokenize_to_set(c['code']) for c in candidates]
        
        # Calculate Adaptive Gamma (The "Thesis Core")
        gamma = self._calculate_adaptive_gamma(norm_scores, candidate_sets)
        
        # Calculate Structure Scores (S)
        structure_scores = [self._evaluate_structure(c['code']) for c in candidates]

        # 2. Greedy Selection Loop
        # ---------------------------------------------------------
        selected_indices = []
        current_tokens = len(tokenizer.encode(query_context))
        remaining_budget = self.budget - current_tokens
        
        # Mask for tracking available candidates
        available_mask = [True] * len(candidates)
        
        while remaining_budget > 0 and sum(available_mask) > 0:
            best_idx = -1
            best_gain = -float('inf')
            
            # Iterate through available candidates to find max Marginal Gain
            for i, is_available in enumerate(available_mask):
                if not is_available:
                    continue
                
                # Check strict budget constraint first
                # (Simple length estimation to save compute, or precise if needed)
                code_len = len(tokenizer.encode(candidates[i]['code']))
                if code_len > remaining_budget:
                    available_mask[i] = False # Too big, drop it
                    continue
                
                # --- CALCULATION OF v_i (Marginal Gain) ---
                # Term 1: Relevance (Normalized)
                rel_term = self.alpha * norm_scores[i]
                
                # Term 2: Structure
                struct_term = self.beta * structure_scores[i]
                
                # Term 3: Redundancy (Max Marginal Redundancy)
                if not selected_indices:
                    red_term = 0.0
                else:
                    # Calculate Jaccard against ALREADY SELECTED items
                    # We utilize the pre-computed sets for speed
                    max_jaccard = 0.0
                    for sel_idx in selected_indices:
                        j_val = self._jaccard_similarity(candidate_sets[i], candidate_sets[sel_idx])
                        if j_val > max_jaccard:
                            max_jaccard = j_val
                    
                    # [CRITICAL FIX] Boosting the penalty
                    # We multiply by (alpha + beta) to ensure penalty can override positive gains
                    # Thesis justification: "Penalty Normalization"
                    penalty_scale = self.alpha + self.beta + 1.0
                    red_term = gamma * max_jaccard * penalty_scale
                
                gain = rel_term + struct_term - red_term

                # [DEBUG LINE] 打印每个候选的得分构成
                # print(f"ID: {candidates[i]['id']} | Gain: {gain:.4f} (Rel:{rel_term:.2f} Str:{struct_term:.2f} Red:{red_term:.2f})")
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            # [CRITICAL FIX] Stopping Criteria
            # If the best candidate contributes more Redundancy than Relevance (Gain <= 0),
            # stop selecting immediately. Do not fill budget with noise.
            if best_gain <= 0:
                break

            # End of search for this step
            if best_idx != -1:
                selected_indices.append(best_idx)
                available_mask[best_idx] = False
                remaining_budget -= len(tokenizer.encode(candidates[best_idx]['code']))
            else:
                # No valid candidate found (all remaining are too big or exhausted)
                break
                
        return [candidates[i] for i in selected_indices]

    def _calculate_adaptive_gamma(self, norm_scores: np.ndarray, candidate_sets: List[Set]) -> float:
        """
        Refined Formula: Gamma scales linearly with pool density to avoid explosion.
        Gamma = Avg(Rel) * (1 + Avg(Red))
        """
        if not candidate_sets:
            return 1.0
            
        avg_relevance = np.mean(norm_scores)
        
        # Estimate pool redundancy (Sample top 5)
        limit = min(len(candidate_sets), 5)
        if limit < 2:
            return 1.0 
            
        j_sum = 0
        count = 0
        for i in range(limit):
            for j in range(i + 1, limit):
                j_sum += self._jaccard_similarity(candidate_sets[i], candidate_sets[j])
                count += 1
        
        avg_redundancy = j_sum / count if count > 0 else 0.0
        
        # New Stable Formula:
        # If pool is diverse (Red~0.1), Gamma ~ Rel * 1.1 (Mild penalty)
        # If pool is clone (Red~0.9), Gamma ~ Rel * 1.9 (Heavy penalty)
        gamma = avg_relevance * (1.0 + avg_redundancy)
        
        return float(gamma)
    def _normalize_scores(self, scores: List[float]) -> np.ndarray:
        s = np.array(scores)
        if len(s) == 0: return s
        min_s, max_s = s.min(), s.max()
        if max_s == min_s: return np.ones_like(s)
        return (s - min_s) / (max_s - min_s)

    def _tokenize_to_set(self, text: str, n: int = 4) -> Set[tuple]:
        """
        升级版: Token-level N-gram Shingling.
        1. 使用正则分词，将 'a.b()' 切分为 ['a', '.', 'b', '(', ')']
        2. 生成 N-gram tuple，保留序列信息。
        """
        # A simple regex for code tokenization: 
        # \w+ matches identifiers/keywords, [^\w\s] matches operators/punctuation
        tokens = re.findall(r'\w+|[^\w\s]', text)
        
        if len(tokens) < n:
            # 如果代码极短，回退到 Unigram 或直接返回整个序列
            return set([tuple(tokens)])
            
        # 生成 N-grams (Shingles)
        # e.g., N=3, [a, b, c, d] -> {(a,b,c), (b,c,d)}
        ngrams = set(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
        return ngrams

    def _jaccard_similarity(self, set_a: Set, set_b: Set) -> float:
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def _evaluate_structure(self, code: str) -> float:
        # Placeholder for AST parsing logic
        # In implementation: use tree-sitter to check for parse errors
        # Tonight: Return 1.0 (Assume valid) to keep momentum
        return 1.0