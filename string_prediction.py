# string_prediction.py

from typing import List, Dict, Tuple
from collections import defaultdict
import math


class StringIndexPredictor:
    """
    An improved predictor for string/word data that predicts index based on
    letter prefixes with adaptive length, RMSE calculation, and interpolation.
    
    This predictor learns a mapping from letter prefixes to average indices,
    computes error metrics (RMSE/std deviation), and uses interpolation for
    better predictions. Useful for lexicographically sorted word lists.
    """

    def __init__(self, max_prefix_length: int = 3, use_rmse: bool = True) -> None:
        """
        Initialize the string index predictor.

        Parameters
        ----------
        max_prefix_length : int, default=3
            Maximum number of letters to use as prefix (1-4 recommended).
            Longer prefixes provide more accuracy but require more memory.
        use_rmse : bool, default=True
            If True, compute RMSE/std deviation for each prefix to enable
            error-based search window optimization.
        """
        self.max_prefix_length = max(1, min(4, max_prefix_length))  # Clamp to 1-4
        self.use_rmse = use_rmse
        self._key_to_indices: Dict[str, List[int]] = defaultdict(list)
        self._key_to_mean_index: Dict[str, float] = {}
        self._key_to_std: Dict[str, float] = {}  # Standard deviation for each prefix
        self._key_to_range: Dict[str, Tuple[int, int]] = {}  # (min_idx, max_idx) for each prefix
        self._words_length: int = 0
        self._is_fitted: bool = False
        self.rmse: float = 0.0  # Overall RMSE across all prefixes

    def _get_key(self, word: str, length: int = None) -> str:
        """
        Extract the key (prefix) from a word.

        Parameters
        ----------
        word : str
            Input word.
        length : int, optional
            Prefix length to use. If None, uses self.max_prefix_length.

        Returns
        -------
        str
            Key (prefix of specified length, lowercased).
        """
        if not word:
            return ""
        
        word_lower = word.lower()
        prefix_len = length if length is not None else self.max_prefix_length
        prefix_len = min(prefix_len, len(word_lower))
        
        return word_lower[:prefix_len] if prefix_len > 0 else ""

    def fit(self, words: List[str]) -> None:
        """
        Fit the predictor on a sorted list of words.

        Parameters
        ----------
        words : List[str]
            Sorted list of strings (words).
        """
        if len(words) == 0:
            raise ValueError("Cannot fit StringIndexPredictor on an empty list.")

        self._words_length = len(words)
        self._key_to_indices.clear()
        self._key_to_mean_index.clear()
        self._key_to_std.clear()
        self._key_to_range.clear()

        # Build mapping from keys to indices using max_prefix_length
        for idx, word in enumerate(words):
            key = self._get_key(word, self.max_prefix_length)
            if key:  # Only process non-empty keys
                self._key_to_indices[key].append(idx)

        # Compute statistics for each key
        all_errors = []
        for key, indices in self._key_to_indices.items():
            if indices:
                mean_idx = sum(indices) / len(indices)
                self._key_to_mean_index[key] = mean_idx
                
                # Store min/max range
                self._key_to_range[key] = (min(indices), max(indices))
                
                # Compute standard deviation if RMSE is enabled
                if self.use_rmse and len(indices) > 1:
                    variance = sum((i - mean_idx) ** 2 for i in indices) / len(indices)
                    std = math.sqrt(variance)
                    self._key_to_std[key] = std
                    
                    # Collect errors for overall RMSE
                    for idx in indices:
                        all_errors.append((idx - mean_idx) ** 2)
                elif self.use_rmse:
                    # Single element, no variance
                    self._key_to_std[key] = 0.0

        # Compute overall RMSE
        if self.use_rmse and all_errors:
            self.rmse = math.sqrt(sum(all_errors) / len(all_errors))
        else:
            self.rmse = 0.0

        self._is_fitted = True

    def predict(self, target: str) -> int:
        """
        Predict an index for the given target word using adaptive prefix matching
        and interpolation.

        Parameters
        ----------
        target : str
            Target word to predict index for.

        Returns
        -------
        int
            Predicted index in [0, len(words) - 1].
        """
        if not self._is_fitted:
            raise RuntimeError("StringIndexPredictor must be fitted before calling predict().")

        if not target:
            return 0

        target_lower = target.lower()
        
        # Try progressively shorter prefixes if longer ones don't exist
        for prefix_len in range(self.max_prefix_length, 0, -1):
            key = self._get_key(target_lower, prefix_len)
            
            if key in self._key_to_mean_index:
                # Found matching prefix, use its mean index
                pred = self._key_to_mean_index[key]
                
                # If we have range information and target is longer than prefix,
                # try to interpolate within the prefix range
                if (prefix_len < len(target_lower) and 
                    key in self._key_to_range and
                    len(self._key_to_indices[key]) > 1):
                    min_idx, max_idx = self._key_to_range[key]
                    
                    # Find neighboring prefixes for interpolation
                    # Try to find prefix that would come after current one
                    next_char = target_lower[prefix_len] if prefix_len < len(target_lower) else None
                    if next_char and next_char.isalpha():
                        # Try to find next prefix (e.g., if key="ab", try "abc", "abd", etc.)
                        next_key = key + next_char
                        if next_key in self._key_to_mean_index:
                            # Interpolate between current and next prefix
                            current_mean = self._key_to_mean_index[key]
                            next_mean = self._key_to_mean_index[next_key]
                            # Simple interpolation: use position between them
                            pred = (current_mean + next_mean) / 2.0
                
                # Clamp to valid range
                pred_clamped = max(0.0, min(pred, float(self._words_length - 1)))
                return int(pred_clamped)

        # Fallback: no matching prefix found, use alphabetical approximation
        first_char = target_lower[0] if target_lower else 'a'
        
        if first_char.isalpha():
            # Map 'a'..'z' to [0 .. len(words)-1]
            # Use distribution-aware mapping if we have prefix data
            if self._key_to_mean_index:
                # Find closest known prefixes
                char_code = ord(first_char)
                closest_before = None
                closest_after = None
                
                for key in self._key_to_mean_index.keys():
                    if len(key) == 1:
                        key_code = ord(key[0])
                        if key_code < char_code:
                            if closest_before is None or key_code > ord(closest_before[0]):
                                closest_before = key
                        elif key_code > char_code:
                            if closest_after is None or key_code < ord(closest_after[0]):
                                closest_after = key
                
                # Interpolate between closest prefixes
                if closest_before and closest_after:
                    before_idx = self._key_to_mean_index[closest_before]
                    after_idx = self._key_to_mean_index[closest_after]
                    before_code = ord(closest_before[0])
                    after_code = ord(closest_after[0])
                    
                    if after_code != before_code:
                        ratio = (char_code - before_code) / (after_code - before_code)
                        pred = before_idx + ratio * (after_idx - before_idx)
                    else:
                        pred = before_idx
                elif closest_before:
                    # Extrapolate forward
                    before_idx = self._key_to_mean_index[closest_before]
                    pred = before_idx + (char_code - ord(closest_before[0])) * (self._words_length / 26.0)
                elif closest_after:
                    # Extrapolate backward
                    after_idx = self._key_to_mean_index[closest_after]
                    pred = after_idx - (ord(closest_after[0]) - char_code) * (self._words_length / 26.0)
                else:
                    # No known prefixes, use simple linear mapping
                    char_pos = char_code - ord('a')
                    pred = (char_pos / 25.0) * (self._words_length - 1)
            else:
                # No prefix data, use simple linear mapping
                char_pos = ord(first_char) - ord('a')
                pred = (char_pos / 25.0) * (self._words_length - 1)
        else:
            # Non-alphabetic character, default to middle
            pred = self._words_length / 2.0

        # Clamp to valid range
        pred_clamped = max(0.0, min(pred, float(self._words_length - 1)))
        return int(pred_clamped)

