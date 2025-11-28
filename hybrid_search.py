# hybrid_search.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Protocol, runtime_checkable, TypeVar, Any
import numpy as np

T = TypeVar('T', bound=Any)


@runtime_checkable
class IndexPredictor(Protocol):
    """
    Protocol so MLIndexPredictor (or any other predictor)
    can be plugged into predictive_binary_search_ml.
    
    Predictors may optionally have an 'rmse' attribute for
    error-based search window optimization.
    """
    def predict(self, target: Any) -> int:  # pragma: no cover - protocol signature
        """
        Predict an index for the given target value.
        
        Parameters
        ----------
        target : Any
            Target value (can be float, str, etc.)
        
        Returns
        -------
        int
            Predicted index.
        """
        ...
    
    # Optional attribute for RMSE-based window optimization
    rmse: float = 0.0  # type: ignore


@dataclass
class SearchResult:
    found: bool
    index: int
    comparisons: int


def classic_binary_search(arr: List[Any], target: Any) -> SearchResult:
    """
    Standard binary search on a sorted array (works with numbers, strings, etc.).

    comparisons = number of comparisons between `target` and array elements.
    
    Parameters
    ----------
    arr : List[Any]
        Sorted array of comparable elements.
    target : Any
        Target value to search for.
    
    Returns
    -------
    SearchResult
        Search result with found flag, index, and comparison count.
    """
    left, right = 0, len(arr) - 1
    comparisons = 0

    while left <= right:
        mid = (left + right) // 2
        value = arr[mid]

        # 1st comparison: target == value
        comparisons += 1
        if target == value:
            return SearchResult(True, mid, comparisons)

        # 2nd comparison: target > value (only if not equal)
        comparisons += 1
        if target > value:
            left = mid + 1
        else:
            right = mid - 1

    return SearchResult(False, -1, comparisons)


def interpolation_search_numeric(arr: List[float], target: float) -> SearchResult:
    """
    Improved interpolation search with adaptation and fallback to binary search.

    - Detects if interpolation is suitable for the data
    - Automatically switches to binary search when interpolation performs poorly
    - Uses smarter iteration limits to prevent infinite loops on bad distributions
    """
    if not arr:
        return SearchResult(False, -1, 0)

    left, right = 0, len(arr) - 1
    comparisons = 0
    max_iterations = int(np.log2(len(arr))) * 2  # Balanced limit
    iterations = 0
    last_range = right - left  # Track range to detect stuck iterations

    # Count comparisons in the while condition
    while left <= right:
        # Count boundary comparisons: arr[left] <= target <= arr[right]
        comparisons += 1
        if arr[left] > target:
            break
        comparisons += 1
        if arr[right] < target:
            break
        
        iterations += 1
        
        # Safety check: if we exceeded max iterations, fallback to binary
        if iterations > max_iterations:
            break
        
        # Check if range is not shrinking (stuck)
        current_range = right - left
        if current_range >= last_range and iterations > 3 and current_range > 1:
            # Range is not shrinking, switch to binary search
            break
        last_range = current_range
        
        if left == right:
            value = arr[left]
            comparisons += 1
            if target == value:
                return SearchResult(True, left, comparisons)
            return SearchResult(False, -1, comparisons)

        # Avoid division by zero
        denom = arr[right] - arr[left]
        if denom == 0:
            # All elements in [left, right] are equal
            value = arr[left]
            comparisons += 1
            if target == value:
                return SearchResult(True, left, comparisons)
            return SearchResult(False, -1, comparisons)

        # Standard interpolation formula
        ratio = (target - arr[left]) / denom
        
        # Calculate position
        pos = left + int(ratio * (right - left))
        
        # Clamp position to valid range
        pos = max(left, min(right, pos))

        value = arr[pos]

        # Comparison 1: target == value
        comparisons += 1
        if target == value:
            return SearchResult(True, pos, comparisons)

        # Comparison 2: target > value
        comparisons += 1
        if target > value:
            left = pos + 1
        else:
            right = pos - 1

    # Fallback: if interpolation didn't work, use binary search
    # (this happens rarely, only for very bad distributions)
    while left <= right:
        mid = (left + right) // 2
        value = arr[mid]
        
        comparisons += 1
        if target == value:
            return SearchResult(True, mid, comparisons)
        
        comparisons += 1
        if target > value:
            left = mid + 1
        else:
            right = mid - 1

    return SearchResult(False, -1, comparisons)


def predictive_binary_search_ml(
    arr: List[Any],
    target: Any,
    predictor: IndexPredictor,
    k: float = 2.0,
) -> SearchResult:
    """
    Improved hybrid search with RMSE-based narrowed search window.

    This implements a simplified learned index approach:
    1. Predict starting index using ML model
    2. If RMSE is available, create a search window: [pred_idx - k*rmse, pred_idx + k*rmse]
    3. Run binary search only inside this window
    4. This makes the search more efficient by focusing on the most likely region

    Parameters
    ----------
    arr : List[Any]
        Sorted array to search in (works with numbers, strings, etc.).
    target : Any
        Target value to find.
    predictor : IndexPredictor
        ML predictor that can predict index from value.
        May optionally have an 'rmse' attribute for window optimization.
    k : float, default=2.0
        Multiplier for RMSE when creating search window.
        Larger k = wider window (more conservative).
        Smaller k = narrower window (more aggressive, faster if accurate).

    Returns
    -------
    SearchResult
        Search result with found flag, index, and comparison count.
    """
    if not arr:
        return SearchResult(False, -1, 0)

    comparisons = 0

    # Get ML prediction
    pred_idx = predictor.predict(target)
    
    # Clamp prediction to valid range
    pred_idx = max(0, min(len(arr) - 1, pred_idx))
    
    # Check if target is at predicted index
    value = arr[pred_idx]
    comparisons += 1
    if target == value:
        return SearchResult(True, pred_idx, comparisons)

    # Determine search window using RMSE if available
    # Check if predictor has rmse attribute (using hasattr for protocol compatibility)
    rmse = getattr(predictor, 'rmse', 0.0)
    
    if rmse > 0:
        # Use RMSE-based window: [pred_idx - k*rmse, pred_idx + k*rmse]
        window_size = int(k * rmse)
        left = max(0, pred_idx - window_size)
        right = min(len(arr) - 1, pred_idx + window_size)
        
        # Verify target is within bounds of the window
        # Count comparisons for boundary checks
        # We always check both boundaries to determine window validity
        comparisons += 1
        target_before_window = target < arr[left]
        if target_before_window:
            # Target is before window, search from start to window start
            left, right = 0, pred_idx - 1
        else:
            # Check upper bound (always done, even if we don't use the result)
            comparisons += 1
            target_after_window = target > arr[right]
            if target_after_window:
                # Target is after window, search from window end to array end
                left, right = pred_idx + 1, len(arr) - 1
            # else: target is within window bounds, use the window
    else:
        # No RMSE available, use traditional approach: choose half based on comparison
        comparisons += 1
        if target > value:
            left, right = pred_idx + 1, len(arr) - 1
        else:
            left, right = 0, pred_idx - 1

    # Binary search within the determined window
    while left <= right:
        mid = (left + right) // 2
        value_mid = arr[mid]

        comparisons += 1
        if target == value_mid:
            return SearchResult(True, mid, comparisons)

        comparisons += 1
        if target > value_mid:
            left = mid + 1
        else:
            right = mid - 1

    return SearchResult(False, -1, comparisons)
