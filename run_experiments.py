# run_experiments.py

import random
import string
from typing import Tuple, List, Optional

import numpy as np

from index_prediction import MLIndexPredictor
from string_prediction import StringIndexPredictor
from hybrid_search import (
    SearchResult,
    classic_binary_search,
    interpolation_search_numeric,
    predictive_binary_search_ml,
)


def build_uniform_array(n: int) -> np.ndarray:
    """
    Uniform-ish numeric data.
    """
    arr = np.random.randint(0, n * 10, size=n)
    arr.sort()
    return arr


def build_skewed_array(n: int) -> np.ndarray:
    """
    Skewed numeric data, e.g. exponential-like distribution.
    """
    # Exponential distribution, then cast to int and sort
    raw = np.random.exponential(scale=1.0, size=n)
    arr = np.sort((raw * 10_000).astype(int))
    return arr


def run_trial_set(
    arr: np.ndarray,
    num_trials: int = 1_000,
    include_interpolation: bool = True,
    use_log: bool = False,
) -> Tuple[float, float, float]:
    """
    Run a batch of random search queries on the given array and
    return average comparison counts for:

        hybrid (ML + binary), classic binary, interpolation.

    If include_interpolation=False, interpolation average will be 0.
    
    Parameters
    ----------
    arr : np.ndarray
        Sorted array to search in.
    num_trials : int
        Number of search trials to run.
    include_interpolation : bool
        Whether to include interpolation search in results.
    use_log : bool
        Whether to use log transformation for ML predictor.
        Useful for exponential/skewed distributions.
    """
    values = arr.tolist()

    # Fit ML predictor with optional log transformation
    predictor = MLIndexPredictor(use_log=use_log)
    predictor.fit(values)

    hybrid_sum = 0.0
    classic_sum = 0.0
    interp_sum = 0.0

    for _ in range(num_trials):
        # 50%: existing element, 50%: random element in range (may or may not exist)
        if random.random() < 0.5:
            target = random.choice(values)
        else:
            target = random.randint(values[0], values[-1])

        # Hybrid ML + binary
        h_res: SearchResult = predictive_binary_search_ml(values, target, predictor)
        hybrid_sum += h_res.comparisons

        # Classic binary search
        c_res: SearchResult = classic_binary_search(values, target)
        classic_sum += c_res.comparisons

        # Interpolation search (optional)
        if include_interpolation:
            i_res: SearchResult = interpolation_search_numeric(values, target)
            interp_sum += i_res.comparisons

    hybrid_avg = hybrid_sum / num_trials
    classic_avg = classic_sum / num_trials
    interp_avg = interp_sum / num_trials if include_interpolation else 0.0

    return hybrid_avg, classic_avg, interp_avg


def main() -> None:
    np.random.seed(42)
    random.seed(42)

    # Test with different array sizes
    array_sizes = [50_000, 200_000]
    trials = 2_000

    for n in array_sizes:
        print(f"\n{'='*60}")
        print(f"Array size: {n}, trials per distribution: {trials}")
        print(f"{'='*60}")
        print("-" * 60)

        # 1) Uniform distribution
        uniform_arr = build_uniform_array(n)
        u_hybrid, u_classic, u_interp = run_trial_set(
            uniform_arr, trials, include_interpolation=True, use_log=False
        )

        print("Uniform distribution:")
        print(f"  Hybrid (ML + Binary)     avg comparisons: {u_hybrid:.2f}")
        print(f"  Classic Binary Search     avg comparisons: {u_classic:.2f}")
        print(f"  Interpolation Search      avg comparisons: {u_interp:.2f}")
        print("-" * 60)

        # 2) Skewed distribution - use log transform for better performance
        skewed_arr = build_skewed_array(n)
        s_hybrid, s_classic, s_interp = run_trial_set(
            skewed_arr, trials, include_interpolation=True, use_log=True
        )

        print("Skewed (exponential-like) distribution:")
        print(f"  Hybrid (ML + Binary)     avg comparisons: {s_hybrid:.2f}")
        print(f"  Classic Binary Search     avg comparisons: {s_classic:.2f}")
        print(f"  Interpolation Search      avg comparisons: {s_interp:.2f}")
        print("-" * 60)

        # 3) String dataset experiments
        # 3a) Synthetic random words
        run_string_experiment(n=n, trials=trials, max_prefix_length=3)
        
        # 3b) Real dictionary words (if available)
        run_real_dictionary_experiment(
            dict_path="/usr/share/dict/words",
            n=n,
            trials=trials,
            max_prefix_length=3
        )


def build_words_array(n: int = 50000) -> List[str]:
    """
    Build a sorted array of synthetic words.

    Generates random lowercase words of length 5-8 characters,
    then sorts them lexicographically.

    NOTE: This uses SYNTHETIC random words, which have uniform
    prefix distribution. Real dictionaries have highly non-uniform
    prefix distributions.

    Parameters
    ----------
    n : int
        Number of words to generate.

    Returns
    -------
    List[str]
        Sorted list of words.
    """
    words = []
    letters = string.ascii_lowercase
    
    for _ in range(n):
        # Random word length between 5 and 8
        word_length = random.randint(5, 8)
        # Generate random word
        word = ''.join(random.choice(letters) for _ in range(word_length))
        words.append(word)
    
    # Sort lexicographically
    words.sort()
    return words


def load_real_words(path: str = "/usr/share/dict/words", n: Optional[int] = None) -> List[str]:
    """
    Load words from a real dictionary file.

    Filters to lowercase alphabetic-only words and sorts them lexicographically.
    This provides a more realistic test case with non-uniform prefix distributions
    (e.g., common prefixes like "st", "co", "re" are much denser than rare ones).

    Parameters
    ----------
    path : str
        Path to dictionary file (one word per line).
        Default: /usr/share/dict/words (system dictionary on macOS/Linux).
    n : int | None
        Maximum number of words to load. If None, loads all words.

    Returns
    -------
    List[str]
        Sorted list of lowercase alphabetic words.

    Raises
    ------
    FileNotFoundError
        If the dictionary file does not exist.
    """
    words = []
    
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                word = line.strip().lower()
                # Filter: only alphabetic words, at least 2 characters
                if word.isalpha() and len(word) >= 2:
                    words.append(word)
                    if n is not None and len(words) >= n:
                        break
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dictionary file not found at {path}. "
            "Please provide a valid path to a word list file."
        )
    
    # Sort lexicographically
    words.sort()
    return words


def run_string_experiment(
    n: int = 50000,
    trials: int = 2000,
    max_prefix_length: int = 3,
) -> None:
    """
    Run string search experiment on SYNTHETIC random words.

    This experiment uses randomly generated words with uniform prefix distribution.
    Results may be more favorable to prefix-based predictors than real dictionaries.

    Parameters
    ----------
    n : int
        Number of words in the dataset.
    trials : int
        Number of search trials to run.
    max_prefix_length : int
        Maximum prefix length to use for prediction (1-4, default=3).
    """
    # Build sorted word list (synthetic random words)
    words = build_words_array(n)
    
    # Fit string predictor with improved features
    predictor = StringIndexPredictor(max_prefix_length=max_prefix_length, use_rmse=True)
    predictor.fit(words)
    
    hybrid_sum = 0.0
    classic_sum = 0.0
    
    for _ in range(trials):
        # Sample a random target word from the list (always existing element)
        target = random.choice(words)
        
        # Hybrid search with string predictor
        h_res: SearchResult = predictive_binary_search_ml(words, target, predictor)
        hybrid_sum += h_res.comparisons
        
        # Classic binary search
        c_res: SearchResult = classic_binary_search(words, target)
        classic_sum += c_res.comparisons
    
    hybrid_avg = hybrid_sum / trials
    classic_avg = classic_sum / trials
    
    print("Synthetic string dataset (random words):")
    print(f"  Hybrid (String predictor) avg comparisons: {hybrid_avg:.2f}")
    print(f"  Classic Binary Search     avg comparisons: {classic_avg:.2f}")
    print("-" * 60)


def run_real_dictionary_experiment(
    dict_path: str = "/usr/share/dict/words",
    n: Optional[int] = 50000,
    trials: int = 2000,
    max_prefix_length: int = 3,
) -> None:
    """
    Run string search experiment on REAL dictionary words.

    This experiment uses actual English words from a dictionary file, which have
    highly non-uniform prefix distributions (e.g., "st", "co", "re" are very dense,
    while others are rare). This provides a more realistic evaluation.

    Parameters
    ----------
    dict_path : str
        Path to dictionary file (one word per line).
        Default: /usr/share/dict/words (system dictionary on macOS/Linux).
    n : int | None
        Maximum number of words to load. If None, loads all words.
    trials : int
        Number of search trials to run.
    max_prefix_length : int
        Maximum prefix length to use for prediction (1-4, default=3).
    """
    try:
        # Load real dictionary words
        words = load_real_words(dict_path, n)
        
        if len(words) == 0:
            print(f"Warning: No words loaded from {dict_path}. Skipping real dictionary experiment.")
            return
        
        print(f"Loaded {len(words)} words from real dictionary")
        
        # Fit string predictor with improved features
        predictor = StringIndexPredictor(max_prefix_length=max_prefix_length, use_rmse=True)
        predictor.fit(words)
        
        hybrid_sum = 0.0
        classic_sum = 0.0
        
        for _ in range(trials):
            # Sample a random target word from the list (always existing element)
            target = random.choice(words)
            
            # Hybrid search with string predictor
            h_res: SearchResult = predictive_binary_search_ml(words, target, predictor)
            hybrid_sum += h_res.comparisons
            
            # Classic binary search
            c_res: SearchResult = classic_binary_search(words, target)
            classic_sum += c_res.comparisons
        
        hybrid_avg = hybrid_sum / trials
        classic_avg = classic_sum / trials
        
        print("Real dictionary dataset:")
        print(f"  Hybrid (String predictor) avg comparisons: {hybrid_avg:.2f}")
        print(f"  Classic Binary Search     avg comparisons: {classic_avg:.2f}")
        print("-" * 60)
        
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Skipping real dictionary experiment.")
        print("-" * 60)


if __name__ == "__main__":
    main()
