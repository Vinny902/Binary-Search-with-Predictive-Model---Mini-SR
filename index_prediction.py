# index_prediction.py

from typing import List
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


class MLIndexPredictor:
    """
    An improved predictor that adapts to data distribution:
    - Uses polynomial features for non-linear distributions
    - Detects distribution type and selects the best model
    - Uses quantile-based approach for heavily skewed data
    - Supports log transformation for exponential/skewed distributions
    - Computes RMSE for error-based search window optimization
    """

    def __init__(self, use_log: bool = False) -> None:
        """
        Initialize the ML index predictor.

        Parameters
        ----------
        use_log : bool, default=False
            If True, apply log1p transformation to values before training.
            Useful for exponential/skewed distributions where the relationship
            between value and index is nonlinear in the original space.
        """
        self.use_log = use_log
        self.linear_model = LinearRegression()
        self.poly_model = Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('linear', LinearRegression())
        ])
        self._arr_length: int = 0
        self._is_fitted: bool = False
        self._arr: np.ndarray = None
        self._use_quantile: bool = False
        self._best_model = None
        self._quantile_values: np.ndarray = None
        self._quantile_indices: np.ndarray = None
        self.rmse: float = 0.0  # Root Mean Squared Error after fitting

    def fit(self, arr: List[float]) -> None:
        """
        Fit the predictor with adaptive model selection.

        Parameters
        ----------
        arr : list[float]
            Sorted list/array of numeric values.
        """
        if len(arr) == 0:
            raise ValueError("Cannot fit MLIndexPredictor on an empty array.")

        self._arr_length = len(arr)
        self._arr = np.array(arr, dtype=float)
        
        # Apply transformation if needed
        values = self._transform(self._arr.reshape(-1, 1))
        indices = np.arange(len(arr), dtype=float)

        # Detect distribution type
        # Check uniformity through coefficient of variation
        if len(arr) > 100:
            # Use quantile-based approach for heavily skewed data
            # Check how much the distribution differs from uniform
            sorted_values = np.sort(self._arr)
            value_range = sorted_values[-1] - sorted_values[0]
            
            # If value range is very large and distribution is non-uniform,
            # use quantile search
            num_quantiles = min(100, len(arr))
            percentiles = np.linspace(0, 100, num_quantiles)
            quantile_values = np.percentile(sorted_values, percentiles)
            
            # Check how uniformly values are distributed
            # by examining differences between adjacent quantiles
            quantile_diffs = np.diff(quantile_values)
            mean_diff = np.mean(quantile_diffs)
            std_diff = np.std(quantile_diffs)
            cv = std_diff / (mean_diff + 1e-10) if mean_diff > 0 else 1.0
            
            # If coefficient of variation is large, use quantile approach
            if cv > 0.5:  # Threshold for detecting non-uniformity
                self._use_quantile = True
                self._quantile_values = quantile_values
                self._quantile_indices = np.linspace(0, len(arr) - 1, len(quantile_values))
            else:
                self._use_quantile = False
                # Train both models and select the best one
                self.linear_model.fit(values, indices)
                self.poly_model.fit(values, indices)
                
                # Select model with lower error on validation
                linear_pred = self.linear_model.predict(values)
                poly_pred = self.poly_model.predict(values)
                
                linear_error = np.mean(np.abs(linear_pred - indices))
                poly_error = np.mean(np.abs(poly_pred - indices))
                
                self._best_model = self.linear_model if linear_error < poly_error else self.poly_model
        else:
            self._use_quantile = False
            self.linear_model.fit(values, indices)
            self._best_model = self.linear_model

        # Compute RMSE for the best model (if not using quantile approach)
        if not self._use_quantile and self._best_model is not None:
            # Transform values if using log
            transformed_values = self._transform(self._arr.reshape(-1, 1))
            predicted_indices = self._best_model.predict(transformed_values)
            actual_indices = np.arange(len(arr), dtype=float)
            self.rmse = np.sqrt(np.mean((predicted_indices - actual_indices) ** 2))
        else:
            # For quantile approach, estimate RMSE based on quantile spacing
            if self._use_quantile and len(self._quantile_indices) > 1:
                # Estimate RMSE as average spacing between quantiles
                quantile_spacing = np.mean(np.diff(self._quantile_indices))
                self.rmse = quantile_spacing * 0.5  # Conservative estimate
            else:
                self.rmse = 0.0

        self._is_fitted = True

    def _transform(self, values: np.ndarray) -> np.ndarray:
        """
        Apply transformation to values (log1p if use_log is True).

        Parameters
        ----------
        values : np.ndarray
            Input values to transform.

        Returns
        -------
        np.ndarray
            Transformed values.
        """
        if self.use_log:
            return np.log1p(np.maximum(values, 0))  # log1p(x) = log(1 + x), handles 0 safely
        return values

    def predict(self, target: float) -> int:
        """
        Predict an index using the best model.

        Returns an integer index in [0, len(arr) - 1].
        """
        if not self._is_fitted:
            raise RuntimeError("MLIndexPredictor must be fitted before calling predict().")

        if self._use_quantile:
            # Quantile search for non-uniform distributions
            # Find the nearest quantile
            idx = np.searchsorted(self._quantile_values, target, side='left')
            idx = min(max(0, idx), len(self._quantile_indices) - 1)
            
            # Interpolate between quantiles
            if idx == 0:
                pred = self._quantile_indices[0]
            elif idx >= len(self._quantile_indices) - 1:
                pred = self._quantile_indices[-1]
            else:
                # Linear interpolation between quantiles
                q1, q2 = self._quantile_values[idx - 1], self._quantile_values[idx]
                i1, i2 = self._quantile_indices[idx - 1], self._quantile_indices[idx]
                
                if q2 != q1:
                    ratio = (target - q1) / (q2 - q1)
                    pred = i1 + ratio * (i2 - i1)
                else:
                    pred = i1
        else:
            # Use selected model (linear or polynomial)
            # Apply transformation to target if needed
            transformed_target = self._transform(np.array([[float(target)]]))
            pred = self._best_model.predict(transformed_target)[0]

        # Clamp to valid range
        pred_clamped = max(0.0, min(pred, float(self._arr_length - 1)))
        return int(pred_clamped)
