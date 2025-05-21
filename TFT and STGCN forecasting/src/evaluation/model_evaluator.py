from ..utils.logger import StructuredLogger
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import r2_score
import time
from datetime import datetime
import json
import os
from pathlib import Path

class ModelEvaluator:
    """Class for evaluating demand forecasting models performance.
    
    This class provides comprehensive evaluation metrics and visualization 
    tools for assessing the performance of demand forecasting models.
    
    Attributes:
        logger: Logger instance for logging evaluation information
        output_dir: Directory to save evaluation results
        metrics: Dictionary to store evaluation metrics
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the ModelEvaluator.
        
        Args:
            output_dir: Directory to save evaluation results and visualizations
        """
        self.logger = StructuredLogger("model_evaluator")
        self.output_dir = output_dir
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.metrics = {}
        self.logger.info("ModelEvaluator initialized")
    
    def evaluate_model(self, model_name: str, y_true: Union[np.ndarray, pd.Series], 
                      y_pred: Union[np.ndarray, pd.Series], 
                      uncertainty: Optional[Union[np.ndarray, pd.Series]] = None,
                      groupby: Optional[pd.Series] = None,
                      timestamps: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Evaluate a model's predictions against ground truth values.
        
        Args:
            model_name: Name of the model being evaluated
            y_true: Ground truth values
            y_pred: Predicted values
            uncertainty: Prediction uncertainty (standard deviation or confidence intervals)
            groupby: Optional series to group results by (e.g., location, hour of day)
            timestamps: Optional time series for temporal evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info(f"Evaluating model: {model_name}")
        
        # Convert inputs to numpy arrays for consistency
        y_true_array = np.array(y_true)
        y_pred_array = np.array(y_pred)
        
        # Calculate basic regression metrics
        metrics = {
            "model_name": model_name,
            "n_samples": len(y_true),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "mape": self._calculate_mape(y_true, y_pred),  # Custom MAPE to handle zeros
            "r2": r2_score(y_true, y_pred),
            "bias": np.mean(y_pred_array - y_true_array),
            "mean_forecast": np.mean(y_pred),
            "std_forecast": np.std(y_pred),
            "mean_actual": np.mean(y_true),
            "std_actual": np.std(y_true)
        }
        
        # Normalized metrics
        metrics["nrmse"] = metrics["rmse"] / (metrics["mean_actual"] if metrics["mean_actual"] > 0 else 1)
        
        # Add uncertainty evaluation if provided
        if uncertainty is not None:
            metrics.update(self._evaluate_uncertainty(y_true_array, y_pred_array, uncertainty))
        
        # Evaluate performance by groups if provided
        if groupby is not None:
            group_metrics = self._evaluate_by_group(y_true, y_pred, groupby)
            metrics["group_metrics"] = group_metrics
        
        # Evaluate temporal patterns if timestamps provided
        if timestamps is not None:
            temporal_metrics = self._evaluate_temporal_patterns(y_true, y_pred, timestamps)
            metrics["temporal_metrics"] = temporal_metrics
        
        # Store metrics for this model
        self.metrics[model_name] = metrics
        
        return metrics
    
    def _calculate_mape(self, y_true: Union[np.ndarray, pd.Series], 
                       y_pred: Union[np.ndarray, pd.Series]) -> float:
        """Calculate Mean Absolute Percentage Error with handling for zero values.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            MAPE value
        """
        # Convert to numpy arrays
        y_true_array = np.array(y_true)
        y_pred_array = np.array(y_pred)
        
        # Handle zero values in y_true to avoid division by zero
        # Only calculate MAPE for non-zero true values
        non_zero_mask = y_true_array != 0
        
        if not np.any(non_zero_mask):
            return np.nan  # All true values are zero
        
        # Calculate MAPE only on non-zero values
        return np.mean(np.abs((y_true_array[non_zero_mask] - y_pred_array[non_zero_mask]) / 
                             y_true_array[non_zero_mask])) * 100
    
    def _evaluate_uncertainty(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            uncertainty: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """Evaluate uncertainty estimates.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            uncertainty: Prediction uncertainty (std or confidence intervals)
            
        Returns:
            Dictionary of uncertainty evaluation metrics
        """
        # Convert to numpy array
        uncertainty_array = np.array(uncertainty)
        
        # Assuming uncertainty_array represents standard deviation
        # Calculate prediction intervals (68% confidence interval for normal distribution)
        lower_bound = y_pred - uncertainty_array
        upper_bound = y_pred + uncertainty_array
        
        # Calculate coverage (percentage of true values within the prediction interval)
        coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound)) * 100
        
        # Calculate interval width
        interval_width = np.mean(upper_bound - lower_bound)
        
        # Calculate calibration score (how well the uncertainty matches the actual error)
        actual_errors = np.abs(y_pred - y_true)
        calibration_score = np.mean(actual_errors / uncertainty_array) if np.all(uncertainty_array > 0) else np.nan
        
        return {
            "uncertainty_coverage": coverage,
            "uncertainty_width": interval_width,
            "uncertainty_calibration": calibration_score,
            "mean_uncertainty": np.mean(uncertainty_array),
            "uncertainty_ratio": np.mean(uncertainty_array) / np.mean(y_pred) if np.mean(y_pred) > 0 else np.nan
        }
    
    def _evaluate_by_group(self, y_true: Union[np.ndarray, pd.Series], 
                         y_pred: Union[np.ndarray, pd.Series],
                         groupby: pd.Series) -> Dict[str, Dict[str, float]]:
        """Evaluate model performance by groups.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            groupby: Series to group results by
            
        Returns:
            Dictionary of group-level metrics
        """
        # Combine data into a DataFrame
        eval_df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
            'group': groupby
        })
        
        group_metrics = {}
        
        # Calculate metrics for each group
        for group_name, group_data in eval_df.groupby('group'):
            group_metrics[str(group_name)] = {
                "rmse": np.sqrt(mean_squared_error(group_data['y_true'], group_data['y_pred'])),
                "mae": mean_absolute_error(group_data['y_true'], group_data['y_pred']),
                "mape": self._calculate_mape(group_data['y_true'], group_data['y_pred']),
                "bias": np.mean(group_data['y_pred'] - group_data['y_true']),
                "n_samples": len(group_data),
                "mean_actual": group_data['y_true'].mean(),
                "mean_forecast": group_data['y_pred'].mean()
            }
        
        return group_metrics
    
    def _evaluate_temporal_patterns(self, y_true: Union[np.ndarray, pd.Series], 
                                  y_pred: Union[np.ndarray, pd.Series],
                                  timestamps: pd.Series) -> Dict[str, Any]:
        """Evaluate temporal patterns in forecast errors.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            timestamps: Time series for temporal evaluation
            
        Returns:
            Dictionary of temporal evaluation metrics
        """
        # Combine data into a DataFrame
        eval_df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
            'timestamp': timestamps,
            'error': np.array(y_pred) - np.array(y_true),
            'abs_error': np.abs(np.array(y_pred) - np.array(y_true))
        })
        
        # Extract time components
        eval_df['hour'] = pd.to_datetime(eval_df['timestamp']).dt.hour
        eval_df['day_of_week'] = pd.to_datetime(eval_df['timestamp']).dt.dayofweek
        
        # Hourly metrics
        hourly_metrics = self._calculate_temporal_group_metrics(eval_df, 'hour')
        
        # Day of week metrics
        dow_metrics = self._calculate_temporal_group_metrics(eval_df, 'day_of_week')
        
        # Check for autocorrelation in errors
        error_autocorr = self._calculate_autocorrelation(eval_df['error'], 24)  # 24-hour lag
        
        return {
            "hourly_metrics": hourly_metrics,
            "day_of_week_metrics": dow_metrics,
            "error_autocorrelation": error_autocorr,
            "trend_strength": self._calculate_trend_strength(eval_df)
        }
    
    def _calculate_temporal_group_metrics(self, eval_df: pd.DataFrame, 
                                        group_col: str) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for temporal groups (hour, day of week, etc.).
        
        Args:
            eval_df: DataFrame with evaluation data
            group_col: Column name to group by
            
        Returns:
            Dictionary of metrics by group
        """
        group_metrics = {}
        
        for group_name, group_data in eval_df.groupby(group_col):
            group_metrics[str(group_name)] = {
                "rmse": np.sqrt(mean_squared_error(group_data['y_true'], group_data['y_pred'])),
                "mae": mean_absolute_error(group_data['y_true'], group_data['y_pred']),
                "bias": group_data['error'].mean(),
                "n_samples": len(group_data),
                "mean_actual": group_data['y_true'].mean(),
                "mean_forecast": group_data['y_pred'].mean()
            }
        
        return group_metrics
    
    def _calculate_autocorrelation(self, series: pd.Series, lag: int) -> float:
        """Calculate autocorrelation at specified lag.
        
        Args:
            series: Time series to analyze
            lag: Lag period for autocorrelation
            
        Returns:
            Autocorrelation value
        """
        # Convert to numpy array for consistency
        data = np.array(series)
        n = len(data)
        
        if n <= lag:
            return np.nan
        
        # Calculate mean and variance
        mean = np.mean(data)
        var = np.var(data)
        
        if var == 0:
            return np.nan
        
        # Calculate autocorrelation
        acf = np.sum((data[lag:] - mean) * (data[:n-lag] - mean)) / ((n - lag) * var)
        
        return acf
    
    def _calculate_trend_strength(self, eval_df: pd.DataFrame) -> float:
        """Calculate trend strength in errors over time.
        
        Args:
            eval_df: DataFrame with evaluation data including timestamps and errors
            
        Returns:
            Trend strength metric
        """
        # Ensure timestamps are sorted
        sorted_df = eval_df.sort_values('timestamp')
        
        # Calculate correlation between time and error
        time_numeric = pd.to_datetime(sorted_df['timestamp']).astype(np.int64)
        correlation = np.corrcoef(time_numeric, sorted_df['error'])[0, 1]
        
        return correlation
    
    def compare_models(self, model_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Compare performance metrics across multiple models.
        
        Args:
            model_names: List of model names to compare. If None, compare all evaluated models.
            
        Returns:
            DataFrame with comparison metrics
        """
        if not self.metrics:
            self.logger.warning("No models have been evaluated yet")
            return pd.DataFrame()
        
        if model_names is None:
            model_names = list(self.metrics.keys())
        
        # Extract top-level metrics for each model
        comparison_data = []
        for model_name in model_names:
            if model_name not in self.metrics:
                self.logger.warning(f"Model '{model_name}' has not been evaluated")
                continue
            
            # Extract metrics, excluding nested dictionaries
            model_metrics = {k: v for k, v in self.metrics[model_name].items()
                           if not isinstance(v, dict)}
            comparison_data.append(model_metrics)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by RMSE (ascending)
        if 'rmse' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('rmse')
        
        return comparison_df
    
    def evaluate_feature_importance(self, model, feature_names: List[str], 
                                  n_features: int = 20) -> pd.DataFrame:
        """Evaluate feature importance for a trained model.
        
        Args:
            model: Trained model with feature_importance_ attribute
            feature_names: List of feature names
            n_features: Number of top features to return
            
        Returns:
            DataFrame with feature importances
        """
        try:
            # Extract feature importance (different approaches for different model types)
            if hasattr(model, 'feature_importances_'):  # For tree-based models
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):  # For linear models
                importances = np.abs(model.coef_)
                if importances.ndim > 1:  # For multi-output models
                    importances = np.mean(importances, axis=0)
            elif hasattr(model, 'get_feature_importance'):  # For LightGBM-like models
                importances = model.get_feature_importance()
            else:
                self.logger.warning("Model does not support direct feature importance extraction")
                return pd.DataFrame()
            
            # Create DataFrame with feature importances
            importance_df = pd.DataFrame({
                'feature': feature_names[:len(importances)],
                'importance': importances
            })
            
            # Sort by importance (descending)
            importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
            
            # Return top N features
            return importance_df.head(n_features)
            
        except Exception as e:
            self.logger.error(f"Error extracting feature importance: {str(e)}")
            return pd.DataFrame()
    
    def calculate_prediction_intervals(self, y_true: Union[np.ndarray, pd.Series], 
                                     y_pred: Union[np.ndarray, pd.Series], 
                                     alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate empirical prediction intervals based on observed errors.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            alpha: Significance level (e.g., 0.05 for 95% confidence interval)
            
        Returns:
            Tuple of lower and upper bounds for prediction intervals
        """
        # Calculate prediction errors
        errors = np.array(y_true) - np.array(y_pred)
        
        # Calculate error percentiles
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        # Calculate error bounds
        lower_error = np.percentile(errors, lower_percentile)
        upper_error = np.percentile(errors, upper_percentile)
        
        # Calculate prediction intervals
        lower_bound = np.array(y_pred) + lower_error
        upper_bound = np.array(y_pred) + upper_error
        
        return lower_bound, upper_bound
    
    def save_evaluation_results(self, filename: Optional[str] = None):
        """Save evaluation results to a JSON file.
        
        Args:
            filename: Name of the file to save results to. If None, uses timestamp.
        """
        if not self.metrics:
            self.logger.warning("No models have been evaluated yet")
            return
        
        if not self.output_dir:
            self.logger.warning("No output directory specified for saving results")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_evaluation_{timestamp}.json"
        
        output_path = os.path.join(self.output_dir, filename)
        
        # Convert numpy values to native Python types for JSON serialization
        serializable_metrics = self._make_json_serializable(self.metrics)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
        
        self.logger.info(f"Evaluation results saved to {output_path}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._make_json_serializable(obj.tolist())
        elif isinstance(obj, pd.DataFrame):
            return self._make_json_serializable(obj.to_dict(orient='records'))
        elif isinstance(obj, pd.Series):
            return self._make_json_serializable(obj.tolist())
        elif obj is np.nan:
            return None
        else:
            return obj
