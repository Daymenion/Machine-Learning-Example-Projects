import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple, Any
import time
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, KFold, GridSearchCV
import joblib
import os

from .base_model import BaseModel
from .gbdt_model import LightGBMModel
from .optimized_tft_model import OptimizedTFTModel
from .optimized_stgcn_model import OptimizedSTGCNModel
from ..utils.logger import StructuredLogger 


class EnsembleModel(BaseModel):
    """Advanced ensemble model for demand forecasting.
    
    This model combines multiple state-of-the-art forecasting models 
    (LightGBM, Temporal Fusion Transformer, and Spatio-Temporal GNN)
    using a meta-learner approach. The ensemble leverages the strengths
    of each base model to produce more accurate and robust predictions.
    
    Key features:
    - Stacked ensemble with cross-validation to prevent data leakage
    - Meta-learner optimization for optimal weighting
    - Location-specific and time-specific model weighting
    - Uncertainty quantification from diverse base models
    - Feature-based model selection for adaptive ensemble
    """
    
    def __init__(self, model_config: Dict, logger: Optional[StructuredLogger] = None):
        """Initialize the ensemble model.
        
        Args:
            model_config: Configuration dictionary for the model
            logger: Logger instance for tracking model operations
        """
        super().__init__(model_config, logger)
        
        # Extract ensemble-specific configurations
        self.ensemble_params = self.model_config.get("params", {})
        
        # Set default hyperparameters if not specified
        self._set_default_params()
        
        # Base models
        self.base_models = {}
        
        # Handle the case where base_models contains actual model instances (from runner.py)
        # or model configurations (from config file)
        base_models_config = self.model_config.get("base_models", {})
        
        if base_models_config and isinstance(next(iter(base_models_config.values())), BaseModel):
            # If we have actual model instances
            self.logger.info("Using provided model instances for ensemble")
            self.base_models = base_models_config
            self.base_model_configs = {}
            
            # Extract configs from the model instances if available
            for model_name, model in self.base_models.items():
                if hasattr(model, 'model_config'):
                    self.base_model_configs[model_name] = model.model_config
                else:
                    self.base_model_configs[model_name] = {}
        else:
            # Traditional case where we just have model configs
            self.base_model_configs = base_models_config
        
        # Meta-learner model
        self.meta_learner = None
        self.meta_features = None
        
        # Cross-validation settings
        self.cv_method = self.ensemble_params.get("cv_method", "time_series")
        self.cv_folds = self.ensemble_params.get("cv_folds", 5)
        
        # Storing cross-validation predictions
        self.cv_predictions = {}
        self.oof_predictions = None  # Out-of-fold predictions
        
        # Feature importance and explainability
        self.model_weights = None
        self.global_feature_importance = None
        
        # Uncertainty estimates
        self.prediction_intervals = None
    
    def _set_default_params(self) -> None:
        """Set default hyperparameters if not specified in the config."""
        # Specify optimal defaults for ensemble
        defaults = {
            "meta_learner": "ridge",  # Options: ridge, lasso, elastic_net, gbm, rf
            "use_model_features": True,  # Whether to use model predictions as features
            "use_context_features": True,  # Whether to use context-related features
            "use_location_specific_weights": True,  # Location-specific model weights
            "use_time_specific_weights": True,  # Time-specific model weights (e.g., hour of day)
            "cv_method": "time_series",  # Options: time_series, kfold
            "cv_folds": 5,
            "include_original_features": False,  # Whether to include original features in meta-learner
            "meta_learner_alpha": 1.0,  # Regularization parameter for meta-learner
            "meta_learner_l1_ratio": 0.5,  # L1 ratio for elastic net
        }
        
        # Update default params with user-specified params
        for key, value in defaults.items():
            if key not in self.ensemble_params:
                self.ensemble_params[key] = value
    
    def _build_base_models(self) -> None:
        """Build the base models for the ensemble."""
        self.logger.info("Building base models for ensemble")
        
        # Initialize base models if configs are provided
        for model_name, model_config in self.base_model_configs.items():
            if model_name == "lightgbm":
                self.base_models[model_name] = LightGBMModel(model_config, self.logger)
            elif model_name == "tft":
                self.base_models[model_name] = OptimizedTFTModel(model_config, self.logger)
            elif model_name == "stgcn":
                self.base_models[model_name] = OptimizedSTGCNModel(model_config, self.logger)
            else:
                self.logger.warning(f"Unknown model type: {model_name}, skipping")
        
        if not self.base_models:
            self.logger.warning("No base models specified, using default LightGBM")
            # Create a default LightGBM model if none specified
            self.base_models["lightgbm"] = LightGBMModel({}, self.logger)
    
    def _build_meta_learner(self, n_features: int) -> Any:
        """Build the meta-learner model.
        
        Args:
            n_features: Number of input features for the meta-learner
            
        Returns:
            Meta-learner model instance
        """
        meta_learner_type = self.ensemble_params.get("meta_learner", "ridge")
        
        if meta_learner_type == "ridge":
            meta_learner = Ridge(
                alpha=self.ensemble_params.get("meta_learner_alpha", 1.0),
                random_state=42
            )
        elif meta_learner_type == "lasso":
            meta_learner = Lasso(
                alpha=self.ensemble_params.get("meta_learner_alpha", 1.0),
                random_state=42
            )
        elif meta_learner_type == "elastic_net":
            meta_learner = ElasticNet(
                alpha=self.ensemble_params.get("meta_learner_alpha", 1.0),
                l1_ratio=self.ensemble_params.get("meta_learner_l1_ratio", 0.5),
                random_state=42
            )
        elif meta_learner_type == "rf":
            meta_learner = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif meta_learner_type == "gbm":
            meta_learner = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            )
        else:
            self.logger.warning(f"Unknown meta-learner type: {meta_learner_type}, using ridge")
            meta_learner = Ridge(alpha=1.0, random_state=42)
        
        return meta_learner
    
    def _get_cv_splitter(self, X: pd.DataFrame) -> Union[TimeSeriesSplit, KFold]:
        """Get the appropriate cross-validation splitter.
        
        Args:
            X: Feature dataframe
            
        Returns:
            CV splitter instance
        """
        if self.cv_method == "time_series":
            return TimeSeriesSplit(n_splits=self.cv_folds)
        else:  # kfold
            return KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
    
    def _prepare_meta_features(self, X: pd.DataFrame, base_model_preds: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Prepare features for the meta-learner.
        
        Args:
            X: Original feature dataframe
            base_model_preds: Dictionary of base model predictions
            
        Returns:
            DataFrame with meta-learner features
        """
        # Ensure all predictions have the same length as X
        for model_name, preds in base_model_preds.items():
            if len(preds) != len(X):
                self.logger.warning(f"Prediction length mismatch for {model_name}: {len(preds)} vs {len(X)}")
                # If prediction array is shorter, pad with mean values
                if len(preds) < len(X):
                    # Pad with mean value
                    mean_val = np.mean(preds)
                    self.logger.info(f"Padding {model_name} predictions with mean value {mean_val:.4f}")
                    pad_length = len(X) - len(preds)
                    base_model_preds[model_name] = np.append(preds, np.full(pad_length, mean_val))
                # If prediction array is longer, truncate
                else:
                    self.logger.info(f"Truncating {model_name} predictions from {len(preds)} to {len(X)}")
                    base_model_preds[model_name] = preds[:len(X)]
        
        # Start with base model predictions
        meta_features = pd.DataFrame(index=range(len(X)))
        
        # First, handle NaN values in the original data
        X_clean = X.copy()
        # Fill NaN values with column means, or zeros for any column that's all NaN
        for col in X_clean.columns:
            if X_clean[col].isna().any():
                self.logger.warning(f"Found NaN values in column {col}, filling with mean or zero")
                if X_clean[col].isna().all():
                    X_clean[col] = 0  # If all values are NaN, fill with zero
                else:
                    X_clean[col] = X_clean[col].fillna(X_clean[col].mean())  # Otherwise use column mean
        
        # Add base model predictions as features
        if self.ensemble_params.get("use_model_features", True):
            for model_name, preds in base_model_preds.items():
                # Handle NaN values in model predictions
                clean_preds = np.array(preds)
                if np.isnan(clean_preds).any():
                    self.logger.warning(f"Found NaN values in {model_name} predictions, filling with mean or zero")
                    if np.isnan(clean_preds).all():
                        clean_preds = np.zeros_like(clean_preds)
                    else:
                        # Replace NaNs with mean of non-NaN values
                        mean_val = np.nanmean(clean_preds)
                        clean_preds = np.where(np.isnan(clean_preds), mean_val, clean_preds)
                meta_features[f"pred_{model_name}"] = clean_preds
        
        # Optionally add original features
        if self.ensemble_params.get("include_original_features", False):
            for col in X.columns:
                if col not in meta_features.columns:
                    # Use the cleaned data without NaNs
                    meta_features[col] = X_clean[col].values
        
        # Add location-specific features if enabled
        if self.ensemble_params.get("use_location_specific_weights", True) and "start_location_id" in X_clean.columns:
            # Convert to strings first to ensure clean integer values
            dummies = pd.get_dummies(X_clean["start_location_id"].astype(str), prefix="loc")
            # Add columns to meta_features with index alignment
            for col in dummies.columns:
                meta_features[col] = dummies[col].values
        
        # Add time-specific features if enabled
        if self.ensemble_params.get("use_time_specific_weights", True) and "hour_start" in X_clean.columns:
            try:
                # Extract hour of day
                hour_of_day = pd.to_datetime(X_clean["hour_start"]).dt.hour
                hour_dummies = pd.get_dummies(hour_of_day, prefix="hour")
                for col in hour_dummies.columns:
                    meta_features[col] = hour_dummies[col].values
                
                # Extract day of week
                day_of_week = pd.to_datetime(X_clean["hour_start"]).dt.dayofweek
                day_dummies = pd.get_dummies(day_of_week, prefix="day")
                for col in day_dummies.columns:
                    meta_features[col] = day_dummies[col].values
            except Exception as e:
                self.logger.warning(f"Error creating time features: {str(e)}. Skipping time features.")
        
        
        # Add context features if enabled
        if self.ensemble_params.get("use_context_features", True):
            context_cols = [
                col for col in X_clean.columns 
                if any(prefix in col for prefix in ["event_", "weather_", "econ_", "holiday_", "temp_", "heat_index", "wind_chill"])
            ]
            
            for col in context_cols:
                try:
                    # Ensure values are numeric
                    meta_features[col] = pd.to_numeric(X_clean[col], errors='coerce').fillna(0).values
                except Exception as e:
                    self.logger.warning(f"Error adding context feature {col}: {str(e)}. Using zeros instead.")
                    meta_features[col] = np.zeros(len(X_clean))
        
        return meta_features
    
    def fit(self, X: pd.DataFrame, y: pd.Series, validation_data: Optional[Tuple] = None) -> Dict:
        """Train the ensemble model on the provided data.
        
        Args:
            X: Feature dataframe for training
            y: Target series for training
            validation_data: Optional tuple of (X_val, y_val) for validation during training
            
        Returns:
            Dictionary of training metrics and information
        """
        self.logger.info(f"Training ensemble model with {X.shape[1]} features and {X.shape[0]} samples")
        start_time = time.time()
        
        # Save feature names for later use
        self.feature_names = list(X.columns)
        self.model_info["feature_count"] = len(self.feature_names)
        
        # Build base models if not already done
        if not self.base_models:
            self._build_base_models()
        
        # Create cross-validation splits
        cv = self._get_cv_splitter(X)
        
        # Dictionary to store out-of-fold predictions
        oof_preds = {model_name: np.zeros(X.shape[0]) for model_name in self.base_models.keys()}
        
        # Perform cross-validation for each base model
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
            self.logger.info(f"Training fold {fold_idx+1}/{self.cv_folds}")
            
            # Split data for this fold
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train and predict with each base model
            for model_name, model in self.base_models.items():
                # Skip models that aren't fitted or are missing
                if model is None:
                    self.logger.warning(f"Skipping {model_name} as it is not available")
                    continue
                    
                self.logger.info(f"Training {model_name} on fold {fold_idx+1}")
                
                try:
                    # Create a new instance for each fold to avoid data leakage
                    if hasattr(model, 'model_config'):
                        model_config = model.model_config
                    else:
                        model_config = self.base_model_configs.get(model_name, {})
                        
                    # Create the appropriate model type
                    if model_name == "lightgbm":
                        fold_model = LightGBMModel(model_config, self.logger)
                    elif model_name == "tft":
                        fold_model = OptimizedTFTModel(model_config, self.logger)
                    elif model_name == "stgcn":
                        fold_model = OptimizedSTGCNModel(model_config, self.logger)
                    else:
                        # Use type(model) as a fallback
                        fold_model = type(model)(model_config, self.logger)
                    
                    # Train the model
                    fold_model.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold))
                    
                    # Generate out-of-fold predictions
                    oof_preds[model_name][val_idx] = fold_model.predict(X_val_fold)
                    
                    # Store the model if it's the last fold (we'll use it for new predictions)
                    if fold_idx == list(cv.split(X))[-1][0][0]:
                        self.base_models[model_name] = fold_model
                except Exception as e:
                    self.logger.error(f"Error training {model_name} on fold {fold_idx+1}: {str(e)}")
                    # Set null predictions for this model to avoid affecting the ensemble
                    oof_preds[model_name][val_idx] = np.nan
        
        # Prepare meta-features for training the meta-learner
        meta_features = self._prepare_meta_features(X, oof_preds)
        self.meta_features = meta_features.columns.tolist()
        
        # Initialize and train the meta-learner
        self.meta_learner = self._build_meta_learner(meta_features.shape[1])
        self.meta_learner.fit(meta_features, y)
        
        # Store model weights (if applicable)
        if hasattr(self.meta_learner, 'coef_'):
            self.model_weights = {}
            for i, feature in enumerate(meta_features.columns):
                self.model_weights[feature] = self.meta_learner.coef_[i]
        
        # Record training time
        self.training_time = time.time() - start_time
        self.logger.info(f"Ensemble model training completed in {self.training_time:.2f} seconds")
        
        # Mark model as fitted
        self.is_fitted = True
        self.model_info["training_status"] = "trained"
        self.model_info["training_time"] = self.training_time
        
        # Calculate ensemble predictions on the training set
        y_pred = self.predict(X)
        
        # Compute metrics
        self.metrics = {
            "MAE": mean_absolute_error(y, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
            "R2": r2_score(y, y_pred),
            "MAPE": np.mean(np.abs((y - y_pred) / (y + 1e-5))) * 100
        }
        
        self.logger.info(f"Training metrics: {self.metrics}")
        
        # If validation data was provided, compute metrics on it
        if validation_data is not None:
            X_val, y_val = validation_data
            y_val_pred = self.predict(X_val)
            
            val_metrics = {
                "val_MAE": mean_absolute_error(y_val, y_val_pred),
                "val_RMSE": np.sqrt(mean_squared_error(y_val, y_val_pred)),
                "val_R2": r2_score(y_val, y_val_pred),
                "val_MAPE": np.mean(np.abs((y_val - y_val_pred) / (y_val + 1e-5))) * 100
            }
            
            self.metrics.update(val_metrics)
            self.logger.info(f"Validation metrics: {val_metrics}")
        
        # Calculate feature importance via model weights
        self._calculate_feature_importance()
        
        return {
            "training_time": self.training_time,
            "metrics": self.metrics,
            "feature_importance": self.feature_importance.to_dict(orient='records') if self.feature_importance is not None else None,
            "model_info": self.model_info
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate point predictions for the input data.
        
        Args:
            X: Feature dataframe for prediction
            
        Returns:
            Array of predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet")
            
        self.logger.info(f"Generating ensemble predictions for {X.shape[0]} samples")
        
        # Generate predictions from base models
        base_preds = {}
        for model_name, model in self.base_models.items():
            self.logger.info(f"Generating predictions with {model_name}")
            base_preds[model_name] = model.predict(X)
        
        # Prepare meta-features for prediction
        meta_features = self._prepare_meta_features(X, base_preds)
        
        # Ensure meta_features has all columns used during training
        for col in self.meta_features:
            if col not in meta_features.columns:
                meta_features[col] = 0
        
        # Make sure columns are in the same order as during training
        meta_features = meta_features[self.meta_features]
        
        # Generate ensemble predictions
        ensemble_preds = self.meta_learner.predict(meta_features)
        
        return ensemble_preds
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate predictions with uncertainty intervals.
        
        Args:
            X: Feature dataframe for prediction
            
        Returns:
            Tuple of (mean predictions, lower bounds, upper bounds)
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet")
            
        self.logger.info(f"Generating ensemble predictions with uncertainty for {X.shape[0]} samples")
        
        # Get base model predictions with uncertainty
        base_preds = {}
        lower_bounds = {}
        upper_bounds = {}
        
        for model_name, model in self.base_models.items():
            try:
                mean, lower, upper = model.predict_with_uncertainty(X)
                base_preds[model_name] = mean
                lower_bounds[model_name] = lower
                upper_bounds[model_name] = upper
            except Exception as e:
                self.logger.warning(f"Error getting uncertainty from {model_name}: {str(e)}")
                # Fallback to point predictions
                base_preds[model_name] = model.predict(X)
                # Estimate uncertainty using a simple heuristic
                lower_bounds[model_name] = base_preds[model_name] * 0.8
                upper_bounds[model_name] = base_preds[model_name] * 1.2
        
        # Generate ensemble point predictions
        mean_pred = self.predict(X)
        
        # For ensemble uncertainty, we combine the uncertainties from base models
        # in a weighted manner according to the meta-learner weights
        
        if hasattr(self.meta_learner, 'coef_'):
            # Get weights for each model
            model_weights = {}
            for model_name in self.base_models.keys():
                # Sum weights for features associated with this model
                model_weights[model_name] = sum(
                    self.model_weights.get(f"pred_{model_name}", 0.0),
                    0.0  # Default if not found
                )
                
            # Normalize weights to sum to 1
            total_weight = sum(model_weights.values())
            if total_weight > 0:
                model_weights = {k: v/total_weight for k, v in model_weights.items()}
            else:
                # Equal weights if total is zero or negative
                model_weights = {k: 1.0/len(model_weights) for k in model_weights.keys()}
            
            # Compute weighted combination of uncertainty intervals
            lower_bound = np.zeros_like(mean_pred)
            upper_bound = np.zeros_like(mean_pred)
            
            for model_name, weight in model_weights.items():
                lower_bound += weight * lower_bounds[model_name]
                upper_bound += weight * upper_bounds[model_name]
        else:
            # If no model weights are available, use simple average
            lower_bound = np.mean([lb for lb in lower_bounds.values()], axis=0)
            upper_bound = np.mean([ub for ub in upper_bounds.values()], axis=0)
        
        # Ensure bounds are sensible
        lower_bound = np.maximum(lower_bound, 0)  # Ensure non-negative
        upper_bound = np.maximum(upper_bound, mean_pred)  # Ensure upper > mean
        
        return mean_pred, lower_bound, upper_bound
    
    def _calculate_feature_importance(self) -> None:
        """Calculate feature importance based on meta-learner weights."""
        if not self.is_fitted or not hasattr(self.meta_learner, 'coef_'):
            return
        
        # Create feature importance DataFrame
        importance = []
        for feature, weight in self.model_weights.items():
            importance.append({
                'feature': feature,
                'importance': abs(weight)  # Use absolute weight as importance
            })
        
        self.feature_importance = pd.DataFrame(importance).sort_values('importance', ascending=False)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance or attribution scores.
        
        Returns:
            DataFrame with feature importance information
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet")
            
        if self.feature_importance is None:
            self._calculate_feature_importance()
            
        return self.feature_importance
    
    def plot_model_weights(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot the weights assigned to each base model.
        
        Args:
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        if not self.is_fitted or not self.model_weights:
            raise ValueError("Model is not trained yet or doesn't have interpretable weights")
        
        # Prepare data for plotting
        model_names = sorted(self.base_models.keys())
        model_impact = []
        
        for model_name in model_names:
            # Sum all weights related to this model
            impact = sum(abs(weight) for feature, weight in self.model_weights.items() 
                         if f"pred_{model_name}" in feature)
            model_impact.append(impact)
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, model_impact, alpha=0.7)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', rotation=0)
        
        plt.title('Base Model Impact in Ensemble')
        plt.ylabel('Absolute Weight Sum')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return plt.gcf()
    
    def plot_ensemble_comparison(self, X: pd.DataFrame, y: pd.Series, 
                               n_samples: int = 100, save_path: Optional[str] = None) -> plt.Figure:
        """Plot a comparison of ensemble vs base model predictions.
        
        Args:
            X: Feature dataframe
            y: Target series
            n_samples: Number of samples to plot
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet")
        
        # Generate predictions
        base_preds = {}
        for model_name, model in self.base_models.items():
            base_preds[model_name] = model.predict(X)
        
        ensemble_preds = self.predict(X)
        
        # Select a subset of samples
        if X.shape[0] > n_samples:
            indices = np.random.choice(X.shape[0], n_samples, replace=False)
            indices = np.sort(indices)  # Sort for better visualization
        else:
            indices = np.arange(X.shape[0])
        
        # Prepare data for plotting
        x_vals = np.arange(len(indices))
        selected_y = y.iloc[indices].values
        selected_ensemble = ensemble_preds[indices]
        selected_base_preds = {model: preds[indices] for model, preds in base_preds.items()}
        
        # Create the plot
        plt.figure(figsize=(15, 8))
        
        # Plot base model predictions
        for model_name, preds in selected_base_preds.items():
            plt.plot(x_vals, preds, 'o-', alpha=0.3, label=f'{model_name} predictions')
        
        # Plot ensemble predictions and actual values
        plt.plot(x_vals, selected_ensemble, 'o-', color='red', linewidth=2, label='Ensemble predictions')
        plt.plot(x_vals, selected_y, 'x-', color='black', linewidth=2, label='Actual values')
        
        plt.title('Ensemble vs Base Model Predictions')
        plt.xlabel('Sample')
        plt.ylabel('Demand')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return plt.gcf()
