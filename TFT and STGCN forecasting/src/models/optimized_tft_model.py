#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
import matplotlib.pyplot as plt
import time
from datetime import datetime
from pathlib import Path
import re

# PyTorch Forecasting imports
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss, SMAPE, MAE
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

# PyTorch Lightning imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from .base_model import BaseModel
from ..utils.logger import StructuredLogger

class OptimizedTFTModel(BaseModel):
    
    def __init__(self, model_config: Dict, logger: Optional[StructuredLogger] = None):
        """
        Initialize the optimized TFT model.
        
        Args:
            model_config: Dictionary containing model configuration
            logger: Logger for tracking operations
        """
        super().__init__(model_config, logger)
        self.tft_params = self.model_config.get("params", {})
        self._set_default_params()
        
        self.model = None
        self.trainer = None
        self.training_data = None
        self.feature_importance = None
        self.is_fitted = False
        self.model_info = {"type": "TFT"}
        self.feature_names = None
        self.feature_categories = {
            'temporal': [],
            'spatial': [],
            'context': [],
            'static': []
        }
        self.is_using_enhanced_features = False
        self.metrics = {}
        self._setup_trainer()
        
    def _set_default_params(self):
        """Set default hyperparameters if not specified in the config."""
        defaults = {
            # Data parameters
            "target_name": "trip_count",
            "time_idx_name": "time_idx",
            "group_id_name": "start_location_id", 
            "max_encoder_length": 24,
            "max_prediction_length": 4,
            "min_encoder_length": 12,  # Minimum context required
            "static_categoricals": [],
            "time_varying_known_categoricals": [],
            "time_varying_known_reals": [],
            "time_varying_unknown_categoricals": [],
            "time_varying_unknown_reals": ["trip_count"],
            
            # Model architecture
            "hidden_size": 64,
            "lstm_layers": 2,
            "attention_head_size": 4,
            "dropout": 0.1,
            "hidden_continuous_size": 32,
            "output_size": 7,  # Number of quantiles for prediction
            
            # Training parameters
            "batch_size": 64,
            "learning_rate": 0.001,
            "max_epochs": 50,
            "early_stopping_patience": 10,
            
            # Other parameters
            "log_interval": 10,
            "reduce_on_plateau_patience": 3,
            "use_learning_rate_scheduler": True
        }
        for key, value in defaults.items():
            if key not in self.tft_params:
                self.tft_params[key] = value
                
        self.target_name = self.tft_params["target_name"]
        self.time_idx_name = self.tft_params["time_idx_name"]
        self.group_id_name = self.tft_params["group_id_name"]
        self.max_encoder_length = self.tft_params["max_encoder_length"]
        self.max_prediction_length = self.tft_params["max_prediction_length"]
        self.batch_size = self.tft_params["batch_size"]
        
    def _setup_trainer(self):
        """Set up the PyTorch Lightning trainer."""
        callbacks = []
        
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=self.tft_params["early_stopping_patience"],
            verbose=False,
            mode="min"
        )
        callbacks.append(early_stop_callback)
        
        if self.tft_params["use_learning_rate_scheduler"]:
            lr_monitor = LearningRateMonitor(logging_interval="epoch")
            callbacks.append(lr_monitor)
        
        logger = True  # Use default logger
        if "log_dir" in self.tft_params:
            log_dir = self.tft_params["log_dir"]
            os.makedirs(log_dir, exist_ok=True)
            logger = TensorBoardLogger(log_dir)
            
        # Configure trainer with current PyTorch Lightning API (v2.0+)
        trainer_kwargs = {
            "max_epochs": self.tft_params["max_epochs"],
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": torch.cuda.device_count() if torch.cuda.is_available() else 1,
            "gradient_clip_val": 0.1,
            "callbacks": callbacks,
            "logger": logger,
            "enable_checkpointing": True,
            "enable_progress_bar": True,
            "log_every_n_steps": self.tft_params["log_interval"],
        }
        
        # Initialize the trainer
        self.trainer = pl.Trainer(**trainer_kwargs)
        
    def _sanitize_column_names(self, data: pd.DataFrame) -> pd.DataFrame:
        """Sanitize column names to be compatible with PyTorch Forecasting.
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with sanitized column names
        """
        sanitized_data = data.copy()
        renamed_columns = {}
        
        for col in data.columns:
            sanitized_col = re.sub(r'[^\w]+', '_', col)
            sanitized_col = sanitized_col.replace('_0_', '_0').replace('_1_', '_1')
            
            if sanitized_col != col:
                sanitized_data = sanitized_data.rename(columns={col: sanitized_col})
                renamed_columns[col] = sanitized_col
                self.logger.info(f"Sanitizing column name: {col} -> {sanitized_col}")
        
        if hasattr(self, 'feature_categories') and self.feature_categories:
            for category, features in self.feature_categories.items():
                self.feature_categories[category] = [
                    renamed_columns.get(feat, feat) for feat in features
                ]
        
        return sanitized_data
    
    def _analyze_data_continuity(self, data: pd.DataFrame) -> Dict:
        """Analyze temporal continuity of the data to determine optimal window sizes.
        
        Args:
            data: Input time series DataFrame
            
        Returns:
            Dictionary of continuity statistics
        """
        self.logger.info("Analyzing data continuity to determine optimal window sizes")
        
        stats = {
            "total_points": len(data),
            "gaps": 0,
            "gap_indices": [],
            "consecutive_segments": [],
            "max_consecutive": 0
        }
        
        if self.time_idx_name not in data.columns:
            self.logger.info("Creating time_idx from hour_start column")
            self._create_time_index(data)
            
        data_sorted = data.sort_values(by=[self.time_idx_name])
        
        if self.group_id_name in data.columns:
            groups = data_sorted.groupby(self.group_id_name)
            all_segments = []
            for name, group in groups:
                time_indices = group[self.time_idx_name].values
                segments = self._find_continuous_segments(time_indices)
                all_segments.extend(segments)
                for i in range(len(time_indices) - 1):
                    if time_indices[i+1] - time_indices[i] > 1:
                        stats["gaps"] += 1
                        stats["gap_indices"].append((name, time_indices[i], time_indices[i+1]))
            
            if all_segments:
                stats["consecutive_segments"] = all_segments
                stats["max_consecutive"] = max([len(seg) for seg in all_segments])
        else:
            time_indices = data_sorted[self.time_idx_name].values
            segments = self._find_continuous_segments(time_indices)
            stats["consecutive_segments"] = segments
            stats["max_consecutive"] = max([len(seg) for seg in segments]) if segments else 0

            for i in range(len(time_indices) - 1):
                if time_indices[i+1] - time_indices[i] > 1:
                    stats["gaps"] += 1
                    stats["gap_indices"].append(("global", time_indices[i], time_indices[i+1]))
        
        self.logger.info(f"Found {stats['gaps']} gaps in time series data")
        self.logger.info(f"Longest continuous segment: {stats['max_consecutive']} timesteps")
        return stats
    
    def _find_continuous_segments(self, time_indices: np.ndarray) -> List[List[int]]:
        """Find continuous segments in time indices.
        
        Args:
            time_indices: Array of time indices
            
        Returns:
            List of continuous segments (each a list of indices)
        """
        segments = []
        current_segment = [time_indices[0]] if len(time_indices) > 0 else []
        
        for i in range(1, len(time_indices)):
            if time_indices[i] == time_indices[i-1] + 1:
                current_segment.append(time_indices[i])
            else:
                segments.append(current_segment)
                current_segment = [time_indices[i]]
        
        if current_segment:
            segments.append(current_segment)
            
        return segments
    
    def _create_time_index(self, data: pd.DataFrame) -> None:
        """Create a sequential time index from the datetime column.
        
        Args:
            data: Input DataFrame
        """
        self.logger.info("Creating sequential integer time index")
        datetime_col = None
        if "hour_start" in data.columns:
            datetime_col = "hour_start"
        elif "date" in data.columns:
            datetime_col = "date"
            
        if datetime_col is not None:
            if not pd.api.types.is_datetime64_dtype(data[datetime_col]):
                try:
                    data[datetime_col] = pd.to_datetime(data[datetime_col])
                except Exception as e:
                    self.logger.warning(f"Couldn't convert {datetime_col} to datetime: {str(e)}")
            
            sorted_datetimes = data[datetime_col].sort_values().unique()
            datetime_to_idx = {dt: i for i, dt in enumerate(sorted_datetimes)}
            
            data[self.time_idx_name] = data[datetime_col].map(datetime_to_idx)
            self.logger.info(f"Created sequential time index. Range: {data[self.time_idx_name].min()} to {data[self.time_idx_name].max()}")
        else:
            if self.time_idx_name not in data.columns:
                data[self.time_idx_name] = range(len(data))
                self.logger.warning("No datetime column found. Created a sequential index without temporal meaning.")
        
        data[self.time_idx_name] = data[self.time_idx_name].astype('int64')
        data[self.time_idx_name] = data[self.time_idx_name].astype('int64')
        
    def _prepare_data(self, X: pd.DataFrame, y: pd.Series = None, is_prediction: bool = False) -> TimeSeriesDataSet:
        """Prepare data for TFT model training or prediction.
        
        Args:
            X: Feature dataframe
            y: Optional target series for training
            is_prediction: Whether preparing for prediction
            
        Returns:
            TimeSeriesDataSet ready for model training/prediction
        """
        data = X.copy()
        if y is not None and not is_prediction:
            data[self.target_name] = y.values
            
        data = self._sanitize_column_names(data)
        continuity_stats = self._analyze_data_continuity(data)
        original_encoder_length = self.max_encoder_length
        original_prediction_length = self.max_prediction_length
        
        if continuity_stats["max_consecutive"] > 0:
            self.max_encoder_length = int(min(
                self.max_encoder_length,
                max(12, continuity_stats["max_consecutive"] // 2)
            ))
            
            self.max_prediction_length = int(min(
                self.max_prediction_length,
                max(1, continuity_stats["max_consecutive"] // 10)
            ))
            
            self.logger.info(f"Adjusted window sizes based on data continuity: encoder={self.max_encoder_length} "
                           f"(was {original_encoder_length}), prediction={self.max_prediction_length} "
                           f"(was {original_prediction_length})")
                
        if self.time_idx_name not in data.columns:
            self._create_time_index(data)
        else:
            data[self.time_idx_name] = data[self.time_idx_name].astype('int64')
            
        if self.group_id_name in data.columns:
            data[self.group_id_name] = data[self.group_id_name].astype(str)
            self.logger.info(f"Converted {self.group_id_name} to string type")
        
        try:
            return self._create_time_series_dataset(data, is_prediction)
        except Exception as e:
            self.logger.error(f"Error creating TimeSeriesDataSet: {str(e)}")
            return self._create_fallback_dataset(data, is_prediction)
            
    def _create_time_series_dataset(self, data: pd.DataFrame, is_prediction: bool = False) -> TimeSeriesDataSet:
        """Create a TimeSeriesDataSet from the prepared data.
        
        Args:
            data: Prepared DataFrame
            is_prediction: Whether creating dataset for prediction
            
        Returns:
            TimeSeriesDataSet for training or prediction
        """
        if self.is_using_enhanced_features and hasattr(self, 'feature_categories') and any(self.feature_categories.values()):
            self.logger.info("Using enhanced feature categories for TimeSeriesDataSet creation")
            
            static_reals = [col for col in self.feature_categories['static'] 
                           if col in data.columns and col not in [self.time_idx_name, self.target_name, "start_location_id"]]
            
            time_varying_known_reals = []
            for category in ['temporal', 'spatial', 'context']:
                time_varying_known_reals.extend([col for col in self.feature_categories[category] 
                                              if col in data.columns and col not in [self.time_idx_name, self.target_name, "start_location_id"]])
            
            self.logger.info(f"Using {len(static_reals)} static features and {len(time_varying_known_reals)} time-varying features")
        else:
            self.logger.info("Using default feature categorization")
            static_reals = []
            time_varying_known_reals = [col for col in data.columns 
                                      if col not in [self.time_idx_name, self.target_name, "start_location_id"]]
        
        for col in time_varying_known_reals + static_reals + [self.target_name]:
            if col in data.columns:
                try:
                    data[col] = data[col].replace([np.inf, -np.inf], np.nan)
                    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
                    data[col] = data[col].astype('float32')
                    data[col] = data[col].replace([np.inf, -np.inf], np.nan)
                    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
                    data[col] = data[col].astype('float32')  # PyTorch works well with float32
                except Exception as e:
                    self.logger.warning(f"Error converting {col} from {data[col].dtype} to float32: {str(e)}")
                    data[col] = data[col].astype('float32', errors='ignore')
        
        if is_prediction and hasattr(self, 'training_data'):
            return TimeSeriesDataSet.from_dataset(self.training_data, data, predict=True)
        else:
            dataset_params = {
                "time_idx": self.time_idx_name,
                "target": self.target_name,
                "group_ids": [self.group_id_name] if self.group_id_name in data.columns else [],
                "max_encoder_length": self.max_encoder_length,
                "max_prediction_length": self.max_prediction_length,
                "time_varying_unknown_reals": [self.target_name],
                "time_varying_known_reals": time_varying_known_reals,
                "static_reals": static_reals,
                "add_relative_time_idx": True,
                "add_target_scales": True,
                "add_encoder_length": True,
                "allow_missing_timesteps": True  # Allow for gaps in time series
            }
            
            dataset_params["target_normalizer"] = GroupNormalizer(
                groups=[self.group_id_name] if self.group_id_name in data.columns else [],
                transformation="softplus"  # Good for count data
            )
            
            return TimeSeriesDataSet(data, **dataset_params)
    
    def _create_fallback_dataset(self, data: pd.DataFrame, is_prediction: bool = False) -> TimeSeriesDataSet:
        """Create a minimal fallback dataset when normal creation fails.
        
        Args:
            data: Input DataFrame
            is_prediction: Whether creating for prediction
            
        Returns:
            Minimal TimeSeriesDataSet
        """
        self.logger.warning("Creating fallback minimal TimeSeriesDataSet")
        min_data = data.copy()[[self.time_idx_name]].reset_index(drop=True)
        
        if self.group_id_name in data.columns:
            min_data[self.group_id_name] = data[self.group_id_name].astype(str)
        else:
            min_data[self.group_id_name] = "0"  # Dummy group ID
            
        if self.target_name in data.columns:
            min_data[self.target_name] = data[self.target_name].astype(float)
        else:
            min_data[self.target_name] = 0.0  # Dummy target
            
        dummy_feature = "dummy_feature"
        min_data[dummy_feature] = 0.0
            
        fallback_dataset = TimeSeriesDataSet(
            min_data,
            time_idx=self.time_idx_name,
            target=self.target_name,
            group_ids=[self.group_id_name],
            max_encoder_length=min(self.max_encoder_length, 12),  # Use smaller window
            max_prediction_length=min(self.max_prediction_length, 1),  # Predict just one step
            time_varying_unknown_reals=[self.target_name],
            time_varying_known_reals=[dummy_feature],
            static_categoricals=[],
            add_relative_time_idx=True,
            allow_missing_timesteps=True
        )
        
        return fallback_dataset
        
    def fit(self, X: pd.DataFrame, y: pd.Series, validation_data: Optional[Tuple] = None) -> Dict:
        """Train the TFT model on the provided data.
        
        Args:
            X: Feature dataframe
            y: Target series
            validation_data: Optional tuple of (X_val, y_val)
            
        Returns:
            Dictionary with training metrics
        """
        self.logger.info(f"Training TFT model with {X.shape[1]} features and {len(y)} samples")
        start_time = time.time()
        self.feature_names = list(X.columns)
        self.model_info["feature_count"] = len(self.feature_names)
        
        try:
            train_dataset = self._prepare_data(X, y)
            train_dataloader = train_dataset.to_dataloader(train=True, batch_size=self.batch_size)
            self.training_data = train_dataset
            
            val_dataloader = None
            if validation_data is not None:
                X_val, y_val = validation_data
                self.logger.info(f"Using validation set with {len(y_val)} samples")
                val_dataset = self._prepare_data(X_val, y_val)
                val_dataloader = val_dataset.to_dataloader(train=False, batch_size=self.batch_size)
            else:
                train_dataloader, val_dataloader = self.training_data.to_dataloader(
                    train=True,
                    batch_size=self.batch_size,
                    val_split=0.2
                )
                
            self.model = TemporalFusionTransformer.from_dataset(
                self.training_data,
                learning_rate=self.tft_params["learning_rate"],
                hidden_size=self.tft_params["hidden_size"],
                lstm_layers=self.tft_params["lstm_layers"],
                attention_head_size=self.tft_params["attention_head_size"],
                dropout=self.tft_params["dropout"],
                hidden_continuous_size=self.tft_params["hidden_continuous_size"],
                output_size=self.tft_params["output_size"],  # Number of quantiles
                loss=QuantileLoss(),
                log_interval=self.tft_params["log_interval"],
                reduce_on_plateau_patience=self.tft_params["reduce_on_plateau_patience"]
            )
            
            self.logger.info(f"TFT model created with {sum(p.numel() for p in self.model.parameters())} parameters")
            
            self.logger.info("Starting TFT model training")
            try:
                # First try the new API
                self.trainer.fit(
                    model=self.model,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader
                )
            except (TypeError, ValueError) as e:
                self.logger.warning(f"Modern PyTorch Lightning API failed: {str(e)}. Trying legacy approach...")
                # Try the legacy API
                try:
                    # Direct approach
                    self.trainer.fit(self.model, train_dataloader, val_dataloader)
                except Exception as e2:
                    self.logger.warning(f"Legacy API also failed: {str(e2)}. Using direct model training...")
                    # Manual training loop as last resort
                    self.model.train()
                    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.tft_params["learning_rate"])
                    for _ in range(min(10, self.tft_params["max_epochs"])):
                        # Just make sure the model is trained enough to be usable
                        for batch in train_dataloader:
                            optimizer.zero_grad()
                            output = self.model(batch)
                            loss = self.model.loss(output, batch)
                            loss.backward()
                            optimizer.step()
            
            if hasattr(self.trainer, 'callback_metrics'):
                val_loss = self.trainer.callback_metrics.get("val_loss", None)
                if val_loss is not None:
                    self.metrics["val_loss"] = float(val_loss)
                    self.logger.info(f"Final validation loss: {self.metrics['val_loss']:.4f}")
                    
            self.logger.info("Computing feature importance")
            try:
                feature_importance = self.model.interpret_output(
                    val_dataloader,
                    reduction="sum"
                )
                self.feature_importance = feature_importance
                self.logger.info("Feature importance computed successfully")
            except Exception as e:
                self.logger.warning(f"Could not compute feature importance: {str(e)}")
                

            if "model_dir" in self.model_config:
                self._save_model(self.model_config["model_dir"])
            
            self.is_fitted = True
            training_time = time.time() - start_time
            self.logger.info(f"TFT model training completed in {training_time:.2f} seconds")
            
            return {
                "status": "success",
                "training_time": training_time,
                "val_loss": self.metrics.get("val_loss"),
                "metrics": self.metrics
            }
            
        except Exception as e:
            training_time = time.time() - start_time
            self.logger.error(f"Error training TFT model: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "training_time": training_time
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
            
        self.logger.info(f"Generating predictions for {X.shape[0]} samples")
        
        try:
            pred_dataset = self._prepare_data(X, is_prediction=True)
            
            try:
                pred_dataloader = pred_dataset.to_dataloader(train=False, batch_size=self.batch_size)
                predictions = self.model.predict(pred_dataloader)
            except (AttributeError, TypeError) as e:
                self.logger.warning(f"Prediction with modern API failed: {str(e)}. Trying legacy API...")
                
                if hasattr(self, 'training_data') and self.training_data is not None:
                    predictions = self.model.predict(
                        X,  # Raw dataframe
                        mode="prediction",
                        return_x=False,
                        trainer_kwargs={"enable_progress_bar": False}
                    )
                else:
                    self.logger.warning("No training dataset available, using minimal prediction approach")
                    pred_dataloader = pred_dataset.to_dataloader(train=False, batch_size=self.batch_size)
                    predictions = self.model.predict(
                        pred_dataloader,
                        mode="prediction",
                        return_x=False,
                        trainer_kwargs={"enable_progress_bar": False}
                    )
            
            # Extract median predictions (default is quantile 0.5)
            if hasattr(predictions, 'numpy'):
                median_predictions = predictions.numpy()[:, -1, 1]  # Last timestep, median quantile
            else:
                median_predictions = np.array(predictions)[:, -1, 1] if predictions.ndim > 2 else np.array(predictions)
            
            return median_predictions
            
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            return np.zeros(len(X))
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate predictions with uncertainty intervals.
        
        Args:
            X: Feature dataframe for prediction
            
        Returns:
            Tuple of (mean predictions, lower bounds, upper bounds)
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet")
            
        self.logger.info(f"Generating predictions with uncertainty for {X.shape[0]} samples")
        
        try:
            pred_dataset = self._prepare_data(X, is_prediction=True)
            pred_dataloader = pred_dataset.to_dataloader(train=False, batch_size=self.batch_size)
            
            try:
                raw_predictions = self.model.predict(
                    pred_dataloader,
                    mode="prediction",
                    return_x=False
                )
                
                if not isinstance(raw_predictions, np.ndarray):
                    predictions = raw_predictions.numpy()
                else:
                    predictions = raw_predictions
                    
                lower_quantile = predictions[:, -1, 0]
                median_quantile = predictions[:, -1, 1]
                upper_quantile = predictions[:, -1, 2]
                
                return median_quantile, lower_quantile, upper_quantile
                
            except Exception as e:
                self.logger.warning(f"Quantile prediction failed: {str(e)}. Using heuristic uncertainty.")
                
                mean_pred = self.predict(X)
                
                uncertainty = np.maximum(mean_pred * 0.2, 1.0)
                
                lower_bound = np.maximum(mean_pred - 1.96 * uncertainty, 0)  # Ensure non-negative
                upper_bound = mean_pred + 1.96 * uncertainty
                
                return mean_pred, lower_bound, upper_bound
                
        except Exception as e:
            self.logger.error(f"Error generating predictions with uncertainty: {str(e)}")
            zeros = np.zeros(len(X))
            return zeros, zeros, zeros
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores from the trained model.
        
        Returns:
            DataFrame with feature importance information
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet")
            
        if self.feature_importance is None:
            self.logger.warning("No feature importance available")
            importance = pd.DataFrame({
                'feature': self.feature_names if self.feature_names else ['unknown'],
                'importance': np.ones(len(self.feature_names) if self.feature_names else 1) / 
                              (len(self.feature_names) if self.feature_names else 1)
            })
            return importance
        
        try:
            raw_importance = self.feature_importance
            
            if isinstance(raw_importance, dict):
                importance_data = []
                for feature_type, values in raw_importance.items():
                    if feature_type != 'static':
                        # For temporal features, average across time
                        for feature, importances in values.items():
                            if isinstance(importances, torch.Tensor):
                                mean_importance = importances.mean().item()
                            elif isinstance(importances, np.ndarray):
                                mean_importance = importances.mean()
                            else:
                                mean_importance = importances
                            importance_data.append({
                                'feature': feature,
                                'importance': mean_importance,
                                'type': feature_type
                            })
                
                importance = pd.DataFrame(importance_data)
                if not importance.empty:
                    importance = importance.sort_values('importance', ascending=False)
            
            elif isinstance(raw_importance, pd.DataFrame):
                importance = raw_importance
            
            else:
                try:
                    if hasattr(raw_importance, '__iter__'):
                        if len(raw_importance) == len(self.feature_names):
                            importance = pd.DataFrame({
                                'feature': self.feature_names,
                                'importance': raw_importance
                            })
                        else:
                            raise ValueError(f"Cannot interpret feature importance format")
                    else:
                        raise ValueError(f"Cannot interpret feature importance format")
                except:
                    self.logger.warning("Could not interpret feature importance, using uniform values")
                    importance = pd.DataFrame({
                        'feature': self.feature_names if self.feature_names else ['unknown'],
                        'importance': np.ones(len(self.feature_names) if self.feature_names else 1) / 
                                     (len(self.feature_names) if self.feature_names else 1)
                    })
            
            return importance
            
        except Exception as e:
            self.logger.error(f"Error processing feature importance: {str(e)}")
            return pd.DataFrame({
                'feature': self.feature_names if self.feature_names else ['unknown'],
                'importance': np.ones(len(self.feature_names) if self.feature_names else 1) / 
                             (len(self.feature_names) if self.feature_names else 1)
            })
    
    def _save_model(self, model_dir: str) -> bool:
        """Save the trained model to disk.
        
        Args:
            model_dir: Directory to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_dir = Path(model_dir)
            os.makedirs(model_dir, exist_ok=True)
            
            ckpt_path = model_dir / "tft_model.ckpt"
            self.logger.info(f"Saving model checkpoint to {ckpt_path}")
            self.trainer.save_checkpoint(ckpt_path)
            
            if self.feature_importance is not None:
                importance_df = self.get_feature_importance()
                if not importance_df.empty:
                    importance_path = model_dir / "feature_importance.csv"
                    importance_df.to_csv(importance_path, index=False)
                    self.logger.info(f"Saved feature importance to {importance_path}")
            
            # Save model info
            info_path = model_dir / "model_info.json"
            import json
            with open(info_path, 'w') as f:
                serializable_info = {}
                for k, v in self.model_info.items():
                    try:
                        json.dumps({k: v})
                        serializable_info[k] = v
                    except:
                        serializable_info[k] = str(v)
                        
                json.dump(serializable_info, f, indent=2)
            
            self.logger.info(f"Model successfully saved to {model_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load(self, model_dir: str) -> bool:
        """Load a trained model from disk.
        
        Args:
            model_dir: Directory containing the saved model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_dir = Path(model_dir)
            ckpt_path = model_dir / "tft_model.ckpt"
            
            if not os.path.exists(ckpt_path):
                self.logger.error(f"Model checkpoint not found at {ckpt_path}")
                return False
                
            # Create a dummy dataset for model loading
            dummy_data = pd.DataFrame({
                self.time_idx_name: range(48),  # 48 timesteps
                self.group_id_name: ["dummy"] * 48,  # One group
                self.target_name: [0.0] * 48,  # Dummy target
                "dummy_feature": [0.0] * 48  # Dummy feature
            })
            
            dummy_dataset = TimeSeriesDataSet(
                dummy_data,
                time_idx=self.time_idx_name,
                target=self.target_name,
                group_ids=[self.group_id_name],
                max_encoder_length=self.max_encoder_length,
                max_prediction_length=self.max_prediction_length,
                time_varying_unknown_reals=[self.target_name],
                time_varying_known_reals=["dummy_feature"],
                add_relative_time_idx=True,
                add_target_scales=True
            )
            
            self.training_data = dummy_dataset
            
            self.model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)
            self.logger.info(f"Successfully loaded TFT model from {ckpt_path}")
            self.is_fitted = True
            
            importance_path = model_dir / "feature_importance.csv"
            if os.path.exists(importance_path):
                self.feature_importance = pd.read_csv(importance_path)
            
            info_path = model_dir / "model_info.json"
            if os.path.exists(info_path):
                import json
                with open(info_path, 'r') as f:
                    self.model_info.update(json.load(f))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load TFT model: {str(e)}")
            self.is_fitted = False
            return False
