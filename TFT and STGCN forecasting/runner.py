#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ride Demand Forecasting System Runner

This script executes the entire pipeline for the ride demand forecasting system.
It integrates all components, from data loading to model training, evaluation, and surge pricing optimization.

Author: Daymenion
Date: May 2025
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # 0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR
import sys
import pandas as pd
import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime

# Ensure src is in the path
root_dir = Path(__file__).resolve().parent
src_dir = root_dir / 'src'
sys.path.append(str(root_dir))
sys.path.append(str(src_dir))

# Import project modules
from src.utils.logger import StructuredLogger
from src.utils.config_manager import ConfigManager
from src.data.optimized_data_processor import EnhancedDataProcessor
from src.features.optimized_feature_engineer import EnhancedFeatureEngineer
from src.models.model_factory import ModelFactory
from src.evaluation.model_evaluator import ModelEvaluator
from src.evaluation.visualization import ForecastVisualizer
from src.evaluation.price_evaluator import PriceEvaluator
from src.pricing.surge_optimizer import SurgePricingOptimizer


class DemandForecastingRunner:
    """
    The main runner class for the ride demand forecasting system.
    
    This class orchestrates the entire pipeline from data loading to model evaluation
    and surge pricing optimization. It provides a simple interface to execute different
    segments of the pipeline in a configurable way.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the runner with the given configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        # Set up paths
        self.root_dir = root_dir
        self.config_path = self.root_dir / config_path
        
        # Set up logger
        log_dir = self.root_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        self.logger = StructuredLogger(
            "demand_forecasting_runner", 
            log_dir=str(log_dir),
            console_level=logging.INFO,
            file_level=logging.DEBUG
        )
        
        self.logger.info(f"Initializing runner with config: {self.config_path}")
        
        # Load configuration
        self.config = ConfigManager(self.config_path)
        
        # Initialize components
        self.data_loader = None
        self.feature_engineer = None
        self.models = {}
        self.model_factory = None
        self.evaluator = None
        self.visualizer = None 
        self.surge_optimizer = None
        self.price_evaluator = None
        
        # Run info
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = self.root_dir / "results" / self.run_id
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Training/validation/test data
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.feature_columns = None
        self.target_column = None
        
        self.logger.info(f"Runner initialized, run_id: {self.run_id}")
    
    def setup_data_pipeline(self):
        """
        Set up the data loading and processing pipeline.
        """
        self.logger.info("Setting up data pipeline")
        self.enhanced_processor = EnhancedDataProcessor(self.config, self.logger)
        self.feature_engineer = EnhancedFeatureEngineer(self.config.get("features"), self.logger)
        # Load data - first check if we have fully prepared and enriched data
        try:
            self.logger.info("Attempting to load enriched data")
            hourly_demand = self.enhanced_processor.load_processed_data("enriched_trips.parquet")
        except FileNotFoundError:
            try:
                self.logger.info("Enriched data not found, attempting to load prepared data")
                hourly_demand = self.enhanced_processor.load_processed_data("prepared_trips.parquet")
            except FileNotFoundError:
                self.logger.info("No preprocessed data found, loading and processing from raw data")
                hourly_demand = self.enhanced_processor.load_all_data()
        
        # Check if enriched_trips.parquet exists in processed directory
        if os.path.exists(self.enhanced_processor.processed_path / "enriched_trips.parquet"):
            self.logger.info("Enriched data found, loading from processed directory")
            hourly_demand = self.enhanced_processor.load_processed_data("enriched_trips.parquet")
        else:   
            self.logger.info("Applying enhanced feature engineering")
            hourly_demand = self.feature_engineer.create_all_features(hourly_demand)
            self.enhanced_processor.save_processed_data(hourly_demand, "enriched_trips.parquet")
            self.logger.info(f"Saved fully enriched trips data with {len(self.feature_engineer.get_feature_columns()['all'])} features")
        
        feature_cols = self.feature_engineer.get_feature_columns()
        self.logger.info(f"Using {len(feature_cols['temporal'])} temporal features, {len(feature_cols['spatial'])} spatial features, {len(feature_cols['context'])} context features")
        self.logger.info(f"Total feature count: {len(feature_cols['all'])}")
        
        split_config = self.config.get("data_splits", {})
        train_ratio = split_config.get("train_ratio", 0.7)
        val_ratio = split_config.get("validation_ratio", 0.15)
        
        # Split data into train, val, test sets
        train_end = int(len(hourly_demand) * train_ratio)
        val_end = int(len(hourly_demand) * (train_ratio + val_ratio))
        
        # Sort data by datetime column before splitting
        datetime_col = self.config.get("features", {}).get("datetime_column", "hour_start")
        hourly_demand = hourly_demand.sort_values(datetime_col).reset_index(drop=True)
        
        self.train_data = hourly_demand.iloc[:train_end].copy()
        self.val_data = hourly_demand.iloc[train_end:val_end].copy()
        self.test_data = hourly_demand.iloc[val_end:].copy()
        
        # If we've already created features for all data, just register them in the feature engineer
        if "weather_temp" in hourly_demand.columns and "is_heavy_rain" in hourly_demand.columns:
            self.logger.info("Using already processed features for train/val/test splits")
            
            # Register the pre-processed features in the feature engineer
            # This ensures feature sets aren't empty even when using pre-processed data
            # Identify feature types based on naming conventions
            for col in hourly_demand.columns:
                if col.startswith(("hour_", "day_", "month_", "year_", "is_holiday", "is_weekend", "is_rush_hour")):
                    self.feature_engineer.temporal_features.add(col)
                elif col.startswith(("location_", "cluster_", "dist_", "proximity_", "centrality_")):
                    self.feature_engineer.spatial_features.add(col)
                elif col.startswith(("weather_", "temp_", "rain_", "wind_", "is_raining", "event_")):
                    self.feature_engineer.context_features.add(col)
        else:
            # Apply feature engineering separately to each split with appropriate is_train flag
            self.logger.info("Creating features for train/val/test splits")
            self.train_data = self.feature_engineer.create_all_features(self.train_data, is_train=True)
            self.val_data = self.feature_engineer.create_all_features(self.val_data, is_train=False)
            self.test_data = self.feature_engineer.create_all_features(self.test_data, is_train=False)
        
        # Get feature and target columns
        feature_config = self.config.get("features", {})
        self.target_column = feature_config.get("target_column", "trip_count")
        
        # Get feature lists from feature engineer
        feature_dict = self.feature_engineer.get_feature_columns()
        # Use the 'all' key which contains all feature names as a list for DataFrame indexing
        self.feature_columns = feature_dict.get('all', [])
        
        # Handle categorical features - convert them to numeric for models like LightGBM
        categorical_cols = []
        for col in self.feature_columns:
            if col in self.train_data.columns and self.train_data[col].dtype == 'object':
                self.logger.info(f"Converting categorical column {col} to numeric")
                categorical_cols.append(col)
                # Use label encoding for categorical features
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                # Fit on all available data
                combined_values = pd.concat([
                    self.train_data[col], 
                    self.val_data[col],
                    self.test_data[col]
                ]).astype(str)
                le.fit(combined_values)
                # Transform each dataset
                self.train_data[col] = le.transform(self.train_data[col].astype(str))
                self.val_data[col] = le.transform(self.val_data[col].astype(str))
                self.test_data[col] = le.transform(self.test_data[col].astype(str))
        
        if categorical_cols:
            self.logger.info(f"Encoded {len(categorical_cols)} categorical features: {categorical_cols}")
        
        self.logger.info(f"Data pipeline setup complete. Features: {len(self.feature_columns)}, Target: {self.target_column}")
        
        # Save feature names for later use
        with open(self.results_dir / "feature_columns.yaml", "w") as f:
            yaml.dump({"features": self.feature_columns}, f, default_flow_style=False)
    
    def setup_models(self):
        """
        Set up all forecasting models.
        """
        self.logger.info("Setting up models")
        
        # Initialize model factory
        model_config = self.config.get("models", {})
        self.model_factory = ModelFactory(model_config, self.logger)
        
        # Use optimized TFT and STGCN models plus LightGBM
        enabled_models = ["tft", "stgcn", "lightgbm"]
        self.logger.info(f"Using optimized models and LightGBM: {enabled_models}")
        
        # Get target column from config
        self.target_column = model_config.get("target_column", "trip_count")
        
        # Set up optimized TFT model
        self.logger.info("Creating Optimized Temporal Fusion Transformer model")
        tft_config = self.config.get("models", {}).get("tft", {}).copy()
        tft_config["is_using_enhanced_features"] = True
        tft_config["use_optimized_version"] = True  # Flag to use optimized version
        self.models["tft"] = self.model_factory.create_model("tft", tft_config)
        
        # Set up optimized STGCN model
        self.logger.info("Creating Optimized Spatio-Temporal GNN model")
        stgcn_config = self.config.get("models", {}).get("stgcn", {}).copy()
        stgcn_config["is_using_enhanced_features"] = True
        stgcn_config["use_optimized_version"] = True  # Flag to use optimized version
        self.models["stgcn"] = self.model_factory.create_model("stgcn", stgcn_config)
        
        # Set up LightGBM model
        self.logger.info("Creating LightGBM model")
        lightgbm_config = self.config.get("models", {}).get("lightgbm", {}).copy()
        lightgbm_config["is_using_enhanced_features"] = True
        self.models["lightgbm"] = self.model_factory.create_model("lightgbm", lightgbm_config)
        
        # Enable ensemble model with the optimized models
        if model_config.get("ensemble", {}).get("enabled", True):
            self.logger.info("Creating Ensemble model with optimized base models")
            # Create ensemble model config that includes the base models
            ensemble_config = model_config.get("ensemble", {}).copy()
            ensemble_config["base_models"] = self.models
            ensemble_config["is_using_enhanced_features"] = True
            self.models["ensemble"] = self.model_factory.create_model("ensemble", ensemble_config)
        
        self.logger.info(f"Model setup complete. Models: {list(self.models.keys())}")
        self.logger.info("Note: Using optimized versions of TFT and STGCN models with enhanced features")
    
    def train_models(self):
        """
        Train all enabled models.
        """
        if not self.models:
            self.logger.error("No models to train. Run setup_models() first.")
            return
        
        self.logger.info("Training models")
        
        # Check if we're using enhanced features
        is_using_enhanced = isinstance(self.feature_engineer, EnhancedFeatureEngineer)
        self.logger.info(f"Using {'enhanced' if is_using_enhanced else 'standard'} feature engineering")
        
        # Update model configs to include feature engineering information
        for model_name, model in self.models.items():
            model.is_using_enhanced_features = is_using_enhanced
            if hasattr(model, 'model_config'):
                model.model_config['is_using_enhanced_features'] = is_using_enhanced
        
        # Identify key columns for different model types
        # Essential datetime columns
        time_columns = ['hour_start'] if 'hour_start' in self.train_data.columns else []
        # Geographic columns for spatial models - only include columns that exist
        spatial_columns = []
        if 'start_location_id' in self.train_data.columns:
            spatial_columns.append('start_location_id')
        if 'end_location_id' in self.train_data.columns:
            spatial_columns.append('end_location_id')
        self.logger.info(f"Using {len(spatial_columns)} spatial columns: {spatial_columns}")
        # Weather columns if available
        weather_base_cols = [col for col in self.train_data.columns if col.startswith('weather_')]
        weather_derived_cols = [col for col in self.train_data.columns if any(col.startswith(p) for p in ['temp_', 'rain_', 'wind_', 'is_heavy_rain', 'is_raining', 'is_snowing', 'is_heatwave', 'is_freezing', 'is_high_wind', 'heat_index', 'wind_chill'])]
        weather_columns = weather_base_cols + weather_derived_cols
        
        self.logger.info(f"Found {len(weather_columns)} weather-related columns")
        
        # For all models, ensure we have the feature columns
        if not self.feature_columns:
            self.logger.warning("Feature columns list is empty, using all non-target columns")
            self.feature_columns = [col for col in self.train_data.columns if col != self.target_column]
            
        # Add essential columns to features
        all_columns = list(set(self.feature_columns + time_columns + spatial_columns))
        
        # Create appropriate training datasets with robust error handling
        # Ensure all requested columns actually exist in the data
        available_features = [col for col in self.feature_columns if col in self.train_data.columns]
        if len(available_features) < len(self.feature_columns):
            missing = set(self.feature_columns) - set(available_features)
            self.logger.warning(f"Missing {len(missing)} requested feature columns: {list(missing)[:5]}...")
        
        # Standard features for basic models
        X_train = self.train_data[available_features].copy()
        y_train = self.train_data[self.target_column].copy()
        
        # Extended features for time series models
        available_time_cols = [col for col in all_columns if col in self.train_data.columns]
        X_train_with_time = self.train_data[available_time_cols].copy()
        self.logger.info(f"Using {len(available_time_cols)} columns for time-aware models")
        
        # Spatial features for graph models
        available_spatial_cols = list(set([col for col in self.feature_columns + spatial_columns if col in self.train_data.columns]))
        X_train_spatial = self.train_data[available_spatial_cols].copy()
        
        # Do the same for validation data
        X_val = self.val_data[available_features].copy()
        y_val = self.val_data[self.target_column].copy()
        X_val_with_time = self.val_data[available_time_cols].copy()
        X_val_spatial = self.val_data[available_spatial_cols].copy()
        
        # Train each model
        for model_name, model in self.models.items():
            # MODIFIED: Only train LightGBM model
            if model_name != "lightgbm":
                self.logger.info(f"Skipping {model_name} model - only training LightGBM")
                continue
                
            self.logger.info(f"Training {model_name} model")
            
            # Use different datasets based on model type
            if model_name == "tft":
                # TFT model needs time information
                self.logger.info(f"Using time-aware dataset for {model_name} model")
                train_data = X_train_with_time
                val_data = X_val_with_time
                
                # Check if we have weather columns and log them
                weather_cols_in_data = [col for col in train_data.columns if col.startswith('weather_')]
                if weather_cols_in_data:
                    self.logger.info(f"TFT model will use {len(weather_cols_in_data)} weather columns")
                    
                # Update model config with time column info
                model.time_idx_name = 'hour_start'
                    
            elif model_name == "stgcn":
                # STGCN model needs spatial information
                self.logger.info(f"Using spatial dataset for {model_name} model")
                train_data = X_train_spatial
                val_data = X_val_spatial
                
                # Update location id column for the graph model
                model.location_id_col = 'start_location_id'
                
                # If we have weather columns, log them for reference
                weather_cols_in_data = [col for col in train_data.columns if col.startswith('weather_')]
                if weather_cols_in_data:
                    self.logger.info(f"STGCN model will use {len(weather_cols_in_data)} weather columns")
                    
            elif model_name == "lightgbm":
                # Use ALL available numeric features for LightGBM, not just the feature_columns subset
                self.logger.info(f"Using ALL available numeric features for {model_name} model")
                
                # Get all columns except the target, datetime columns, and potential leakage features
                all_numeric_cols = [col for col in self.train_data.columns 
                                   if col != self.target_column 
                                   and not pd.api.types.is_datetime64_any_dtype(self.train_data[col])]
                
                self.logger.info(f"Found {len(all_numeric_cols)} safe numeric features out of {len(self.train_data.columns)} total columns")
                
                # Create dataframes with filtered features
                train_data = self.train_data[all_numeric_cols].copy()
                val_data = self.val_data[all_numeric_cols].copy()
                
                # Store filtered feature list for prediction
                self.lightgbm_feature_columns = all_numeric_cols
                self.logger.info(f"LightGBM will use {train_data.shape[1]} features instead of the standard {X_train.shape[1]} features")
                self.logger.info(f"Saved {len(self.lightgbm_feature_columns)} feature names for LightGBM prediction")
                
                # Make sure models have feature names updated
                if hasattr(model, 'feature_names'):
                    model.feature_names = train_data.columns.tolist()
                    
            else:
                # Standard models use regular feature columns
                self.logger.info(f"Using standard feature dataset for {model_name} model")
                train_data = X_train
                val_data = X_val
            
            # Make sure models have feature names updated
            if hasattr(model, 'feature_names'):
                model.feature_names = train_data.columns.tolist()
                
            # Train with validation data
            try:
                self.logger.info(f"Starting training for {model_name} model with {train_data.shape[1]} features")
                model.fit(train_data, y_train, validation_data=(val_data, y_val))
                self.logger.info(f"Finished training {model_name} model")
            except Exception as e:
                self.logger.error(f"Error training {model_name} model: {str(e)}")
                # Continue with other models even if one fails
                continue
            
            # Save model
            model_path = self.results_dir / f"model_{model_name}.pkl"
            model.save(model_path)
            self.logger.info(f"Saved {model_name} model to {model_path}")
        
        # Skip ensemble model when only training LightGBM
        if "ensemble" in self.models:
            self.logger.info("Skipping ensemble model since we're only training LightGBM")
            # Mark ensemble as not fitted to avoid evaluation errors
            self.models["ensemble"].is_fitted = False

    
    def evaluate_models(self):
        """
        Evaluate all trained models on the test set.
        """
        if not self.models:
            self.logger.error("No models to evaluate. Run train_models() first.")
            return
        
        self.logger.info("Evaluating models on test data")
        
        # Initialize evaluator
        eval_dir = self.results_dir / "evaluation"
        eval_dir.mkdir(exist_ok=True)
        
        self.evaluator = ModelEvaluator(output_dir=str(eval_dir))
        
        # Initialize visualizer
        viz_dir = self.results_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        self.visualizer = ForecastVisualizer(output_dir=str(viz_dir))
        
        # Prepare data
        if hasattr(self, 'lightgbm_feature_columns') and self.lightgbm_feature_columns:
            # Use the expanded feature set for LightGBM
            self.logger.info(f"Using full feature set for evaluation: {len(self.lightgbm_feature_columns)} features")
            X_test = self.test_data[self.lightgbm_feature_columns]
        else:
            # Use standard feature set
            self.logger.info("Using standard feature set for evaluation")
            X_test = self.test_data[self.feature_columns]
        
        y_test = self.test_data[self.target_column]
        
        # Evaluate each model
        model_predictions = {}
        model_uncertainties = {}
        
        for model_name, model in self.models.items():
            # Only evaluate LightGBM model
            if model_name != "lightgbm":
                self.logger.info(f"Skipping evaluation of {model_name} - only evaluating LightGBM")
                continue
            self.logger.info(f"Evaluating {model_name} model")
            
            # Get predictions and uncertainty
            predictions, uncertainty = model.predict(X_test, return_uncertainty=True)
            model_predictions[model_name] = predictions
            model_uncertainties[model_name] = uncertainty
            
            # Evaluate predictions
            metrics = self.evaluator.evaluate_model(
                model_name, 
                y_test, 
                predictions, 
                uncertainty=uncertainty,
                groupby=self.test_data["start_location_id"] if "start_location_id" in self.test_data.columns else None,
                timestamps=self.test_data["hour_start"] if "hour_start" in self.test_data.columns else None
            )
            
            self.logger.info(f"{model_name} evaluation complete. Key metrics: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
            
            # Plot time series forecasts
            if "hour_start" in self.test_data.columns:
                self.visualizer.plot_time_series_forecast(
                    self.test_data["hour_start"],
                    y_test,
                    predictions,
                    uncertainty=uncertainty,
                    title=f"{model_name} - Test Set Forecast",
                    filename=f"{model_name}_time_series.png"
                )
            
            # Plot error distribution
            self.visualizer.plot_error_distribution(
                y_test,
                predictions,
                model_name=model_name,
                filename=f"{model_name}_error_distribution.png"
            )
            
            # Plot feature importance with safety checks
            if hasattr(model, "feature_importance") and model.feature_importance is not None:
                try:
                    self.logger.info(f"Plotting feature importance for {model_name}")
                    # Make sure we have valid feature names and importance values
                    if isinstance(model.feature_importance, pd.DataFrame) and 'feature' in model.feature_importance.columns and 'importance' in model.feature_importance.columns:
                        # Get top features
                        top_features = model.feature_importance.sort_values('importance', ascending=False).head(20)
                        feature_names = top_features['feature'].tolist()
                        importances = top_features['importance'].tolist()
                        
                        if len(feature_names) > 0 and len(importances) > 0:
                            self.visualizer.plot_feature_importance(
                                feature_names,
                                importances,
                                model_name=model_name,
                                filename=f"{model_name}_feature_importance.png",
                                show=False
                            )
                except Exception as e:
                    self.logger.warning(f"Error plotting feature importance: {str(e)}")
                    # Continue with evaluation even if feature importance fails
            
            # Skip model comparison since we only have one model
            self.logger.info("Skipping model comparison since we're only evaluating LightGBM")
            
            # Generate weekly hourly forecast visualization
            try:
                self.logger.info("Generating weekly hourly forecast visualization")
                
                # Create a dataframe with test data and predictions for visualization
                forecast_df = self.test_data.copy()
                forecast_df['predicted_demand'] = predictions
                
                # Generate the weekly hourly visualization
                self.visualizer.plot_weekly_hourly_forecast(
                    data=forecast_df,
                    time_col='hour_start',
                    value_col='predicted_demand',
                    location_col='start_location_id' if 'start_location_id' in forecast_df.columns else None,
                    title=f"Weekly Hourly Demand Forecast - {model_name.upper()}",
                    filename=f"{model_name}_weekly_hourly_forecast.png",
                    show=False
                )
                self.logger.info("Weekly hourly forecast visualization completed")
            except Exception as e:
                self.logger.warning(f"Error generating weekly hourly forecast: {str(e)}")
            
            # Save evaluation results
            self.evaluator.save_evaluation_results("model_evaluation_results.json")
            
            # Since we only have one model (LightGBM), it's automatically the best
            best_model = "lightgbm"
            self.logger.info(f"Best model: {best_model} (only model evaluated)")
        else:
            self.logger.warning("No models were successfully evaluated")
            # Set default best model to lightgbm since that's the only one we're training
            best_model = "lightgbm"
        
        return best_model, model_predictions, model_uncertainties
    
    def optimize_pricing(self, model_name: str = "ensemble"):
        """
        Optimize surge pricing based on demand forecasts.
        
        Args:
            model_name: Name of the model to use for forecasting
        """
        if model_name not in self.models:
            self.logger.error(f"Model {model_name} not found. Run train_models() first.")
            return
        
        self.logger.info(f"Optimizing surge pricing using {model_name} model")
        
        # Initialize surge optimizer
        pricing_dir = self.results_dir / "pricing"
        pricing_dir.mkdir(exist_ok=True)
        
        self.surge_optimizer = SurgePricingOptimizer(self.config, self.logger)
        
        # Initialize price evaluator
        self.price_evaluator = PriceEvaluator(output_dir=str(pricing_dir))
        
        # Prepare data for forecasting with the correct feature set
        if hasattr(self, 'lightgbm_feature_columns') and self.lightgbm_feature_columns and model_name == 'lightgbm':
            # Use the expanded feature set that was used for training LightGBM
            self.logger.info(f"Using full feature set for pricing optimization: {len(self.lightgbm_feature_columns)} features")
            X_forecast = self.test_data[self.lightgbm_feature_columns]
        else:
            # Use standard feature set
            self.logger.info("Using standard feature set for pricing optimization")
            X_forecast = self.test_data[self.feature_columns]
        
        # Make demand predictions
        self.logger.info("Generating demand forecast for pricing optimization")
        predictions, uncertainty = self.models[model_name].predict(X_forecast, return_uncertainty=True)
        
        # Create forecast DataFrame
        forecast_df = self.test_data[["hour_start", "start_location_id"]].copy()
        forecast_df["predicted_demand"] = predictions
        forecast_df["prediction_uncertainty"] = uncertainty
        
        # Add location coordinates if available
        if "latitude" in self.test_data.columns and "longitude" in self.test_data.columns:
            forecast_df["latitude"] = self.test_data["latitude"]
            forecast_df["longitude"] = self.test_data["longitude"]
        
        # Optimize pricing
        self.logger.info("Calculating optimal surge multipliers")
        pricing_results = self.surge_optimizer.calculate_surge_multipliers(forecast_df)
        
        # Save pricing results
        pricing_results.to_parquet(pricing_dir / "surge_pricing_results.parquet", index=False)
        self.logger.info(f"Saved pricing results to {pricing_dir / 'surge_pricing_results.parquet'}")
        
        # Create a simple baseline for comparison (fixed multiplier)
        baseline_multiplier = 1.2  # Common baseline multiplier
        
        # Evaluate pricing strategy
        evaluation_results = self.price_evaluator.evaluate_pricing_strategy(
            pricing_results,
            baseline_multiplier=baseline_multiplier
        )
        
        # Save evaluation results
        self.price_evaluator.save_evaluation_results(evaluation_results, "pricing_evaluation.json")
        
        # Analyze pricing fairness
        fairness_results = self.price_evaluator.evaluate_pricing_fairness(pricing_results)
        self.price_evaluator.save_evaluation_results(fairness_results, "pricing_fairness.json")
        
        # Create visualizations
        viz_dir = self.results_dir / "visualizations"
        
        # Generate pricing heatmap by location and hour if spatial data is available
        if "latitude" in pricing_results.columns and "longitude" in pricing_results.columns:
            # Group by location to get average multiplier
            location_avg = pricing_results.groupby("start_location_id").agg({
                "surge_multiplier": "mean",
                "latitude": "first",
                "longitude": "first"
            }).reset_index()
            
            # Plot spatial distribution of pricing
            self.visualizer.plot_spatial_performance(
                location_avg,
                metric_name="surge_multiplier",
                title="Spatial Distribution of Surge Multipliers",
                filename="surge_multiplier_spatial.png"
            )
        
        # Generate pricing by hour heatmap
        if "hour" in pricing_results.columns:
            # Pivot data for heatmap
            hour_location_data = pricing_results[["hour", "start_location_id", "surge_multiplier"]]
            
            # Create heatmap
            self.visualizer.plot_heatmap(
                hour_location_data,
                x_col="hour",
                y_col="start_location_id",
                value_col="surge_multiplier",
                title="Surge Multipliers by Hour and Location",
                filename="surge_multiplier_heatmap.png"
            )
        
        self.logger.info("Pricing optimization and evaluation complete")
        return pricing_results, evaluation_results, fairness_results
    
    def run_all(self):
        """
        Run the full pipeline from data loading to pricing optimization.
        """
        self.logger.info("Starting full pipeline execution")
        
        # Step 1: Set up data pipeline
        self.setup_data_pipeline()
        
        # Step 2: Set up models
        self.setup_models()
        
        # Step 3: Train models
        self.train_models()
        
        # Step 4: Evaluate models
        best_model, _, _ = self.evaluate_models()
        
        # Step 5: Optimize pricing
        self.optimize_pricing(model_name=best_model)
        
        # Step 6: Generate summary report
        self.generate_summary_report()
        
        self.logger.info("Full pipeline execution complete")
    
    def generate_summary_report(self):
        """
        Generate a summary report of the entire process.
        """
        self.logger.info("Generating summary report")
        
        # Create report directory
        report_dir = self.results_dir / "report"
        report_dir.mkdir(exist_ok=True)
        
        # Gather metrics for report
        report = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "data": {
                "train_size": len(self.train_data) if self.train_data is not None else 0,
                "val_size": len(self.val_data) if self.val_data is not None else 0,
                "test_size": len(self.test_data) if self.test_data is not None else 0,
                "n_features": len(self.feature_columns) if self.feature_columns is not None else 0,
                "features": self.feature_columns[:10] + ["..."] if len(self.feature_columns or []) > 10 else self.feature_columns,
            },
            "models": list(self.models.keys()) if self.models else [],
        }
        
        # Add evaluation metrics if available
        if hasattr(self, "evaluator") and self.evaluator is not None and self.evaluator.metrics:
            # Create a simplified metrics summary
            metrics_summary = {}
            for model_name, metrics in self.evaluator.metrics.items():
                metrics_summary[model_name] = {
                    "rmse": metrics.get("rmse"),
                    "mae": metrics.get("mae"),
                    "mape": metrics.get("mape"),
                    "r2": metrics.get("r2")
                }
            report["evaluation"] = metrics_summary
            
            # Find best model
            comparison_df = self.evaluator.compare_models()
            if not comparison_df.empty:
                best_model = comparison_df.sort_values("rmse").iloc[0]["model_name"]
                report["best_model"] = best_model
        
        # Add pricing summary if available
        pricing_file = self.results_dir / "pricing" / "pricing_evaluation.json"
        if pricing_file.exists():
            import json
            with open(pricing_file, "r") as f:
                pricing_data = json.load(f)
                
            # Extract key pricing metrics
            if "strategy_metrics" in pricing_data:
                strategy_metrics = pricing_data["strategy_metrics"]
                report["pricing"] = {
                    "mean_multiplier": strategy_metrics.get("mean_multiplier"),
                    "max_multiplier": strategy_metrics.get("max_multiplier"),
                    "total_expected_revenue": strategy_metrics.get("total_expected_revenue")
                }
            
            # Add baseline comparison
            if "baseline_comparison" in pricing_data:
                baseline = pricing_data["baseline_comparison"]
                report["pricing"]["revenue_improvement"] = baseline.get("revenue_improvement")
        
        # Save report to file
        import json
        with open(report_dir / "summary_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        # Generate markdown report
        md_report = f"# Ride Demand Forecasting System - Run Summary\n\n"
        md_report += f"**Run ID:** {report['run_id']}\n\n"
        md_report += f"**Timestamp:** {report['timestamp']}\n\n"
        
        md_report += f"## Data Summary\n\n"
        md_report += f"- Train set size: {report['data']['train_size']}\n"
        md_report += f"- Validation set size: {report['data']['val_size']}\n"
        md_report += f"- Test set size: {report['data']['test_size']}\n"
        md_report += f"- Number of features: {report['data']['n_features']}\n\n"
        
        if "evaluation" in report:
            md_report += f"## Model Evaluation\n\n"
            md_report += f"| Model | RMSE | MAE | MAPE | R² |\n"
            md_report += f"|-------|------|-----|------|-----|\n"
            
            for model, metrics in report["evaluation"].items():
                md_report += f"| {model} | {metrics['rmse']:.4f} | {metrics['mae']:.4f} | "
                md_report += f"{metrics['mape']:.2f}% | {metrics['r2']:.4f} |\n"
            
            if "best_model" in report:
                md_report += f"\n**Best model:** {report['best_model']}\n"
        
        if "pricing" in report:
            md_report += f"\n## Pricing Optimization\n\n"
            md_report += f"- Mean surge multiplier: {report['pricing']['mean_multiplier']:.2f}\n"
            md_report += f"- Maximum surge multiplier: {report['pricing']['max_multiplier']:.2f}\n"
            md_report += f"- Total expected revenue: ${report['pricing']['total_expected_revenue']:.2f}\n"
            
            if "revenue_improvement" in report["pricing"]:
                md_report += f"- Revenue improvement vs. baseline: {report['pricing']['revenue_improvement']:.2f}%\n"
        
        # Add links to visualizations
        viz_dir = self.results_dir / "visualizations"
        if viz_dir.exists():
            viz_files = list(viz_dir.glob("*.png"))
            if viz_files:
                md_report += f"\n## Visualizations\n\n"
                for viz_file in viz_files:
                    rel_path = viz_file.relative_to(self.results_dir)
                    md_report += f"- [{viz_file.stem}]({rel_path})\n"
        
        # Save markdown report
        with open(report_dir / "summary_report.md", "w") as f:
            f.write(md_report)
        
        self.logger.info(f"Summary report generated at {report_dir}")
        return report


def setup_arg_parser():
    """Set up argument parser for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Advanced Ride Demand Forecasting System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Path to configuration file"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Run all pipeline
    run_all_parser = subparsers.add_parser("run-all", help="Run the entire pipeline")
    
    # Data pipeline
    data_parser = subparsers.add_parser("data", help="Set up data pipeline only")
    
    # Model training
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument(
        "--models", 
        type=str, 
        nargs="+", 
        choices=["lightgbm", "tft", "stgcn", "ensemble", "all"],
        default=["all"],
        help="Models to train"
    )
    
    # Model evaluation
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate models")
    
    # Pricing optimization
    pricing_parser = subparsers.add_parser("optimize-pricing", help="Optimize surge pricing")
    pricing_parser.add_argument(
        "--model", 
        type=str, 
        default="ensemble",
        help="Model to use for optimization"
    )
    
    return parser


def main():
    """Main entry point for the command-line interface."""
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize runner
    runner = DemandForecastingRunner(config_path=args.config)
    
    if args.command == "run-all":
        runner.run_all()
    
    elif args.command == "data":
        runner.setup_data_pipeline()
    
    elif args.command == "train":
        # First ensure data is loaded
        runner.setup_data_pipeline()
        
        # Set up models
        runner.setup_models()
        
        # Train models
        runner.train_models()
    
    elif args.command == "evaluate":
        # First ensure data is loaded and models are trained
        runner.setup_data_pipeline()
        runner.setup_models()
        runner.train_models()
        
        # Evaluate models
        runner.evaluate_models()
    
    elif args.command == "optimize-pricing":
        # First ensure data is loaded and models are trained
        runner.setup_data_pipeline()
        runner.setup_models()
        runner.train_models()
        
        # Optimize pricing
        runner.optimize_pricing(model_name=args.model)


if __name__ == "__main__":
    main()
