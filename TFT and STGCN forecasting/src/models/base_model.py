import numpy as np
import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple, Any
import os
import pickle
import json
import time
from datetime import datetime

from ..utils.logger import StructuredLogger


class BaseModel(ABC):
    """Abstract base class for all forecasting models.
    
    This class defines the common interface that all models must implement,
    regardless of their underlying architecture or framework. It provides
    a consistent API for training, prediction, evaluation, serialization,
    and hyperparameter optimization.
    
    All specialized model implementations (LightGBM, Temporal Fusion Transformer,
    etc.) must extend this class and implement its abstract methods.
    """
    
    def __init__(self, model_config: Dict, logger: Optional[StructuredLogger] = None):
        """Initialize the base model.
        
        Args:
            model_config: Configuration dictionary for the model
            logger: Logger instance for tracking model operations
        """
        self.model_config = model_config
        self.logger = logger or StructuredLogger("base_model")
        self.model = None
        self.feature_importance = None
        self.metrics = {}
        self.training_time = 0
        self.is_fitted = False
        
        # Track if using enhanced features
        self.is_using_enhanced_features = model_config.get("is_using_enhanced_features", False)
        
        # Track feature types for advanced models
        self.feature_types = {
            "temporal_features": [],
            "spatial_features": [],
            "context_features": [],
            "weather_features": []
        }
        
        self.model_info = {
            "model_type": self.__class__.__name__,
            "creation_time": datetime.now().isoformat(),
            "training_status": "not_trained",
            "feature_count": 0,
            "params": model_config.get("params", {}),
            "is_using_enhanced_features": self.is_using_enhanced_features
        }
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, validation_data: Optional[Tuple] = None) -> Dict:
        """Train the model on the provided data.
        
        Args:
            X: Feature dataframe for training
            y: Target series for training
            validation_data: Optional tuple of (X_val, y_val) for validation during training
            
        Returns:
            Dictionary of training metrics and information
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate point predictions for the input data.
        
        Args:
            X: Feature dataframe for prediction
            
        Returns:
            Array of predicted values
        """
        pass
    
    @abstractmethod
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate predictions with uncertainty intervals.
        
        Args:
            X: Feature dataframe for prediction
            
        Returns:
            Tuple of (mean predictions, lower bounds, upper bounds)
        """
        pass
    
    @abstractmethod    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance or attribution scores.
        
        Returns:
            DataFrame with feature importance information
        """
        pass
    
    def save(self, path: str) -> str:
        """Save the model to disk.
        
        Args:
            path: Directory path to save the model
            
        Returns:
            Full path to the saved model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save an untrained model")
            
        os.makedirs(path, exist_ok=True)
        model_dir = os.path.join(path, self.__class__.__name__)
        os.makedirs(model_dir, exist_ok=True)
        
        # Update model info before saving
        self.model_info["training_status"] = "trained"
        self.model_info["last_saved"] = datetime.now().isoformat()
        self.model_info["metrics"] = self.metrics
        
        # Save metadata in JSON format for easy inspection
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.model_info, f, indent=2)
        
        # Each model implementation should override this method to save model-specific components
        self._save_model_components(model_dir)
        
        self.logger.info(f"Model saved to {model_dir}")
        return model_dir
        
    def _save_model_components(self, model_dir: str) -> None:
        """Save model-specific components to disk.
        
        This method should be overridden by each model implementation to save
        their specific model objects and data structures in a way that avoids
        pickling issues with thread locks and other non-serializable components.
        
        Args:
            model_dir: Directory path to save model components
        """
        # Default implementation just saves model_config as a fallback
        config_path = os.path.join(model_dir, "model_config.json")
        with open(config_path, 'w') as f:
            # Convert any non-serializable values to strings
            config_copy = {}
            for k, v in self.model_config.items():
                try:
                    json.dumps({k: v})
                    config_copy[k] = v
                except (TypeError, OverflowError):
                    config_copy[k] = str(v)
            
            json.dump(config_copy, f, indent=2)
    
    @classmethod
    def load(cls, path: str, logger: Optional[StructuredLogger] = None) -> 'BaseModel':
        """Load a model from disk.
        
        Args:
            path: Path to the saved model directory
            logger: Optional logger instance
            
        Returns:
            Loaded model instance
        """
        # Check if path is a directory or a file
        if os.path.isfile(path):
            # Legacy support for old pickle-based models
            try:
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                if logger:
                    model.logger = logger
                model.logger.info(f"Model loaded from {path}")
                return model
            except Exception as e:
                if logger:
                    log = logger
                else:
                    log = StructuredLogger("base_model")
                log.error(f"Failed to load model from {path}: {str(e)}")
                raise
        
        # New component-based loading - implemented by subclasses
        model_dir = path
        if not os.path.isdir(model_dir):
            raise ValueError(f"Model directory not found: {model_dir}")
            
        # Load metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise ValueError(f"Model metadata not found at {metadata_path}")
            
        with open(metadata_path, 'r') as f:
            model_info = json.load(f)
            
        # Load config
        config_path = os.path.join(model_dir, "model_config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"Model config not found at {config_path}")
            
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        
        # Create a new instance with the loaded config
        instance = cls(model_config, logger)
        instance.model_info = model_info
        instance.metrics = model_info.get("metrics", {})
        instance.is_fitted = model_info.get("training_status", "") == "trained"
        
        # Load model-specific components
        instance._load_model_components(model_dir)
        
        if logger:
            instance.logger = logger
            
        instance.logger.info(f"Model loaded from {model_dir}")
        return instance
    
    def _load_model_components(self, model_dir: str) -> None:
        """Load model-specific components from disk.
        
        This method should be overridden by each model implementation to load
        their specific model objects and data structures.
        
        Args:
            model_dir: Directory path where model components are saved
        """
        # Default implementation does nothing - must be overridden by subclasses
        pass
    
    def get_params(self) -> Dict:
        """Get the current model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        return self.model_config.get("hyperparameters", {})
    
    def set_params(self, params: Dict) -> None:
        """Set new model parameters.
        
        Args:
            params: Dictionary of parameters to set
        """
        if self.is_fitted:
            self.logger.warning("Setting parameters on an already trained model. "
                              "You'll need to retrain for parameters to take effect.")
        
        # Update hyperparameters in the config
        if "hyperparameters" not in self.model_config:
            self.model_config["hyperparameters"] = {}
            
        self.model_config["hyperparameters"].update(params)
        self.model_info["hyperparameters"] = self.model_config["hyperparameters"]
    
    def get_model_info(self) -> Dict:
        """Get information about the model.
        
        Returns:
            Dictionary with model metadata and performance information
        """
        return self.model_info
