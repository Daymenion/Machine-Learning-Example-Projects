from typing import Dict, Optional, Type

from .base_model import BaseModel
from .optimized_tft_model import OptimizedTFTModel
from .optimized_stgcn_model import OptimizedSTGCNModel
from .gbdt_model import LightGBMModel
from .ensemble_model import EnsembleModel
from ..utils.logger import StructuredLogger


class ModelFactory:
    """Factory class for creating different model instances.
    
    This factory encapsulates the creation logic for all model types,
    allowing clients to create model instances without knowing the
    specific implementation details or dependencies of each model type.
    
    The factory pattern makes it easy to add new model types in the future
    and provides a centralized location for model instantiation logic.
    """
    
    def __init__(self, config, logger=None):
        """Initialize the model factory.
        
        Args:
            config: Configuration manager or dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or StructuredLogger("model_factory")
        self.models_config = config.get("models", {})
    
    # Registry of available model types
    MODEL_REGISTRY = {
        "tft": OptimizedTFTModel,
        "stgcn": OptimizedSTGCNModel,
        "lightgbm": LightGBMModel,
        "ensemble": EnsembleModel,
    }
    
    def create_model(self, model_type: str, model_config: Dict = None) -> BaseModel:
        """Create a model instance of the specified type.
        
        Args:
            model_type: Type of model to create (e.g., 'tft', 'stgcn', 'lightgbm', 'ensemble')
            model_config: Optional configuration dictionary for the model, if None uses config from models_config
            
        Returns:
            Instance of the requested model type
            
        Raises:
            ValueError: If the requested model type is not registered
        """
        if model_type not in self.MODEL_REGISTRY:
            registered_models = ", ".join(self.MODEL_REGISTRY.keys())
            raise ValueError(
                f"Unknown model type: {model_type}. Available models: {registered_models}"
            )
            
        # If no specific model_config provided, use the one from the global config
        if model_config is None:
            model_config = self.models_config.get(model_type, {})
            
        # Create the model using instance attributes
        return self._create_model_instance(model_type, model_config)

    def _create_model_instance(self, model_type: str, model_config: Dict) -> BaseModel:
        """Create and initialize a model instance.
        
        Args:
            model_type: Type of model to create
            model_config: Configuration for the model
            
        Returns:
            Initialized model instance
        """
        model_class = self.MODEL_REGISTRY[model_type]
        self.logger.info(f"Initializing standard {model_type} model")
        return model_class(model_config, self.logger)
        
    @classmethod
    def create_model_cls(cls, model_type: str, model_config: Dict, logger: Optional[StructuredLogger] = None) -> BaseModel:
        """Create a model instance of the specified type (class method version).
        
        Args:
            model_type: Type of model to create (e.g., 'tft', 'stgcn', 'lightgbm', 'ensemble')
            model_config: Configuration dictionary for the model
            logger: Optional logger instance
            
        Returns:
            Instance of the requested model type
            
        Raises:
            ValueError: If the requested model type is not registered
        """
        if model_type not in cls.MODEL_REGISTRY:
            registered_models = ", ".join(cls.MODEL_REGISTRY.keys())
            raise ValueError(
                f"Unknown model type: {model_type}. Available models: {registered_models}"
            )
            
        # Instantiate the model class with the provided configuration
        model_class = cls.MODEL_REGISTRY[model_type]
        if logger:
            logger.info(f"Initializing {model_type} model")
        return model_class(model_config, logger)
    
    @classmethod
    def register_model(cls, model_name: str, model_class: Type[BaseModel]) -> None:
        """Register a new model type.
        
        Args:
            model_name: Name to register the model under
            model_class: Model class to register
            
        Raises:
            TypeError: If the model_class doesn't inherit from BaseModel
        """
        if not issubclass(model_class, BaseModel):
            raise TypeError(f"Model class must inherit from BaseModel")
            
        cls.MODEL_REGISTRY[model_name] = model_class
    
    @classmethod
    def get_available_models(cls) -> Dict[str, Type[BaseModel]]:
        """Get a dictionary of all registered model types.
        
        Returns:
            Dictionary mapping model names to model classes
        """
        return cls.MODEL_REGISTRY.copy()
