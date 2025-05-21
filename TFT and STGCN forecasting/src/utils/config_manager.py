import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union

class ConfigManager:
    """Advanced configuration manager for the ride demand forecasting system.
    
    Provides robust configuration loading with environment variable interpolation,
    config validation, and hierarchical access to nested configuration values.
    """
    
    def __init__(self, config_path: Union[str, Path]):
        """Initialize the configuration manager.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
            
        self.config: Dict[str, Any] = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load the configuration from the YAML file with environment variable interpolation."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # Interpolate environment variables
        self._interpolate_env_vars(config)
        return config
    
    def _interpolate_env_vars(self, config_dict: Dict[str, Any]) -> None:
        """Recursively interpolate environment variables in the configuration.
        
        Replaces ${ENV_VAR} or $ENV_VAR with the value from environment variables.
        """
        for key, value in config_dict.items():
            if isinstance(value, dict):
                self._interpolate_env_vars(value)
            elif isinstance(value, str):
                # Replace ${VAR} with the environment variable
                if '${' in value and '}' in value:
                    env_var = value.split('${')[1].split('}')[0]
                    env_value = os.environ.get(env_var)
                    if env_value is not None:
                        config_dict[key] = value.replace(f'${{{env_var}}}', env_value)
                # Replace $VAR with the environment variable
                elif value.startswith('$'):
                    env_var = value[1:]
                    env_value = os.environ.get(env_var)
                    if env_value is not None:
                        config_dict[key] = env_value
    
    def get(self, path: str, default: Optional[Any] = None) -> Any:
        """Get a configuration value using dot notation for nested access.
        
        Args:
            path: Dot-separated path to the configuration value (e.g., 'models.lightgbm.enabled')
            default: Default value if the path does not exist
            
        Returns:
            The configuration value or the default
        """
        keys = path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value
    
    def get_full_path(self, relative_path: str) -> Path:
        """Convert a relative path from the config to a full path.
        
        Args:
            relative_path: Relative path from the project root
            
        Returns:
            Full path
        """
        project_root = self.config_path.parent
        return project_root / relative_path
    
    def update_config(self, path: str, value: Any) -> None:
        """Update a configuration value using dot notation.
        
        Args:
            path: Dot-separated path to the configuration value
            value: New value to set
        """
        keys = path.split('.')
        config_ref = self.config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        # Set the value
        config_ref[keys[-1]] = value
    
    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save the current configuration to a YAML file.
        
        Args:
            path: Path to save the configuration file, defaults to the original path
        """
        save_path = Path(path) if path else self.config_path
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    def __contains__(self, key: str) -> bool:
        """Check if a top-level key exists in the configuration."""
        return key in self.config
    
    def __getitem__(self, key: str) -> Any:
        """Get a top-level configuration value."""
        return self.config[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set a top-level configuration value."""
        self.config[key] = value
    
    @property
    def data_paths(self) -> Dict[str, Path]:
        """Get all data paths as absolute Path objects."""
        paths = {}
        data_config = self.get('data', {})
        
        for key, rel_path in data_config.items():
            if key.endswith('_path') and isinstance(rel_path, str):
                paths[key] = self.get_full_path(rel_path)
                
        return paths
