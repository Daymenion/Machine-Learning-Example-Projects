import logging
import sys
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any, Optional, Union

class StructuredLogger:
    """Advanced structured logger for the ride demand forecasting system.
    
    Features:
    - Configurable console and file logging
    - Structured JSON logging for machine-readable logs
    - Context manager for tracking operations with timing
    - Automatic exception tracking
    - Log levels customization
    """
    
    def __init__(self, 
                 name: str,
                 log_dir: Optional[Union[str, Path]] = None,
                 console_level: int = logging.INFO,
                 file_level: int = logging.DEBUG,
                 structured_json: bool = True):
        """Initialize the structured logger.
        
        Args:
            name: Logger name
            log_dir: Directory to store log files
            console_level: Logging level for console output
            file_level: Logging level for file output
            structured_json: Whether to use structured JSON logging
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Capture all levels
        self.logger.propagate = False  # Don't propagate to root logger
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
            
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler (if log_dir is provided)
        self.file_handler = None
        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"{name}_{timestamp}.log"
            
            self.file_handler = logging.FileHandler(log_file)
            self.file_handler.setLevel(file_level)
            
            if structured_json:
                file_format = self.JsonFormatter()
            else:
                file_format = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
                )
                
            self.file_handler.setFormatter(file_format)
            self.logger.addHandler(self.file_handler)
    
    class JsonFormatter(logging.Formatter):
        """Custom formatter for structured JSON logs."""
        
        def format(self, record: logging.LogRecord) -> str:
            """Format the log record as a JSON string."""
            log_data = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'name': record.name,
                'level': record.levelname,
                'message': record.getMessage(),
                'module': record.module,
                'filename': record.filename,
                'lineno': record.lineno,
                'thread': record.thread,
                'thread_name': record.threadName,
            }
            
            # Add exception info if available
            if record.exc_info:
                exc_type, exc_value, exc_tb = record.exc_info
                log_data['exception'] = {
                    'type': exc_type.__name__,
                    'message': str(exc_value)
                }
            
            # Add extra attributes
            if hasattr(record, 'props'):
                log_data.update(record.props)
                
            return json.dumps(log_data)
    
    def info(self, msg: str, **kwargs) -> None:
        """Log an info message with optional structured data."""
        self._log(logging.INFO, msg, **kwargs)
    
    def debug(self, msg: str, **kwargs) -> None:
        """Log a debug message with optional structured data."""
        self._log(logging.DEBUG, msg, **kwargs)
    
    def warning(self, msg: str, **kwargs) -> None:
        """Log a warning message with optional structured data."""
        self._log(logging.WARNING, msg, **kwargs)
    
    def error(self, msg: str, **kwargs) -> None:
        """Log an error message with optional structured data."""
        self._log(logging.ERROR, msg, **kwargs)
    
    def critical(self, msg: str, **kwargs) -> None:
        """Log a critical message with optional structured data."""
        self._log(logging.CRITICAL, msg, **kwargs)
    
    def _log(self, level: int, msg: str, **kwargs) -> None:
        """Log a message with the given level and structured data."""
        extra = {'props': kwargs} if kwargs else None
        self.logger.log(level, msg, extra=extra)
    
    def set_level(self, level: int) -> None:
        """Set the logging level for all handlers."""
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)
    
    class LogOperation:
        """Context manager for tracking operations with timing."""
        
        def __init__(self, logger, operation_name: str, level: int = logging.INFO, **context):
            self.logger = logger
            self.operation_name = operation_name
            self.level = level
            self.context = context
            self.start_time = None
            
        def __enter__(self):
            self.start_time = datetime.now()
            self.logger._log(
                self.level,
                f"Starting operation: {self.operation_name}",
                operation=self.operation_name,
                status="started",
                **self.context
            )
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()
            
            if exc_type is None:
                # Operation succeeded
                self.logger._log(
                    self.level,
                    f"Completed operation: {self.operation_name} in {duration:.2f}s",
                    operation=self.operation_name,
                    status="completed",
                    duration_seconds=duration,
                    **self.context
                )
            else:
                # Operation failed
                self.logger._log(
                    logging.ERROR,
                    f"Failed operation: {self.operation_name} after {duration:.2f}s: {exc_val}",
                    operation=self.operation_name,
                    status="failed",
                    duration_seconds=duration,
                    error_type=exc_type.__name__,
                    error_message=str(exc_val),
                    **self.context
                )
                
            # Don't suppress exceptions
            return False
    
    def operation(self, operation_name: str, level: int = logging.INFO, **context):
        """Create a context manager for tracking an operation."""
        return self.LogOperation(self, operation_name, level, **context)
    
    def exception(self, msg: str, **kwargs) -> None:
        """Log an exception message with traceback and structured data."""
        extra = {'props': kwargs} if kwargs else None
        self.logger.exception(msg, extra=extra)
        
    def close(self) -> None:
        """Close all handlers and release resources."""
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)
