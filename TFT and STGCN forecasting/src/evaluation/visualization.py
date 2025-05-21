from ..utils.logger import StructuredLogger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
import os
from pathlib import Path
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec
import matplotlib.ticker as ticker

class ForecastVisualizer:
    """Class for visualizing forecast results and model performance.
    
    This class provides comprehensive visualization tools for analyzing
    time series forecasts, model errors, and comparative performance.
    
    Attributes:
        logger: Logger instance for logging visualization information
        output_dir: Directory to save visualization outputs
        style: Matplotlib style to use for plots
        palette: Color palette for plots
        figsize: Default figure size
    """
    
    def __init__(self, output_dir: Optional[str] = None, 
                 style: str = 'seaborn-v0_8-whitegrid',
                 palette: Optional[str] = None, 
                 figsize: Tuple[int, int] = (12, 8)):
        """Initialize the ForecastVisualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
            style: Matplotlib style to use for plots
            palette: Color palette for plots
            figsize: Default figure size
        """
        self.logger = StructuredLogger("forecast_visualizer")
        self.output_dir = output_dir
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.style = style
        self.palette = palette or 'viridis'
        self.figsize = figsize
        
        # Set default plotting style
        plt.style.use(self.style)
        sns.set_palette(self.palette)
        
        self.logger.info("ForecastVisualizer initialized")
    
    def plot_time_series_forecast(self, timestamps: pd.Series, 
                                y_true: Union[np.ndarray, pd.Series],
                                y_pred: Union[np.ndarray, pd.Series],
                                uncertainty: Optional[Union[np.ndarray, pd.Series]] = None,
                                title: str = "Time Series Forecast",
                                y_label: str = "Demand",
                                filename: Optional[str] = None,
                                highlight_anomalies: bool = False,
                                anomaly_threshold: float = 0.3,
                                show: bool = True) -> plt.Figure:
        """Plot time series forecasts against actual values.
        
        Args:
            timestamps: Series of timestamps
            y_true: Ground truth values
            y_pred: Predicted values
            uncertainty: Optional prediction uncertainty (std deviation)
            title: Plot title
            y_label: Y-axis label
            filename: Optional filename to save the plot
            highlight_anomalies: Whether to highlight points with large errors
            anomaly_threshold: Threshold for highlighting anomalies (as fraction of max value)
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Convert inputs to numpy arrays
            timestamps_array = pd.to_datetime(timestamps)
            y_true_array = np.array(y_true)
            y_pred_array = np.array(y_pred)
            
            # Plot actual values
            ax.plot(timestamps_array, y_true_array, 'o-', label='Actual', 
                   alpha=0.7, markersize=4, color='#1f77b4')
            
            # Plot predicted values
            ax.plot(timestamps_array, y_pred_array, 's-', label='Forecast', 
                   alpha=0.7, markersize=4, color='#ff7f0e')
            
            # Plot uncertainty if provided
            if uncertainty is not None:
                uncertainty_array = np.array(uncertainty)
                ax.fill_between(timestamps_array, 
                               y_pred_array - uncertainty_array,
                               y_pred_array + uncertainty_array,
                               alpha=0.2, color='#ff7f0e',
                               label='Prediction Interval')
            
            # Highlight anomalies if requested
            if highlight_anomalies:
                errors = np.abs(y_true_array - y_pred_array)
                threshold = anomaly_threshold * np.max(y_true_array)
                anomaly_indices = errors > threshold
                
                if np.any(anomaly_indices):
                    ax.scatter(timestamps_array[anomaly_indices], 
                              y_true_array[anomaly_indices],
                              color='red', s=80, alpha=0.7, marker='*',
                              label='Anomalies', zorder=5)
            
            # Format x-axis for datetime
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            if len(timestamps_array) > 20:
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45)
            
            # Set labels and title
            ax.set_xlabel('Time')
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.legend(loc='best')
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Add error statistics as text
            rmse = np.sqrt(np.mean((y_true_array - y_pred_array) ** 2))
            mae = np.mean(np.abs(y_true_array - y_pred_array))
            
            stats_text = f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}'
            plt.figtext(0.02, 0.02, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            # Save figure if filename provided
            if filename and self.output_dir:
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved forecast plot to {filepath}")
            
            if not show:
                plt.close(fig)
            
            return fig
    
    def plot_error_distribution(self, y_true: Union[np.ndarray, pd.Series],
                               y_pred: Union[np.ndarray, pd.Series],
                               model_name: str = "Model",
                               bins: int = 30,
                               filename: Optional[str] = None,
                               show: bool = True) -> plt.Figure:
        """Plot distribution of forecast errors.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            model_name: Name of the model
            bins: Number of histogram bins
            filename: Optional filename to save the plot
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        with plt.style.context(self.style):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
            
            # Convert to numpy arrays
            y_true_array = np.array(y_true)
            y_pred_array = np.array(y_pred)
            
            # Calculate errors
            errors = y_pred_array - y_true_array
            abs_errors = np.abs(errors)
            
            # Histogram of errors
            sns.histplot(errors, bins=bins, kde=True, ax=ax1, 
                        color='#2c7fb8', alpha=0.7)
            ax1.axvline(x=0, color='red', linestyle='--', alpha=0.8)
            ax1.set_title(f"Error Distribution - {model_name}")
            ax1.set_xlabel("Forecast Error")
            ax1.set_ylabel("Frequency")
            
            # Add error statistics
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            ax1.text(0.05, 0.95, f"Mean: {mean_error:.2f}\nStd: {std_error:.2f}",
                    transform=ax1.transAxes, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8),
                    verticalalignment='top')
            
            # Q-Q plot of errors
            from scipy import stats
            stats.probplot(errors, dist="norm", plot=ax2)
            ax2.set_title("Q-Q Plot of Errors")
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure if filename provided
            if filename and self.output_dir:
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved error distribution plot to {filepath}")
            
            if not show:
                plt.close(fig)
            
            return fig
    
    def plot_actual_vs_predicted(self, y_true: Union[np.ndarray, pd.Series],
                                y_pred: Union[np.ndarray, pd.Series],
                                model_name: str = "Model",
                                uncertainty: Optional[Union[np.ndarray, pd.Series]] = None,
                                filename: Optional[str] = None,
                                show: bool = True) -> plt.Figure:
        """Plot actual vs predicted values with a regression line.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            model_name: Name of the model
            uncertainty: Optional prediction uncertainty
            filename: Optional filename to save the plot
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Convert to numpy arrays
            y_true_array = np.array(y_true)
            y_pred_array = np.array(y_pred)
            
            # Create scatter plot with or without uncertainty
            if uncertainty is not None:
                uncertainty_array = np.array(uncertainty)
                # Normalize uncertainty for color mapping
                norm_uncertainty = uncertainty_array / uncertainty_array.max()
                
                # Create a custom colormap
                cmap = LinearSegmentedColormap.from_list(
                    'uncertainty_cmap', [(0, '#1f77b4'), (1, '#ff7f0e')])
                
                scatter = ax.scatter(y_true_array, y_pred_array, 
                                   c=norm_uncertainty, cmap=cmap, 
                                   alpha=0.7, s=30, edgecolors='white', linewidth=0.5)
                
                # Add a colorbar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Normalized Uncertainty')
            else:
                ax.scatter(y_true_array, y_pred_array, 
                         alpha=0.7, s=30, color='#1f77b4',
                         edgecolors='white', linewidth=0.5)
            
            # Calculate and plot the regression line
            z = np.polyfit(y_true_array, y_pred_array, 1)
            p = np.poly1d(z)
            ax.plot(y_true_array, p(y_true_array), '#e41a1c', linestyle='--')
            
            # Add y=x reference line
            min_val = min(np.min(y_true_array), np.min(y_pred_array))
            max_val = max(np.max(y_true_array), np.max(y_pred_array))
            padding = (max_val - min_val) * 0.05
            ax.plot([min_val - padding, max_val + padding], 
                   [min_val - padding, max_val + padding], 
                   'k-', alpha=0.3)
            
            # Set equal aspect
            ax.set_aspect('equal')
            
            # Set labels, title and grid
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(f"Actual vs Predicted - {model_name}")
            ax.grid(True, alpha=0.3)
            
            # Add r² and equation to the plot
            r2 = np.corrcoef(y_true_array, y_pred_array)[0, 1]**2
            equation = f"y = {z[0]:.3f}x + {z[1]:.3f}"
            ax.text(0.05, 0.95, f"$R^2 = {r2:.3f}$\n{equation}",
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.8),
                   verticalalignment='top')
            
            plt.tight_layout()
            
            # Save figure if filename provided
            if filename and self.output_dir:
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved actual vs predicted plot to {filepath}")
            
            if not show:
                plt.close(fig)
            
            return fig
    
    def plot_feature_importance(self, feature_names: List[str], 
                               importances: Union[np.ndarray, pd.Series], 
                               model_name: str = "Model",
                               top_n: int = 20,
                               filename: Optional[str] = None,
                               show: bool = True) -> plt.Figure:
        """Plot feature importance for a model.
        
        Args:
            feature_names: List of feature names
            importances: Feature importance values
            model_name: Name of the model
            top_n: Number of top features to display
            filename: Optional filename to save the plot
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        with plt.style.context(self.style):
            # Create DataFrame and sort by importance
            importance_df = pd.DataFrame({
                'feature': feature_names[:len(importances)],
                'importance': importances
            })
            importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
            
            # Create horizontal bar chart
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Plot horizontal bars
            sns.barplot(x='importance', y='feature', data=importance_df, 
                       ax=ax, palette='viridis')
            
            # Set labels and title
            ax.set_title(f"Feature Importance - {model_name}")
            ax.set_xlabel('Importance')
            ax.set_ylabel('Features')
            
            # Add grid lines
            ax.grid(True, axis='x', alpha=0.3)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if filename provided
            if filename and self.output_dir:
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved feature importance plot to {filepath}")
            
            if not show:
                plt.close(fig)
            
            return fig
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame,
                            metric_columns: List[str] = ['rmse', 'mae', 'mape', 'r2'],
                            filename: Optional[str] = None,
                            show: bool = True) -> plt.Figure:
        """Plot comparison of multiple models' performance metrics.
        
        Args:
            comparison_df: DataFrame with model comparison metrics
            metric_columns: List of metric columns to include in the plot
            filename: Optional filename to save the plot
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        with plt.style.context(self.style):
            # Validate input
            if 'model_name' not in comparison_df.columns:
                self.logger.error("comparison_df must contain a 'model_name' column")
                return None
            
            # Filter to include only requested metrics that exist in the dataframe
            available_metrics = [col for col in metric_columns if col in comparison_df.columns]
            if not available_metrics:
                self.logger.error("None of the requested metrics are available in the dataframe")
                return None
            
            # Determine number of metrics to plot
            n_metrics = len(available_metrics)
            n_models = len(comparison_df)
            
            # Create subplots - one for each metric
            fig, axes = plt.subplots(n_metrics, 1, figsize=(self.figsize[0], 3 * n_metrics))
            
            # If only one metric, axes won't be an array
            if n_metrics == 1:
                axes = [axes]
            
            # Plot each metric
            for i, metric in enumerate(available_metrics):
                ax = axes[i]
                
                # Sort by this metric for this plot
                # For R², higher is better, so sort descending
                ascending = metric != 'r2'
                df_sorted = comparison_df.sort_values(metric, ascending=ascending).reset_index(drop=True)
                
                # Create bar plot
                bars = sns.barplot(x=metric, y='model_name', data=df_sorted, ax=ax, 
                                 palette='viridis')
                
                # Add value labels to bars
                for j, v in enumerate(df_sorted[metric]):
                    ax.text(v + (0.01 * df_sorted[metric].max()), j, f"{v:.3f}", 
                           va='center', fontsize=9)
                
                # Set labels and title
                metric_name = metric.upper() if metric.lower() == 'rmse' or metric.lower() == 'mae' else metric
                ax.set_title(f"{metric_name} Comparison")
                ax.set_xlabel(metric_name)
                ax.set_ylabel("")
                
                # Add grid lines
                ax.grid(True, axis='x', alpha=0.3)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if filename provided
            if filename and self.output_dir:
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved model comparison plot to {filepath}")
            
            if not show:
                plt.close(fig)
            
            return fig
    
    def plot_temporal_performance(self, timestamps: pd.Series,
                                metrics: Dict[str, np.ndarray],
                                title: str = "Temporal Performance Metrics",
                                filename: Optional[str] = None,
                                show: bool = True) -> plt.Figure:
        """Plot performance metrics over time.
        
        Args:
            timestamps: Series of timestamps
            metrics: Dictionary mapping metric names to arrays of values
            title: Plot title
            filename: Optional filename to save the plot
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        with plt.style.context(self.style):
            # Count the number of metrics to plot
            n_metrics = len(metrics)
            
            # Create figure with subplots - one row per metric
            fig, axes = plt.subplots(n_metrics, 1, figsize=(self.figsize[0], 3 * n_metrics),
                                    sharex=True)
            
            # If only one metric, axes won't be an array
            if n_metrics == 1:
                axes = [axes]
            
            # Convert timestamps to datetime
            timestamps_array = pd.to_datetime(timestamps)
            
            # Plot each metric
            for i, (metric_name, metric_values) in enumerate(metrics.items()):
                ax = axes[i]
                
                # Plot the metric over time
                ax.plot(timestamps_array, metric_values, '-o', markersize=4)
                
                # Add a smoothed trendline
                try:
                    from scipy.signal import savgol_filter
                    window_length = min(15, len(metric_values) - 2 if len(metric_values) % 2 == 0 else len(metric_values) - 1)
                    if window_length > 2:  # Ensure window length is valid
                        smoothed = savgol_filter(metric_values, window_length, 2)
                        ax.plot(timestamps_array, smoothed, 'r-', alpha=0.7, linewidth=2)
                except ImportError:
                    pass  # Skip smoothing if scipy not available
                except Exception as e:
                    self.logger.warning(f"Error applying smoothing: {str(e)}")
                
                # Format x-axis for datetime
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                
                # Set labels and add grid
                metric_label = metric_name.upper() if metric_name.lower() in ['rmse', 'mae'] else metric_name
                ax.set_ylabel(metric_label)
                ax.set_title(f"{metric_label} over Time")
                ax.grid(True, alpha=0.3)
                
                # Add average line
                avg = np.mean(metric_values)
                ax.axhline(y=avg, color='gray', linestyle='--', alpha=0.7)
                ax.text(timestamps_array[0], avg, f" Average: {avg:.3f}", va='bottom')
            
            # Format x-axis labels for the bottom subplot
            plt.xticks(rotation=45)
            plt.xlabel('Time')
            
            # Set overall title
            fig.suptitle(title, fontsize=14, y=1.02)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if filename provided
            if filename and self.output_dir:
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved temporal performance plot to {filepath}")
            
            if not show:
                plt.close(fig)
            
            return fig
    
    def plot_spatial_performance(self, location_df: pd.DataFrame,
                               metric_name: str = 'rmse',
                               lat_col: str = 'latitude',
                               lon_col: str = 'longitude',
                               location_id_col: str = 'location_id',
                               title: str = "Spatial Performance Distribution",
                               filename: Optional[str] = None,
                               show: bool = True) -> plt.Figure:
        """Plot performance metrics across geographical locations.
        
        Args:
            location_df: DataFrame with locations and performance metrics
            metric_name: Name of the metric column to visualize
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            location_id_col: Name of location ID column
            title: Plot title
            filename: Optional filename to save the plot
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        # Check if required columns exist
        required_cols = [lat_col, lon_col, metric_name]
        missing_cols = [col for col in required_cols if col not in location_df.columns]
        
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            return None
        
        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Create scatter plot with colored points by metric value
            scatter = ax.scatter(location_df[lon_col], location_df[lat_col],
                               c=location_df[metric_name], cmap='viridis',
                               s=100, alpha=0.7, edgecolors='white', linewidth=0.5)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(metric_name.upper() if metric_name.lower() in ['rmse', 'mae'] else metric_name)
            
            # Add location labels if not too many points
            if len(location_df) <= 30 and location_id_col in location_df.columns:
                for i, row in location_df.iterrows():
                    ax.annotate(str(row[location_id_col]),
                              (row[lon_col], row[lat_col]),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
            
            # Set labels and title
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title(title)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Make aspect ratio equal
            ax.set_aspect('equal', adjustable='datalim')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if filename provided
            if filename and self.output_dir:
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved spatial performance plot to {filepath}")
            
            if not show:
                plt.close(fig)
            
            return fig
    
    def plot_heatmap(self, data: pd.DataFrame, 
                   x_col: str, y_col: str, value_col: str,
                   title: str = "Heatmap",
                   cmap: str = "viridis",
                   filename: Optional[str] = None,
                   show: bool = True) -> plt.Figure:
        """Create a heatmap from data.
        
        Args:
            data: DataFrame with data to visualize
            x_col: Column to use for x-axis
            y_col: Column to use for y-axis
            value_col: Column with values for heatmap colors
            title: Plot title
            cmap: Colormap name
            filename: Optional filename to save the plot
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        # Check if required columns exist
        required_cols = [x_col, y_col, value_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            return None
        
        with plt.style.context(self.style):
            # Pivot data to create heatmap
            pivot_data = data.pivot_table(index=y_col, columns=x_col, values=value_col, aggfunc='mean')
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Create heatmap
            heatmap = sns.heatmap(pivot_data, annot=True, fmt=".2f", 
                                cmap=cmap, linewidths=.5, ax=ax,
                                cbar_kws={'label': value_col})
            
            # Set title
            ax.set_title(title)
            
            # Rotate x-labels if needed
            plt.xticks(rotation=45, ha='right')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if filename provided
            if filename and self.output_dir:
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved heatmap to {filepath}")
            
            if not show:
                plt.close(fig)
            
            return fig
            
    def plot_weekly_hourly_forecast(self, 
                               data: pd.DataFrame,
                               time_col: str = 'hour_start',
                               value_col: str = 'predicted_demand',
                               location_col: Optional[str] = None,
                               title: str = "Weekly Hourly Demand Forecast",
                               cmap: str = "viridis",
                               filename: Optional[str] = "weekly_hourly_forecast.png",
                               show: bool = True) -> plt.Figure:
        """Create a heatmap showing weekly patterns on an hourly scale.
        
        Args:
            data: DataFrame with time series data
            time_col: Column containing datetime information
            value_col: Column with values to visualize (e.g., demand predictions)
            location_col: Optional column to filter by location
            title: Plot title
            cmap: Colormap name
            filename: Optional filename to save the plot
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        # Check if required columns exist
        if time_col not in data.columns or value_col not in data.columns:
            self.logger.error(f"Missing required columns: {time_col} or {value_col}")
            return None
        
        # Ensure time column is datetime type
        data_copy = data.copy()
        if not pd.api.types.is_datetime64_any_dtype(data_copy[time_col]):
            try:
                data_copy[time_col] = pd.to_datetime(data_copy[time_col])
            except Exception as e:
                self.logger.error(f"Error converting {time_col} to datetime: {str(e)}")
                return None
        
        # Extract day of week and hour of day
        data_copy['day_of_week'] = data_copy[time_col].dt.dayofweek
        data_copy['hour_of_day'] = data_copy[time_col].dt.hour
        
        # Map day of week to names for better readability
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        data_copy['day_name'] = data_copy['day_of_week'].map(lambda x: day_names[x])
        
        # Filter by location if specified
        if location_col is not None and location_col in data_copy.columns:
            # Get top locations by volume for better visualization
            top_locations = data_copy.groupby(location_col)[value_col].sum().nlargest(5).index.tolist()
            location_name = top_locations[0]  # Use the busiest location by default
            data_copy = data_copy[data_copy[location_col] == location_name]
            title = f"{title} - Location {location_name}"
        
        with plt.style.context(self.style):
            # Create figure with subplots: main heatmap and two line plots for aggregates
            fig = plt.figure(figsize=(14, 10))
            gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3])
            
            # Main heatmap (hour x day)
            ax_heatmap = plt.subplot(gs[1, 0])
            
            # Aggregate by hour (right)
            ax_hour = plt.subplot(gs[1, 1])
            
            # Aggregate by day (top)
            ax_day = plt.subplot(gs[0, 0])
            
            # Calculate aggregated values
            hourly_avg = data_copy.groupby('hour_of_day')[value_col].mean()
            daily_avg = data_copy.groupby('day_name')[value_col].mean()
            
            # Sort days in correct order
            daily_avg = daily_avg.reindex(day_names)
            
            # Create pivot table for heatmap
            pivot_data = data_copy.pivot_table(
                index='day_name', 
                columns='hour_of_day', 
                values=value_col, 
                aggfunc='mean'
            )
            
            # Reindex to ensure correct day order
            pivot_data = pivot_data.reindex(day_names)
            
            # Plot heatmap
            sns.heatmap(
                pivot_data, 
                annot=True, 
                fmt=".1f", 
                cmap=cmap, 
                linewidths=.5, 
                ax=ax_heatmap,
                cbar_kws={'label': 'Predicted Demand'}
            )
            
            # Configure heatmap
            ax_heatmap.set_xlabel('Hour of Day')
            ax_heatmap.set_ylabel('Day of Week')
            ax_heatmap.set_title('Hourly Demand by Day of Week')
            
            # Plot hourly aggregate
            ax_hour.plot(hourly_avg.values, hourly_avg.index, '-o', color='darkblue')
            ax_hour.set_ylabel('Hour of Day')
            ax_hour.set_xlabel('Avg. Demand')
            ax_hour.set_title('Hourly Average')
            ax_hour.grid(True, alpha=0.3)
            
            # Plot daily aggregate
            ax_day.plot(daily_avg.index, daily_avg.values, '-o', color='darkred')
            ax_day.set_xlabel('Day of Week')
            ax_day.set_ylabel('Avg. Demand')
            ax_day.set_title('Daily Average')
            ax_day.grid(True, alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            
            # Overall title
            plt.suptitle(title, fontsize=16, y=0.98)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Save figure if filename provided
            if filename and self.output_dir:
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved weekly hourly forecast plot to {filepath}")
            
            if not show:
                plt.close(fig)
            
            return fig
