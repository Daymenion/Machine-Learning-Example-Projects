from ..utils.logger import StructuredLogger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
import os
from datetime import datetime
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

class PriceEvaluator:
    """Class for evaluating surge pricing strategies.
    
    This class provides methods to assess the effectiveness of
    surge pricing strategies by evaluating revenue impact,
    supply-demand equilibrium, and other business metrics.
    
    Attributes:
        logger: Logger instance for logging evaluation information
        output_dir: Directory to save evaluation results
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the PriceEvaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.logger = StructuredLogger("price_evaluator")
        self.output_dir = output_dir
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.logger.info("PriceEvaluator initialized")
    
    def evaluate_pricing_strategy(self, pricing_results: pd.DataFrame,
                                historical_data: Optional[pd.DataFrame] = None,
                                baseline_multiplier: float = 1.0) -> Dict[str, Any]:
        """Evaluate a pricing strategy against historical data or baseline.
        
        Args:
            pricing_results: DataFrame with pricing strategy results
            historical_data: Optional historical data for comparison
            baseline_multiplier: Baseline multiplier for comparison
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info("Evaluating pricing strategy")
        
        # Validate input
        required_columns = ['hour_start', 'start_location_id', 'surge_multiplier', 
                         'adjusted_demand', 'adjusted_supply', 'expected_revenue']
        
        missing_columns = [col for col in required_columns if col not in pricing_results.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns in pricing_results: {missing_columns}")
            return {"error": f"Missing required columns: {missing_columns}"}
        
        # Calculate baseline metrics (assuming a fixed multiplier)
        baseline_metrics = self._calculate_baseline_metrics(pricing_results, baseline_multiplier)
        
        # Calculate metrics for the pricing strategy
        strategy_metrics = {
            "mean_multiplier": pricing_results['surge_multiplier'].mean(),
            "median_multiplier": pricing_results['surge_multiplier'].median(),
            "max_multiplier": pricing_results['surge_multiplier'].max(),
            "min_multiplier": pricing_results['surge_multiplier'].min(),
            "std_multiplier": pricing_results['surge_multiplier'].std(),
            "total_expected_revenue": pricing_results['expected_revenue'].sum(),
            "mean_utilization_rate": pricing_results['utilization_rate'].mean() if 'utilization_rate' in pricing_results.columns else None,
            "mean_expected_wait_time": pricing_results['expected_wait_time'].mean() if 'expected_wait_time' in pricing_results.columns else None,
        }
        
        # Calculate supply-demand equilibrium metrics
        equilibrium_metrics = self._calculate_equilibrium_metrics(pricing_results)
        
        # Calculate location-specific metrics
        location_metrics = self._calculate_location_metrics(pricing_results)
        
        # Calculate time-specific metrics
        time_metrics = self._calculate_time_metrics(pricing_results)
        
        # Compare with historical data if provided
        historical_comparison = None
        if historical_data is not None:
            historical_comparison = self._compare_with_historical(pricing_results, historical_data)
        
        # Combine all metrics
        all_metrics = {
            "strategy_metrics": strategy_metrics,
            "baseline_comparison": baseline_metrics,
            "equilibrium_metrics": equilibrium_metrics,
            "location_metrics": location_metrics,
            "time_metrics": time_metrics,
        }
        
        if historical_comparison:
            all_metrics["historical_comparison"] = historical_comparison
        
        return all_metrics
    
    def _calculate_baseline_metrics(self, pricing_results: pd.DataFrame, 
                                  baseline_multiplier: float) -> Dict[str, Any]:
        """Calculate metrics for a baseline pricing strategy.
        
        Args:
            pricing_results: DataFrame with pricing strategy results
            baseline_multiplier: Baseline multiplier to compare against
            
        Returns:
            Dictionary of baseline comparison metrics
        """
        # Create a copy to avoid modifying the input
        df = pricing_results.copy()
        
        # Calculate baseline demand and supply
        # Assuming price elasticity of -0.5 (typical for ride-hailing)
        elasticity = -0.5
        
        # Adjust demand based on elasticity formula: Q2 = Q1 * (P2/P1)^elasticity
        # where P2/P1 is the ratio of baseline_multiplier to surge_multiplier
        df['baseline_demand'] = df.apply(
            lambda row: row['adjusted_demand'] * (baseline_multiplier / row['surge_multiplier'])**elasticity 
            if row['surge_multiplier'] > 0 else row['adjusted_demand'],
            axis=1
        )
        
        # Adjust supply based on a simple model
        # Assuming a linear response: 50% increase for each 1.0 surge increase
        df['baseline_supply'] = df.apply(
            lambda row: row['adjusted_supply'] * (1 + 0.5 * (baseline_multiplier - 1)) / 
                      (1 + 0.5 * (row['surge_multiplier'] - 1)) 
            if row['surge_multiplier'] > 1 else row['adjusted_supply'],
            axis=1
        )
        
        # Calculate baseline revenue
        base_price = 10.0  # Same as used in surge_optimizer.py
        df['baseline_revenue'] = df.apply(
            lambda row: min(row['baseline_demand'], row['baseline_supply']) * base_price * baseline_multiplier,
            axis=1
        )
        
        # Calculate baseline utilization
        df['baseline_utilization'] = df.apply(
            lambda row: min(1.0, row['baseline_demand'] / max(1, row['baseline_supply'])),
            axis=1
        )
        
        # Calculate metrics
        baseline_metrics = {
            "baseline_multiplier": baseline_multiplier,
            "baseline_total_revenue": df['baseline_revenue'].sum(),
            "baseline_mean_utilization": df['baseline_utilization'].mean(),
            "revenue_improvement": (df['expected_revenue'].sum() / df['baseline_revenue'].sum() - 1) * 100,
            "utilization_improvement": (df['utilization_rate'].mean() / df['baseline_utilization'].mean() - 1) * 100 
                if 'utilization_rate' in df.columns else None,
            "demand_change": (df['adjusted_demand'].sum() / df['baseline_demand'].sum() - 1) * 100,
            "supply_change": (df['adjusted_supply'].sum() / df['baseline_supply'].sum() - 1) * 100,
        }
        
        return baseline_metrics
    
    def _calculate_equilibrium_metrics(self, pricing_results: pd.DataFrame) -> Dict[str, Any]:
        """Calculate supply-demand equilibrium metrics.
        
        Args:
            pricing_results: DataFrame with pricing strategy results
            
        Returns:
            Dictionary of equilibrium metrics
        """
        # Create a copy to avoid modifying the input
        df = pricing_results.copy()
        
        # Calculate supply-demand mismatch
        df['supply_demand_diff'] = df['adjusted_supply'] - df['adjusted_demand']
        df['abs_supply_demand_diff'] = np.abs(df['supply_demand_diff'])
        df['supply_demand_ratio'] = df['adjusted_supply'] / df['adjusted_demand'].clip(lower=1)
        
        # Calculate metrics
        equilibrium_metrics = {
            "mean_abs_supply_demand_diff": df['abs_supply_demand_diff'].mean(),
            "median_abs_supply_demand_diff": df['abs_supply_demand_diff'].median(),
            "mean_supply_demand_ratio": df['supply_demand_ratio'].mean(),
            "pct_oversupply": (df['supply_demand_diff'] > 0).mean() * 100,
            "pct_undersupply": (df['supply_demand_diff'] < 0).mean() * 100,
            "pct_equilibrium": (np.abs(df['supply_demand_diff']) <= 2).mean() * 100,  # Within 2 units
        }
        
        return equilibrium_metrics
    
    def _calculate_location_metrics(self, pricing_results: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Calculate metrics for each location.
        
        Args:
            pricing_results: DataFrame with pricing strategy results
            
        Returns:
            Dictionary of location metrics
        """
        location_metrics = {}
        
        # Group by location
        for location_id, group in pricing_results.groupby('start_location_id'):
            location_metrics[str(location_id)] = {
                "mean_multiplier": group['surge_multiplier'].mean(),
                "max_multiplier": group['surge_multiplier'].max(),
                "total_revenue": group['expected_revenue'].sum(),
                "mean_utilization": group['utilization_rate'].mean() if 'utilization_rate' in group.columns else None,
                "mean_wait_time": group['expected_wait_time'].mean() if 'expected_wait_time' in group.columns else None,
                "demand_supply_balance": (group['adjusted_supply'] / group['adjusted_demand'].clip(lower=1)).mean(),
            }
        
        return location_metrics
    
    def _calculate_time_metrics(self, pricing_results: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Calculate metrics for different time periods.
        
        Args:
            pricing_results: DataFrame with pricing strategy results
            
        Returns:
            Dictionary of time metrics
        """
        time_metrics = {}
        
        # Extract hour of day
        df = pricing_results.copy()
        if 'hour' not in df.columns and 'hour_start' in df.columns:
            df['hour'] = pd.to_datetime(df['hour_start']).dt.hour
        
        # Group by hour
        hourly_metrics = {}
        for hour, group in df.groupby('hour'):
            hourly_metrics[str(hour)] = {
                "mean_multiplier": group['surge_multiplier'].mean(),
                "total_revenue": group['expected_revenue'].sum(),
                "mean_utilization": group['utilization_rate'].mean() if 'utilization_rate' in group.columns else None,
                "demand_supply_balance": (group['adjusted_supply'] / group['adjusted_demand'].clip(lower=1)).mean(),
            }
        
        # Categorize hours
        def categorize_hour(hour):
            if 7 <= hour <= 10:  # Morning rush
                return "morning_rush"
            elif 11 <= hour <= 15:  # Midday
                return "midday"
            elif 16 <= hour <= 19:  # Evening rush
                return "evening_rush"
            elif 20 <= hour <= 23:  # Evening
                return "evening"
            else:  # Night (0-6)
                return "night"
        
        df['time_category'] = df['hour'].apply(categorize_hour)
        
        # Group by time category
        period_metrics = {}
        for period, group in df.groupby('time_category'):
            period_metrics[period] = {
                "mean_multiplier": group['surge_multiplier'].mean(),
                "total_revenue": group['expected_revenue'].sum(),
                "mean_utilization": group['utilization_rate'].mean() if 'utilization_rate' in group.columns else None,
                "demand_supply_balance": (group['adjusted_supply'] / group['adjusted_demand'].clip(lower=1)).mean(),
            }
        
        time_metrics = {
            "hourly": hourly_metrics,
            "period": period_metrics,
        }
        
        return time_metrics
    
    def _compare_with_historical(self, pricing_results: pd.DataFrame, 
                               historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Compare pricing strategy with historical data.
        
        Args:
            pricing_results: DataFrame with pricing strategy results
            historical_data: Historical data for comparison
            
        Returns:
            Dictionary of comparison metrics
        """
        # Validate historical data
        required_columns = ['hour_start', 'start_location_id', 'price_multiplier', 'demand']
        missing_columns = [col for col in required_columns if col not in historical_data.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns in historical_data: {missing_columns}")
            return {"error": f"Missing required columns: {missing_columns}"}
        
        # Create a copy of both dataframes
        strategy_df = pricing_results.copy()
        historical_df = historical_data.copy()
        
        # Prepare for merging
        strategy_df = strategy_df[['hour_start', 'start_location_id', 'surge_multiplier', 'adjusted_demand', 'expected_revenue']]
        strategy_df.rename(columns={'surge_multiplier': 'strategy_multiplier', 
                                  'adjusted_demand': 'strategy_demand',
                                  'expected_revenue': 'strategy_revenue'}, inplace=True)
        
        historical_df = historical_df[['hour_start', 'start_location_id', 'price_multiplier', 'demand']]
        historical_df.rename(columns={'price_multiplier': 'historical_multiplier',
                                    'demand': 'historical_demand'}, inplace=True)
        
        # Calculate historical revenue (using same base price assumption)
        base_price = 10.0
        historical_df['historical_revenue'] = historical_df['historical_demand'] * base_price * historical_df['historical_multiplier']
        
        # Merge on location and time
        merged_df = pd.merge(strategy_df, historical_df, on=['hour_start', 'start_location_id'], how='inner')
        
        # Calculate comparison metrics
        if len(merged_df) == 0:
            self.logger.warning("No matching data points between strategy and historical data")
            return {"error": "No matching data points for comparison"}
        
        # Calculate differences
        merged_df['multiplier_diff'] = merged_df['strategy_multiplier'] - merged_df['historical_multiplier']
        merged_df['demand_diff'] = merged_df['strategy_demand'] - merged_df['historical_demand']
        merged_df['revenue_diff'] = merged_df['strategy_revenue'] - merged_df['historical_revenue']
        merged_df['multiplier_pct_change'] = (merged_df['strategy_multiplier'] / merged_df['historical_multiplier'] - 1) * 100
        merged_df['demand_pct_change'] = (merged_df['strategy_demand'] / merged_df['historical_demand'] - 1) * 100
        merged_df['revenue_pct_change'] = (merged_df['strategy_revenue'] / merged_df['historical_revenue'] - 1) * 100
        
        comparison_metrics = {
            "n_comparison_points": len(merged_df),
            "mean_multiplier_diff": merged_df['multiplier_diff'].mean(),
            "mean_demand_diff": merged_df['demand_diff'].mean(),
            "mean_revenue_diff": merged_df['revenue_diff'].mean(),
            "total_revenue_improvement": (merged_df['strategy_revenue'].sum() / merged_df['historical_revenue'].sum() - 1) * 100,
            "multiplier_rmse": np.sqrt(mean_squared_error(merged_df['historical_multiplier'], merged_df['strategy_multiplier'])),
            "pct_higher_multiplier": (merged_df['strategy_multiplier'] > merged_df['historical_multiplier']).mean() * 100,
            "pct_lower_multiplier": (merged_df['strategy_multiplier'] < merged_df['historical_multiplier']).mean() * 100,
            "pct_higher_revenue": (merged_df['strategy_revenue'] > merged_df['historical_revenue']).mean() * 100,
        }
        
        return comparison_metrics
    
    def evaluate_pricing_fairness(self, pricing_results: pd.DataFrame,
                               location_metadata: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Evaluate fairness of pricing strategy across different locations.
        
        Args:
            pricing_results: DataFrame with pricing strategy results
            location_metadata: Optional DataFrame with location metadata (demographics, income, etc.)
            
        Returns:
            Dictionary of fairness metrics
        """
        # Create a copy to avoid modifying the input
        df = pricing_results.copy()
        
        # Calculate distribution of surge multipliers
        multiplier_distribution = {
            "mean": df['surge_multiplier'].mean(),
            "median": df['surge_multiplier'].median(),
            "std": df['surge_multiplier'].std(),
            "min": df['surge_multiplier'].min(),
            "max": df['surge_multiplier'].max(),
            "pct_no_surge": (df['surge_multiplier'] == 1.0).mean() * 100,
            "pct_high_surge": (df['surge_multiplier'] >= 2.0).mean() * 100,
        }
        
        # Calculate location-level multiplier statistics
        location_stats = df.groupby('start_location_id')['surge_multiplier'].agg(['mean', 'std', 'min', 'max']).reset_index()
        
        # Calculate fairness metrics
        fairness_metrics = {
            "multiplier_distribution": multiplier_distribution,
            "location_variation": float(location_stats['mean'].std()),  # Variation across locations
            "location_max_diff": float(location_stats['mean'].max() - location_stats['mean'].min()),  # Max difference
            "gini_coefficient": self._calculate_gini(location_stats['mean']),  # Inequality measure
        }
        
        # If location metadata is provided, calculate correlation with demographics
        if location_metadata is not None:
            demographic_metrics = self._calculate_demographic_fairness(df, location_metadata)
            fairness_metrics["demographic_metrics"] = demographic_metrics
        
        return fairness_metrics
    
    def _calculate_gini(self, array: np.ndarray) -> float:
        """Calculate Gini coefficient to measure inequality.
        
        Args:
            array: Array of values
            
        Returns:
            Gini coefficient (0 = perfect equality, 1 = perfect inequality)
        """
        array = np.array(array).flatten()
        if np.amin(array) < 0:
            array -= np.amin(array)  # Values must be non-negative
        
        # All values are equal
        if np.all(array == array[0]):
            return 0.0
        
        # Sort array
        array = np.sort(array)
        index = np.arange(1, array.shape[0] + 1)
        n = array.shape[0]
        
        # Calculate Gini coefficient
        return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))
    
    def _calculate_demographic_fairness(self, pricing_results: pd.DataFrame,
                                       location_metadata: pd.DataFrame) -> Dict[str, Any]:
        """Calculate fairness metrics based on demographic information.
        
        Args:
            pricing_results: DataFrame with pricing strategy results
            location_metadata: DataFrame with location demographic information
            
        Returns:
            Dictionary of demographic fairness metrics
        """
        # Merge pricing results with location metadata
        df = pricing_results.copy()
        location_avg = df.groupby('start_location_id')['surge_multiplier'].mean().reset_index()
        
        # Check if required columns exist in location_metadata
        demographic_cols = [col for col in location_metadata.columns 
                          if col != 'start_location_id' and 
                          location_metadata[col].dtype in [np.float64, np.int64]]
        
        if not demographic_cols:
            return {"error": "No numeric demographic columns found in location_metadata"}
        
        # Merge with location average multipliers
        merged = pd.merge(location_avg, location_metadata, on='start_location_id', how='inner')
        
        if len(merged) == 0:
            return {"error": "No matching locations between pricing results and metadata"}
        
        # Calculate correlations between multipliers and demographics
        correlations = {}
        for col in demographic_cols:
            correlations[col] = float(np.corrcoef(merged['surge_multiplier'], merged[col])[0, 1])
        
        # Identify strongest correlations (positive and negative)
        sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        fairness_metrics = {
            "demographic_correlations": correlations,
            "strongest_correlations": sorted_correlations[:3] if len(sorted_correlations) >= 3 else sorted_correlations,
        }
        
        return fairness_metrics
    
    def save_evaluation_results(self, results: Dict[str, Any], filename: Optional[str] = None):
        """Save evaluation results to a JSON file.
        
        Args:
            results: Dictionary of evaluation results
            filename: Optional filename to save results to
        """
        if not self.output_dir:
            self.logger.warning("No output directory specified for saving results")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pricing_evaluation_{timestamp}.json"
        
        output_path = os.path.join(self.output_dir, filename)
        
        # Convert numpy values to native Python types for JSON serialization
        # This is a recursive function to handle nested dictionaries
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return convert_to_serializable(obj.tolist())
            elif isinstance(obj, pd.DataFrame):
                return convert_to_serializable(obj.to_dict(orient='records'))
            elif isinstance(obj, pd.Series):
                return convert_to_serializable(obj.tolist())
            elif obj is np.nan:
                return None
            else:
                return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        self.logger.info(f"Evaluation results saved to {output_path}")
