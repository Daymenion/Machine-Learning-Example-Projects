import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple, Any
import time
from datetime import datetime, timedelta
import scipy.optimize as optimize
from scipy.stats import norm
import joblib
import os

from ..utils.logger import StructuredLogger


class SurgePricingOptimizer:
    """Advanced surge pricing optimizer for ride demand management.
    
    This class implements sophisticated strategies for determining optimal
    surge pricing multipliers based on predicted demand and supply patterns.
    It balances revenue optimization with market equilibrium and user satisfaction.
    
    Key features:
    - Supply-demand equilibrium modeling
    - Price elasticity estimation
    - Dynamic temporal and spatial pricing strategies
    - Revenue simulation and optimization
    - What-if scenario analysis
    - Smoothing algorithms to prevent price oscillation
    """
    
    def __init__(self, config: Dict = None, logger: Optional[StructuredLogger] = None):
        """Initialize the surge pricing optimizer.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config or {}
        self.logger = logger or StructuredLogger("surge_optimizer")
        
        # Set default parameters if not specified
        self.price_elasticity = self.config.get("price_elasticity", -0.6)  # Default elasticity of demand
        self.max_surge_multiplier = self.config.get("max_surge_multiplier", 3.0)  # Maximum allowed surge multiplier
        self.min_surge_multiplier = self.config.get("min_surge_multiplier", 1.0)  # Minimum allowed surge multiplier
        self.surge_step = self.config.get("surge_step", 0.05)  # Minimum step for surge pricing
        self.smoothing_factor = self.config.get("smoothing_factor", 0.3)  # For exponential smoothing of price changes
        self.spatial_smoothing = self.config.get("spatial_smoothing", True)  # Whether to apply spatial smoothing
        self.target_utilization = self.config.get("target_utilization", 0.85)  # Target driver utilization rate
        
        # User satisfaction parameters
        self.max_acceptable_wait_time = self.config.get("max_acceptable_wait_time", 15)  # in minutes
        self.wait_time_sensitivity = self.config.get("wait_time_sensitivity", 0.1)  # Weight for wait time in optimization
        
        # Learned parameters (to be estimated from data)
        self.historical_multipliers = {}
        self.location_elasticities = {}
        self.time_elasticities = {}
        self.supply_response_curves = {}
        
        # Status tracking
        self.is_trained = False
    
    def fit(self, historical_data: pd.DataFrame) -> None:
        """Train the optimizer on historical demand, supply, and pricing data.
        
        Args:
            historical_data: DataFrame with historical demand, supply, pricing, and outcomes
        """
        self.logger.info("Training surge pricing optimizer on historical data")
        
        required_columns = ["hour_start", "start_location_id", "demand", "supply", "price_multiplier", "wait_time"]
        missing_columns = [col for col in required_columns if col not in historical_data.columns]
        
        if missing_columns:
            self.logger.warning(f"Missing required columns: {missing_columns}")
            self.logger.info("Using default elasticity and supply response parameters")
            return
        
        self.location_elasticities = self._estimate_location_elasticities(historical_data)
        
        self.time_elasticities = self._estimate_time_elasticities(historical_data)
        
        self.supply_response_curves = self._estimate_supply_response(historical_data)
        
        self.historical_multipliers = self._aggregate_historical_multipliers(historical_data)
        
        self.is_trained = True
        self.logger.info("Surge pricing optimizer training completed")
    
    def _estimate_location_elasticities(self, data: pd.DataFrame) -> Dict[Any, float]:
        """Estimate price elasticity for each location.
        
        Args:
            data: Historical data with pricing and demand
            
        Returns:
            Dictionary mapping location IDs to elasticity values
        """
        location_elasticities = {}
        
        for location_id, group in data.groupby("start_location_id"):
            if len(group) < 10:  # Skip locations with too few data points
                location_elasticities[location_id] = self.price_elasticity  # Use default
                continue
                
            try:
                group = group.copy()
                group["log_demand"] = np.log(group["demand"] + 1)  # Add 1 to handle zeros
                group["log_price"] = np.log(group["price_multiplier"])
                
                from sklearn.linear_model import LinearRegression
                X = group[["log_price"]]
                y = group["log_demand"]
                model = LinearRegression().fit(X, y)
                
                elasticity = model.coef_[0]  # The slope coefficient is our elasticity
                
                # Validate the elasticity is reasonable
                if elasticity < -2.0 or elasticity > 0.0:  # Most elasticities should be negative but not extreme
                    elasticity = self.price_elasticity  # Use default if estimate is unreasonable
                    
                location_elasticities[location_id] = elasticity
                
            except Exception as e:
                self.logger.warning(f"Error estimating elasticity for location {location_id}: {str(e)}")
                location_elasticities[location_id] = self.price_elasticity  # Use default
        
        return location_elasticities
    
    def _estimate_time_elasticities(self, data: pd.DataFrame) -> Dict[int, float]:
        """Estimate price elasticity for different hours of the day.
        
        Args:
            data: Historical data with pricing and demand
            
        Returns:
            Dictionary mapping hours to elasticity values
        """
        time_elasticities = {}
        
        # Extract hour of day
        data = data.copy()
        data["hour"] = pd.to_datetime(data["hour_start"]).dt.hour
        
        # Group by hour
        for hour, group in data.groupby("hour"):
            if len(group) < 20:  # Skip hours with too few data points
                time_elasticities[hour] = self.price_elasticity  # Use default
                continue
                
            try:
                group["log_demand"] = np.log(group["demand"] + 1)  # Add 1 to handle zeros
                group["log_price"] = np.log(group["price_multiplier"])
                
                from sklearn.linear_model import LinearRegression
                X = group[["log_price"]]
                y = group["log_demand"]
                model = LinearRegression().fit(X, y)
                
                elasticity = model.coef_[0]  # The slope coefficient is our elasticity
                
                if elasticity < -2.0 or elasticity > 0.0:  # Most elasticities should be negative but not extreme
                    elasticity = self.price_elasticity  # Use default if estimate is unreasonable
                    
                time_elasticities[hour] = elasticity
                
            except Exception as e:
                self.logger.warning(f"Error estimating elasticity for hour {hour}: {str(e)}")
                time_elasticities[hour] = self.price_elasticity  # Use default
        
        return time_elasticities
    
    def _estimate_supply_response(self, data: pd.DataFrame) -> Dict[Any, Dict[str, Any]]:
        """Estimate how supply (driver availability) responds to surge pricing.
        
        Args:
            data: Historical data with pricing and supply
            
        Returns:
            Dictionary mapping location IDs to supply response parameters
        """
        supply_response = {}
        
        for location_id, group in data.groupby("start_location_id"):
            if len(group) < 10:  # Skip locations with too few data points
                supply_response[location_id] = {
                    "model": "linear",
                    "base_supply": 10,  # Default base supply
                    "coefficient": 5,    # Default coefficient (drivers added per 1.0 surge increase)
                }
                continue
                
            try:
                from sklearn.linear_model import LinearRegression
                X = group[["price_multiplier"]]
                y = group["supply"]
                model = LinearRegression().fit(X, y)
                
                base_supply = model.intercept_ - model.coef_[0]
                coefficient = model.coef_[0]
                
                if base_supply <= 0 or coefficient <= 0:
                    base_supply = 10
                    coefficient = 5
                
                supply_response[location_id] = {
                    "model": "linear",
                    "base_supply": base_supply,
                    "coefficient": coefficient,
                }
                
            except Exception as e:
                self.logger.warning(f"Error estimating supply response for location {location_id}: {str(e)}")
                # Use default values
                supply_response[location_id] = {
                    "model": "linear",
                    "base_supply": 10,
                    "coefficient": 5,
                }
        
        return supply_response
    
    def _aggregate_historical_multipliers(self, data: pd.DataFrame) -> Dict[Tuple[Any, int], float]:
        """Aggregate historical price multipliers by location and hour.
        
        Args:
            data: Historical data with pricing information
            
        Returns:
            Dictionary mapping (location_id, hour) to average multiplier
        """
        historical_multipliers = {}
        
        # Extract hour of day
        data = data.copy()
        data["hour"] = pd.to_datetime(data["hour_start"]).dt.hour
        
        # Group by location and hour
        for (location_id, hour), group in data.groupby(["start_location_id", "hour"]):
            # Calculate average multiplier
            avg_multiplier = group["price_multiplier"].mean()
            historical_multipliers[(location_id, hour)] = avg_multiplier
        
        return historical_multipliers
    
    def calculate_surge_multipliers(self, demand_predictions: pd.DataFrame, 
                                 supply_estimates: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Calculate optimal surge multipliers based on predicted demand.
        
        Args:
            demand_predictions: DataFrame with predicted demand per location and time
            supply_estimates: Optional DataFrame with estimated supply per location and time
            
        Returns:
            DataFrame with optimal surge multipliers
        """
        self.logger.info("Calculating optimal surge multipliers based on predicted demand")
        
        # Ensure required columns are present
        required_columns = ["hour_start", "start_location_id", "predicted_demand"]
        missing_columns = [col for col in required_columns if col not in demand_predictions.columns]
        
        if missing_columns:
            self.logger.error(f"Missing required columns in demand predictions: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        result_df = demand_predictions.copy()
        
        result_df["hour"] = pd.to_datetime(result_df["hour_start"]).dt.hour
        
        if supply_estimates is not None:
            result_df = pd.merge(
                result_df,
                supply_estimates[["hour_start", "start_location_id", "estimated_supply"]],
                on=["hour_start", "start_location_id"],
                how="left"
            )
            
            result_df["estimated_supply"].fillna(result_df["predicted_demand"] * 0.7, inplace=True)
        else:
            result_df["estimated_supply"] = result_df.apply(
                lambda row: self._estimate_base_supply(row["start_location_id"], row["hour"]), 
                axis=1
            )
        
        result_df["demand_supply_ratio"] = result_df["predicted_demand"] / result_df["estimated_supply"].clip(lower=1)
        
        result_df["optimal_multiplier"] = result_df.apply(
            lambda row: self._optimize_multiplier(
                location_id=row["start_location_id"],
                hour=row["hour"],
                predicted_demand=row["predicted_demand"],
                estimated_supply=row["estimated_supply"]
            ),
            axis=1
        )
        
        if self.spatial_smoothing and "latitude" in result_df.columns and "longitude" in result_df.columns:
            result_df = self._apply_spatial_smoothing(result_df)
        
        result_df = self._apply_temporal_smoothing(result_df)
        
        result_df["surge_multiplier"] = (
            np.round(result_df["optimal_multiplier"] / self.surge_step) * self.surge_step
        ).clip(self.min_surge_multiplier, self.max_surge_multiplier)
        
        result_df = self._calculate_expected_outcomes(result_df)
        
        self.logger.info(f"Surge multipliers calculated for {len(result_df)} location-time pairs")
        return result_df
    
    def _estimate_base_supply(self, location_id: Any, hour: int) -> float:
        """Estimate base supply for a given location and hour.
        
        Args:
            location_id: Location identifier
            hour: Hour of day (0-23)
            
        Returns:
            Estimated base supply
        """
        if location_id in self.supply_response_curves:
            return self.supply_response_curves[location_id]["base_supply"]
        
        if 7 <= hour <= 22:  # Daytime (7 AM to 10 PM)
            return 15 + 5 * np.sin(np.pi * (hour - 7) / 15)  # Peak at around 3 PM
        else:  # Nighttime
            return 8  # Lower base supply at night
    
    def _optimize_multiplier(self, location_id: Any, hour: int, 
                           predicted_demand: float, estimated_supply: float) -> float:
        """Optimize surge multiplier to maximize revenue while balancing other factors.
        
        Args:
            location_id: Location identifier
            hour: Hour of day (0-23)
            predicted_demand: Predicted demand for this location and time
            estimated_supply: Estimated supply for this location and time
            
        Returns:
            Optimal surge multiplier
        """
        location_elasticity = self.location_elasticities.get(location_id, self.price_elasticity)
        time_elasticity = self.time_elasticities.get(hour, self.price_elasticity)
        
        elasticity = 0.7 * location_elasticity + 0.3 * time_elasticity
        
        historical_multiplier = self.historical_multipliers.get((location_id, hour), 1.0)
        
        if predicted_demand <= estimated_supply * 0.8:
            return max(1.0, historical_multiplier * 0.9)
        
        ds_ratio = predicted_demand / estimated_supply
        
        if elasticity < -1.0:
            optimal_markup = 1.0
        else:
            optimal_markup = min(3.0, max(1.0, -1.0 / elasticity))
        
        surge_factor = min(3.0, max(1.0, ds_ratio))
        
        revenue_weight = min(1.0, ds_ratio - 0.8)
        
        weighted_surge = (
            revenue_weight * optimal_markup + 
            (1 - revenue_weight) * surge_factor
        )
        
        return 0.7 * weighted_surge + 0.3 * historical_multiplier
    
    def _apply_spatial_smoothing(self, result_df: pd.DataFrame) -> pd.DataFrame:
        """Apply spatial smoothing to surge multipliers.
        
        Args:
            result_df: DataFrame with calculated multipliers
            
        Returns:
            DataFrame with spatially smoothed multipliers
        """
        if "latitude" not in result_df.columns or "longitude" not in result_df.columns:
            return result_df
        
        smoothed_df = result_df.copy()
        
        for time, group in result_df.groupby("hour_start"):
            from sklearn.metrics.pairwise import haversine_distances
            
            coords = np.radians(group[["latitude", "longitude"]].values)
            
            dist_matrix = haversine_distances(coords) * 6371
            
            weights = np.exp(-dist_matrix**2 / (2 * 2**2))
            
            row_sums = weights.sum(axis=1, keepdims=True)
            weights = weights / row_sums
            
            original_multipliers = group["optimal_multiplier"].values
            smoothed_multipliers = weights @ original_multipliers
            
            # Update the original dataframe with smoothed values
            for i, idx in enumerate(group.index):
                smoothed_df.loc[idx, "optimal_multiplier"] = (
                    0.7 * original_multipliers[i] + 0.3 * smoothed_multipliers[i]
                )
        
        return smoothed_df
    
    def _apply_temporal_smoothing(self, result_df: pd.DataFrame) -> pd.DataFrame:
        """Apply temporal smoothing to surge multipliers.
        
        Args:
            result_df: DataFrame with calculated multipliers
            
        Returns:
            DataFrame with temporally smoothed multipliers
        """
        # Create a copy to avoid modifying the input
        smoothed_df = result_df.copy()
        
        for location, group in result_df.groupby("start_location_id"):
            group = group.sort_values("hour_start")
            
            alpha = self.smoothing_factor
            multipliers = group["optimal_multiplier"].values
            smoothed_multipliers = np.zeros_like(multipliers)
            
            # First value is unchanged
            smoothed_multipliers[0] = multipliers[0]
            
            # Apply smoothing formula: s_t = α * x_t + (1-α) * s_(t-1)
            for i in range(1, len(multipliers)):
                smoothed_multipliers[i] = alpha * multipliers[i] + (1 - alpha) * smoothed_multipliers[i-1]
            
            # Update the original dataframe with smoothed values
            for i, idx in enumerate(group.index):
                smoothed_df.loc[idx, "optimal_multiplier"] = smoothed_multipliers[i]
        
        return smoothed_df
    
    def _calculate_expected_outcomes(self, result_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate expected outcomes based on surge multipliers.
        
        Args:
            result_df: DataFrame with calculated multipliers
            
        Returns:
            DataFrame with expected outcomes
        """
        result_df = result_df.copy()
        
        result_df["adjusted_demand"] = result_df.apply(
            lambda row: self._calculate_adjusted_demand(
                location_id=row["start_location_id"],
                hour=row["hour"],
                base_demand=row["predicted_demand"],
                multiplier=row["surge_multiplier"]
            ),
            axis=1
        )
        
        # Calculate adjusted supply based on surge response
        result_df["adjusted_supply"] = result_df.apply(
            lambda row: self._calculate_adjusted_supply(
                location_id=row["start_location_id"],
                hour=row["hour"],
                base_supply=row["estimated_supply"],
                multiplier=row["surge_multiplier"]
            ),
            axis=1
        )
        
        base_price = 10.0  # Default base price
        result_df["expected_revenue"] = result_df.apply(
            lambda row: min(row["adjusted_demand"], row["adjusted_supply"]) * base_price * row["surge_multiplier"],
            axis=1
        )
        
        # Calculate utilization rate
        result_df["utilization_rate"] = result_df.apply(
            lambda row: min(1.0, row["adjusted_demand"] / max(1, row["adjusted_supply"])),
            axis=1
        )
        
        # Calculate expected wait time (simple model)
        result_df["expected_wait_time"] = result_df.apply(
            lambda row: self._calculate_expected_wait_time(
                row["adjusted_demand"], row["adjusted_supply"]
            ),
            axis=1
        )
        
        return result_df
    
    def _calculate_adjusted_demand(self, location_id: Any, hour: int, 
                                base_demand: float, multiplier: float) -> float:
        """Calculate demand adjusted for price elasticity.
        
        Args:
            location_id: Location identifier
            hour: Hour of day (0-23)
            base_demand: Base demand prediction
            multiplier: Surge multiplier
            
        Returns:
            Adjusted demand
        """
        # Get elasticity for this location and hour
        location_elasticity = self.location_elasticities.get(location_id, self.price_elasticity)
        time_elasticity = self.time_elasticities.get(hour, self.price_elasticity)
        
        # Use a weighted average of location and time elasticities
        elasticity = 0.7 * location_elasticity + 0.3 * time_elasticity
        
        # Calculate adjusted demand using elasticity formula: Q2 = Q1 * (P2/P1)^elasticity
        # where P2/P1 is our multiplier
        adjusted_demand = base_demand * (multiplier ** elasticity)
        
        return max(0, adjusted_demand)  # Ensure non-negative
    
    def _calculate_adjusted_supply(self, location_id: Any, hour: int, 
                                base_supply: float, multiplier: float) -> float:
        """Calculate supply adjusted for surge response.
        
        Args:
            location_id: Location identifier
            hour: Hour of day (0-23)
            base_supply: Base supply estimate
            multiplier: Surge multiplier
            
        Returns:
            Adjusted supply
        """
        if location_id in self.supply_response_curves:
            response = self.supply_response_curves[location_id]
            
            if response["model"] == "linear":
                adjusted_supply = base_supply + response["coefficient"] * (multiplier - 1)
            else:
                # Default to a simple model
                adjusted_supply = base_supply * (1 + 0.5 * (multiplier - 1))
        else:
            # Default supply response model (50% increase for each 1.0 surge increase)
            adjusted_supply = base_supply * (1 + 0.5 * (multiplier - 1))
        
        return max(base_supply, adjusted_supply)  # Ensure supply doesn't decrease
    
    def _calculate_expected_wait_time(self, demand: float, supply: float) -> float:
        """Calculate expected wait time based on demand and supply.
        
        Args:
            demand: Adjusted demand
            supply: Adjusted supply
            
        Returns:
            Expected wait time in minutes
        """
        if supply >= demand:
            utilization = demand / max(1, supply)
            wait_time = 2 + 6 * (utilization ** 2)
        else:
            excess_ratio = demand / max(1, supply)
            wait_time = 8 + 12 * (excess_ratio - 1)
            
        return min(30, wait_time)
