"""Advanced, vectorised
========================================================
Improvements over v1
--------------------
* **Missing features added** – driver_next_trip_location, user_next_trip_days,
  global lagged top locations and lagged average regain potential.
* **Config + logger integration** – supports `ConfigManager` and
  `StructuredLogger` via relative imports.
* **Single‑pass entity joins** – still zero redundant merges.
* **Fully typed & documented** – NumPy‑docstring style.
* **Reproducible** – single random‑seed entry.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import os
import warnings

# Project‑level utilities (resolved at runtime via package structure)
from ..utils.logger import StructuredLogger  # type: ignore
from ..utils.config_manager import ConfigManager  # type: ignore

# ---------------------------------------------------------------------------
# Global seed for deterministic behaviour
# ---------------------------------------------------------------------------
SEED: int = 42
np.random.seed(SEED)


def _safe_mode(series: pd.Series) -> np.ndarray:
    """Return first mode value or NaN."""
    return series.mode().iat[0] if not series.mode().empty else np.nan

# ---------------------------------------------------------------------------
# 1  Entity enrichment & frequency metrics
# ---------------------------------------------------------------------------


def compute_entity_stats(trips: pd.DataFrame) -> pd.DataFrame:
    """Add passenger/driver frequency & next‑trip features (vectorised)."""

    # ------------------------------------------------------
    # Passenger stats
    # ------------------------------------------------------
    p_grp = trips.sort_values("start_time").groupby("passenger_id", sort=False)
    p_last = p_grp["start_time"].transform("last")
    using_time = (p_last - p_grp["signup_date_passenger"].transform("first")).dt.total_seconds() / 3600.0
    total_trips = p_grp["trip_id"].transform("count")
    using_freq = np.log1p(total_trips / np.log1p(using_time)) * np.log10(1 + using_time / 2)

    # Next‑trip interval in days
    pnext_time = p_grp["start_time"].shift(-1)
    if pnext_time.isnull().any():
      pnext_time = pnext_time.fillna(trips["end_time"].max())
    trips["user_next_trip_days"]  = (pnext_time - trips["end_time"]).dt.total_seconds() / 86400.0

    trips["user_using_time"] = using_time
    trips["user_using_frequency"] = using_freq

    # ------------------------------------------------------
    # Driver stats
    # ------------------------------------------------------
    d_grp = trips.sort_values("start_time").groupby("driver_id", sort=False)
    d_last = d_grp["start_time"].transform("last")
    work_time = (d_last - d_grp["signup_date"].transform("first")).dt.total_seconds() / 3600.0
    trip_count = d_grp["trip_id"].transform("count")
    trip_freq = np.log1p(trip_count / np.log1p(work_time)) * np.log10(1 + work_time / 2)

    # Next‑trip location (start_location_id of following trip) if exists if not 0
    trips["driver_next_trip_location"] = d_grp["start_location_id"].shift(-1).fillna(0).astype("int32")
    dnext_time = d_grp["start_time"].shift(-1)
    if dnext_time.isnull().any():
        dnext_time = dnext_time.fillna(trips["end_time"].max())
    trips["driver_next_trip_days"] = (dnext_time - trips["end_time"]).dt.total_seconds() / 86400.0
    

    trips["driver_trip_count"] = trip_count.astype("int32")
    trips["driver_work_time"] = work_time
    trips["driver_trip_frequency"] = trip_freq

    trips["regain_potential"] = np.log1p(
        (trips["price"] / np.log10(trips["distance_km"] * trips["duration_min"]))
        * np.log1p(trips["user_using_frequency"] * trips["driver_trip_frequency"])
    )

    return trips

# ---------------------------------------------------------------------------
# 2  Hourly aggregation – include new mode & averages
# ---------------------------------------------------------------------------
def hourly_aggregates(trips: pd.DataFrame) -> pd.DataFrame:
    key = ["hour_start", "start_location_id"]
    trips["hour_start"] = trips["start_time"].dt.floor("h")
    trips.drop(columns=["vehicle_type"], inplace=True)
    trips['trip_count'] = trips.groupby(['start_location_id', 'hour_start'])['trip_id'].transform('count')

    # Add time features
    trips["hour"] = trips["hour_start"].dt.hour
    trips["day"] = trips["hour_start"].dt.day
    trips["day_of_week"] = trips["hour_start"].dt.dayofweek
    trips["is_weekend"] = trips["day_of_week"].isin([5, 6]).astype(int)
    trips["month"] = trips["hour_start"].dt.month
    trips["year"] = trips["hour_start"].dt.year
    trips["week_of_year"] = trips["hour_start"].dt.isocalendar().week
    
    return trips

# lagged features moved completely to feature engineer
# ---------------------------------------------------------------------------
# 4  Public orchestrator
# ---------------------------------------------------------------------------

def build_features(
    trips: pd.DataFrame,
    drivers: pd.DataFrame,
    passengers: pd.DataFrame,
    logger: Optional[StructuredLogger] = None,
) -> pd.DataFrame:
    logger = logger or StructuredLogger("feat_engine")
    logger.info("Starting feature‑engineering pipeline …")

    trips = pd.merge(trips, drivers, on="driver_id", how="left", suffixes=("", "_driver"))
    trips = pd.merge(trips, passengers, on="passenger_id", how="left", suffixes=("", "_passenger"))
    logger.debug("Entities joined ✚")
    
    trips = compute_entity_stats(trips)
    logger.debug("Entity stats computed ✚")

    trips = hourly_aggregates(trips)
    logger.debug("Hourly aggregates attached ✚")

    logger.info("Feature‑engineering done.")
    return trips

# ---------------------------------------------------------------------------
# Enhanced Data Processor Class (wrapper around the functional pipeline)
# ---------------------------------------------------------------------------

class EnhancedDataProcessor:
    """
    Advanced data processor for ride demand forecasting with robust feature engineering.
    
    This class handles loading data from parquet files, joining related tables,
    creating entity-level features, and generating temporal aggregates with proper
    safeguards against data leakage.
    """
    
    def __init__(self, config: ConfigManager, logger: Optional[StructuredLogger] = None):
        """
        Initialize the data processor.
        
        Args:
            config: Configuration manager
            logger: Structured logger instance
        """
        self.config = config
        self.logger = logger or StructuredLogger("enhanced_data_processor")
        
        # Extract data paths from config
        self.data_paths = self.config.data_paths
        self.raw_path = self.data_paths.get('raw_path')
        self.processed_path = self.data_paths.get('processed_path')
        
        # Create processed directory if it doesn't exist
        if self.processed_path and not os.path.exists(self.processed_path):
            os.makedirs(self.processed_path, exist_ok=True)
        
        # Initialize datasets container
        self.datasets = {}
        self.metadata = {}
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all required datasets (trips, drivers, passengers).
        
        Returns:
            Dictionary of loaded DataFrames
        """
        with self.logger.operation("load_data"):
            # Load trips data
            trips_path = Path(self.raw_path) / "trips.parquet"
            self.logger.info(f"Loading trips data from {trips_path}")
            trips_df = pd.read_parquet(trips_path)
            
            # Load drivers data
            drivers_path = Path(self.raw_path) / "drivers.parquet"
            self.logger.info(f"Loading drivers data from {drivers_path}")
            drivers_df = pd.read_parquet(drivers_path)
            
            # Load passengers data
            passengers_path = Path(self.raw_path) / "passengers.parquet"
            self.logger.info(f"Loading passengers data from {passengers_path}")
            passengers_df = pd.read_parquet(passengers_path)
            
            # Convert date columns to datetime
            trips_df['start_time'] = pd.to_datetime(trips_df['start_time'])
            trips_df['end_time'] = pd.to_datetime(trips_df['end_time'])
            drivers_df['signup_date'] = pd.to_datetime(drivers_df['signup_date'])
            passengers_df['signup_date'] = pd.to_datetime(passengers_df['signup_date'])
            
            # Store datasets for later use
            self.datasets['trips'] = trips_df
            self.datasets['drivers'] = drivers_df
            self.datasets['passengers'] = passengers_df
            
            # Log the loaded data shapes
            self.logger.info(f"Successfully loaded all datasets: trips={len(trips_df)}, "
                          f"drivers={len(drivers_df)}, passengers={len(passengers_df)}")
                          
            return self.datasets
    
    def process_full_pipeline(self) -> pd.DataFrame:
        """
        Run the full data processing pipeline.
        
        Returns:
            Fully enriched DataFrame
        """
        with self.logger.operation("process_full_pipeline"):
            # Load data if not already loaded
            if not self.datasets.get('trips'):
                self.load_data()
            
            # Get the loaded data
            trips_df = self.datasets['trips']
            drivers_df = self.datasets['drivers']
            passengers_df = self.datasets['passengers']
            
            # Use the functional pipeline to process data
            processed_df = build_features(
                trips_df, 
                drivers_df, 
                passengers_df, 
                self.logger
            )
            
            # Store the result
            self.datasets['prepared_trips'] = processed_df
            
            self.logger.info("Full data processing pipeline completed successfully")
            return processed_df
            
    def load_all_data(self) -> pd.DataFrame:
        """
        Load all data from raw sources and process it.
        This is the main entry point for data processing.
        
        Returns:
            Processed DataFrame ready for feature engineering
        """
        with self.logger.operation("load_all_data"):
            self.logger.info("Loading and processing all data from raw sources")
            
            try:
                # First try to load the final processed data if it exists
                self.logger.info("Attempting to load pre-processed data first")
                processed_df = self.load_processed_data("prepared_trips.parquet")
                self.datasets['prepared_trips'] = processed_df
                self.logger.info("Successfully loaded pre-processed data")
                return processed_df
            except FileNotFoundError:
                self.logger.info("No prepared_trips.parquet found, running full processing pipeline")
                
                # Process everything from scratch
                processed_df = self.process_full_pipeline()
                
                processed_df.fillna(0, inplace=True)
                processed_df = processed_df.reset_index()

                # Save the processed data
                self.save_processed_data(processed_df, "prepared_trips.parquet")
                
                return processed_df
    
    def load_processed_data(self, file_name: str) -> pd.DataFrame:
        """
        Load a processed data file from the processed directory.
        
        Args:
            file_name: Name of the file to load
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        with self.logger.operation("load_processed_data"):
            file_path = Path(self.processed_path) / file_name
            
            if not file_path.exists():
                self.logger.error(f"Processed file not found: {file_path}")
                raise FileNotFoundError(f"Processed file not found: {file_path}")
                
            self.logger.info(f"Loading processed data from {file_path}")
            df = pd.read_parquet(file_path)
            self.logger.info(f"Loaded {len(df)} records from {file_path}")
            
            return df
            
    def save_processed_data(self, df: pd.DataFrame, file_name: str) -> str:
        """
        Save a DataFrame to the processed directory.
        
        Args:
            df: DataFrame to save
            file_name: Name of the file to save
            
        Returns:
            Path to the saved file
        """
        with self.logger.operation("save_processed_data"):
            # Create processed directory if it doesn't exist
            os.makedirs(self.processed_path, exist_ok=True)
            
            file_path = Path(self.processed_path) / file_name
            self.logger.info(f"Saving processed data to {file_path}")
            
            df.to_parquet(file_path, compression='snappy', index=False)
            self.logger.info(f"Saved {len(df)} records to {file_path}")
            
            return str(file_path)
    
    def save_output(self, output_file: str = 'prepared_trips.parquet') -> str:
        """
        Save the fully enriched dataframe as a parquet file.
        
        Args:
            output_file: Name of the output file
            
        Returns:
            Path to the saved file
        """
        with self.logger.operation("save_output"):
            if 'prepared_trips' not in self.datasets:
                self.logger.error("No prepared_trips dataset available to save")
                raise ValueError("No prepared_trips dataset available to save")
            
            output_path = Path(self.processed_path) / output_file
            
            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as parquet with snappy compression, no index
            self.datasets['prepared_trips'].to_parquet(
                output_path,
                compression='snappy',
                index=False
            )
            
            self.logger.info(f"Saved prepared_trips to {output_path}")
            return str(output_path)

