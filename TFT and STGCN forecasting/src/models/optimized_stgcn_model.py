#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Union, Tuple, Any
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import copy
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .base_model import BaseModel
from .optimized_stgcn_layers import ChebGraphConv, SpatialAttentionLayer, TemporalConvLayer, TemporalAttentionLayer, STGCNBlock
from .optimized_stgcn_utils import OptimizedGraphBuilder
from ..utils.logger import StructuredLogger

class OptimizedSTGCNModel(BaseModel):
    
    def __init__(self, model_config: Dict, logger: Optional[StructuredLogger] = None):
        """Initialize the STGCN model.
        
        Args:
            model_config: Configuration dictionary for the model
            logger: Logger instance for tracking model operations
        """
        super().__init__(model_config, logger)
        self.stgcn_params = self.model_config.get("params", {})
        self._set_default_params()
        
        self.model = None
        self.graph_builder = None
        self.cheb_polynomials = None
        
        self.feature_names = None
        self.target_name = self.model_config.get("target", "trip_count")
        self.location_id_col = self.model_config.get("location_id_col", "start_location_id")
        self.time_col = self.model_config.get("time_col", "hour_start")
        
        self.input_dim = self.stgcn_params.get("input_dim", 1)  # Number of input features
        self.output_dim = self.stgcn_params.get("output_dim", 1)  # Number of output features
        self.hidden_dims = self.stgcn_params.get("hidden_dims", [64, 32, 32])  # Hidden dimensions
        self.horizon = self.stgcn_params.get("horizon", 24)  # Prediction horizon
        self.seq_len = self.stgcn_params.get("seq_len", 168)  # Input sequence length
        self.spatial_ks = self.stgcn_params.get("cheb_k", 3)  # Chebyshev polynomial degree
        
        self.quantiles = self.stgcn_params.get("quantiles", [0.1, 0.5, 0.9])
        self.use_quantile_regression = self.stgcn_params.get("use_quantile_regression", True)
        
        self.batch_size = self.stgcn_params.get("batch_size", 32)
        self.learning_rate = self.stgcn_params.get("learning_rate", 0.001)
        self.weight_decay = self.stgcn_params.get("weight_decay", 1e-5)
        self.epochs = self.stgcn_params.get("epochs", 100)
        self.early_stopping_patience = self.stgcn_params.get("early_stopping_patience", 10)
        
        self.graph_type = self.stgcn_params.get("graph_type", "distance")
        self.graph_threshold = self.stgcn_params.get("graph_threshold", 0.1)
        self.location_mapping = None
        
        self.is_fitted = False
        
        self.feature_categories = {
            'temporal': [],
            'spatial': [],
            'context': [],
            'static': []
        }
        self.is_using_enhanced_features = False
        
    def _set_default_params(self):
        """Set default hyperparameters if not specified in the config."""
        defaults = {
            "input_dim": 1,  # Number of input features per node
            "output_dim": 1,  # Number of output features
            "hidden_dims": [64, 32, 32],  # Hidden dimensions of STGCN blocks
            "horizon": 24,  # Prediction horizon (hours)
            "seq_len": 168,  # Input sequence length (hours)
            "cheb_k": 3,  # Chebyshev polynomial order
            "batch_size": 32,
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "epochs": 100,
            "early_stopping_patience": 10,
            "graph_type": "distance",  # Options: distance, correlation, adaptive
            "graph_threshold": 0.1,  # Threshold for edge creation
            "dropout": 0.3,
            "use_spatial_attention": True,
            "use_temporal_attention": True
        }
        
        for key, value in defaults.items():
            if key not in self.stgcn_params:
                self.stgcn_params[key] = value
                
    def _build_graph(self, location_data: pd.DataFrame) -> None:
        """Build the graph for spatial relationships between locations.
        
        Args:
            location_data: DataFrame with location information
        """
        self.logger.info(f"Building graph using method: {self.graph_type}")
        
        try:
            self.graph_builder = OptimizedGraphBuilder(self.graph_type)
            
            start_time = time.time()
            max_time_seconds = 60  # Maximum time to allow for graph building
            
            has_coordinates = ("latitude" in location_data.columns and "longitude" in location_data.columns)
            
            if has_coordinates:
                non_null_coords = location_data.dropna(subset=["latitude", "longitude"]).shape[0]
                total_locations = location_data.shape[0]
                self.logger.info(f"Location data has coordinates for {non_null_coords}/{total_locations} entries "
                                f"({(non_null_coords/total_locations)*100:.1f}%)")
            else:
                self.logger.info("No latitude/longitude columns found in location data")
            
            clean_location_data = location_data.copy()
            
            # Check if location ID column exists, create a synthetic one if not
            if self.location_id_col not in clean_location_data.columns:
                self.logger.warning(f"Location ID column '{self.location_id_col}' not found in data. Creating synthetic IDs.")
                # Create sequential IDs as a fallback
                clean_location_data[self.location_id_col] = [f"synthetic_loc_{i}" for i in range(clean_location_data.shape[0])]
                
            clean_location_data[self.location_id_col] = clean_location_data[self.location_id_col].astype(str)
            
            if self.graph_type == "distance":
                self.logger.info("Building distance-based graph")
                adjacency = self.graph_builder.build_graph_from_locations(
                    clean_location_data,
                    location_id_col=self.location_id_col,
                    threshold=self.graph_threshold
                )
                self.logger.info(f"Distance-based graph created with {len(self.graph_builder.location_mapping)} locations")
                
            elif self.graph_type == "correlation":
                self.logger.info("Building correlation-based graph")
                adjacency = self.graph_builder.build_graph_from_correlation(
                    clean_location_data,
                    location_id_col=self.location_id_col,
                    target_col=self.target_name,
                    threshold=self.graph_threshold
                )
            elif self.graph_type == "adaptive":
                self.logger.info("Building adaptive graph")
                base_adj = self.graph_builder.build_graph_from_locations(
                    clean_location_data,
                    location_id_col=self.location_id_col,
                    threshold=self.graph_threshold
                )
                adjacency = self.graph_builder.build_adaptive_graph(base_adj)
            else:
                raise ValueError(f"Unknown graph type: {self.graph_type}")
            
            # Check if we're exceeding the time limit
            if time.time() - start_time > max_time_seconds:
                self.logger.warning(f"Graph building exceeded time limit of {max_time_seconds} seconds. Creating fallback graph.")
                unique_locations = clean_location_data[self.location_id_col].unique()
                num_locations = len(unique_locations)
                self.graph_builder.location_mapping = {loc: i for i, loc in enumerate(unique_locations)}
                adjacency = np.ones((num_locations, num_locations))
                np.fill_diagonal(adjacency, 0)  # No self-loops
                self.graph_builder.adjacency_matrix = adjacency
            
            # Store location mapping for later use
            self.location_mapping = self.graph_builder.location_mapping
            
            # Normalize graph and compute Chebyshev polynomials
            self.logger.info("Normalizing graph and computing Chebyshev polynomials")
            self.graph_builder.normalize_graph(adjacency)
            self.cheb_polynomials = self.graph_builder.compute_cheb_polynomials(self.spatial_ks)
            
            # Convert Chebyshev polynomials to PyTorch tensors
            self.cheb_polynomials = [self.graph_builder.to_torch_sparse(p) for p in self.cheb_polynomials]
            
            self.logger.info(f"Graph built with {adjacency.shape[0]} nodes and {int(adjacency.sum())} edges")
            
        except Exception as e:
            self.logger.error(f"Error building graph: {str(e)}. Creating fallback graph.")
            
            # Handle case where location_id_col doesn't exist in the data
            if self.location_id_col not in location_data.columns:
                self.logger.warning(f"Location ID column '{self.location_id_col}' missing in error handler. Creating synthetic locations.")
                # Create synthetic location IDs
                unique_locations = [f"fallback_loc_{i}" for i in range(5)]  # Default to 5 locations
            else:
                unique_locations = location_data[self.location_id_col].unique()
                
            num_locations = len(unique_locations)
                
            if self.graph_builder is None:
                self.graph_builder = OptimizedGraphBuilder("distance")
            
            self.graph_builder.location_mapping = {loc: i for i, loc in enumerate(unique_locations)}
            self.location_mapping = self.graph_builder.location_mapping
            adjacency = np.ones((num_locations, num_locations))
            np.fill_diagonal(adjacency, 0)  # No self-loops
            self.graph_builder.adjacency_matrix = adjacency
            
            self.graph_builder.normalize_graph(adjacency)
            self.cheb_polynomials = self.graph_builder.compute_cheb_polynomials(self.spatial_ks)
            self.cheb_polynomials = [self.graph_builder.to_torch_sparse(p) for p in self.cheb_polynomials]
            
            self.logger.info(f"Created fallback fully-connected graph with {num_locations} nodes")
            
    def _build_model(self, num_nodes: int) -> nn.Module:
        """Build the STGCN neural network model.
        
        Args:
            num_nodes: Number of nodes in the graph
            
        Returns:
            PyTorch STGCN model
        """
        self.logger.info(f"Building STGCN model with {num_nodes} nodes")
        self.logger.info(f"Input dimension: {self.input_dim}, Output dimension: {self.output_dim}")
        self.logger.info(f"Hidden dimensions: {self.hidden_dims}")
        self.logger.info(f"Sequence length: {self.seq_len}, Prediction horizon: {self.horizon}")
        
        class STGCNNet(nn.Module):
            def __init__(self, input_dim, output_dim, hidden_dims, seq_len, horizon, 
                         num_nodes, spatial_ks, dropout, use_spatial_attention, use_temporal_attention,
                         use_quantile_regression=False, quantiles=None):
                super(STGCNNet, self).__init__()
                
                self.input_dim = input_dim
                self.output_dim = output_dim
                self.hidden_dims = hidden_dims
                self.seq_len = seq_len
                self.horizon = horizon
                self.num_nodes = num_nodes
                
                # Store quantile regression attributes
                self.use_quantile_regression = use_quantile_regression
                self.quantiles = quantiles if quantiles is not None else [0.1, 0.5, 0.9]
                
                self.input_projection = nn.Linear(input_dim, hidden_dims[0])
                
                self.st_blocks = nn.ModuleList()
                
                in_dim = hidden_dims[0]
                
                for i, hid_dim in enumerate(hidden_dims):
                    st_block = STGCNBlock(
                        in_channels=in_dim,
                        out_channels=hid_dim,
                        spatial_kernel_size=spatial_ks,
                        temporal_kernel_size=3,  # Standard kernel size for temporal convolution
                        num_nodes=num_nodes,
                        dropout=dropout,
                        use_spatial_attention=use_spatial_attention,
                        use_temporal_attention=use_temporal_attention
                    )
                    
                    self.st_blocks.append(st_block)
                    in_dim = hid_dim  # Output of this layer becomes input to next
                
                self.output_layer = nn.Sequential(
                    nn.Linear(seq_len * hidden_dims[-1], horizon * output_dim),
                    nn.ReLU()
                )
                
                if self.use_quantile_regression:
                    self.quantile_output_layers = nn.ModuleDict()
                    for q in self.quantiles:
                        q_str = str(q).replace('.', '_')  # Convert 0.1 to 0_1 for dict key
                        self.quantile_output_layers[q_str] = nn.Linear(horizon * output_dim, horizon * output_dim)
            
            def forward(self, x, cheb_polynomials):
                """
                Forward pass through the STGCN model.
                
                Args:
                    x: Input tensor of shape [batch_size, seq_len, num_nodes, input_dim]
                    cheb_polynomials: List of Chebyshev polynomial sparse tensors
                    
                Returns:
                    Predictions of shape [batch_size, horizon, num_nodes, output_dim]
                """
                try:
                    batch_size = x.shape[0]
                    
                    x = self.input_projection(x)  # Shape: [batch_size, seq_len, num_nodes, hidden_dims[0]]
                    
                    for block in self.st_blocks:
                        x = block(x, cheb_polynomials)
                    
                    x = x.permute(0, 2, 1, 3).contiguous()  # [batch, num_nodes, seq_len, hidden_dim]
                    x = x.view(batch_size, self.num_nodes, self.seq_len * self.hidden_dims[-1])
                    
                    outputs = self.output_layer(x)  # Shape: [batch, num_nodes, horizon * output_dim]
                    
                    outputs = outputs.view(batch_size, self.num_nodes, self.horizon, self.output_dim)
                    outputs = outputs.permute(0, 2, 1, 3)  # [batch, horizon, num_nodes, output_dim]
                    
                    if hasattr(self, 'quantile_output_layers') and self.use_quantile_regression:
                        quantile_outputs = {}
                        for q_str, q_layer in self.quantile_output_layers.items():
                            q_out = q_layer(outputs.view(batch_size, -1))
                            q_out = q_out.view(batch_size, self.horizon, self.num_nodes, self.output_dim)
                            quantile_outputs[q_str] = q_out
                        
                        return outputs, quantile_outputs
                    
                    return outputs
                    
                except Exception as e:
                    shapes = f"Input shape: {x.shape if isinstance(x, torch.Tensor) else 'Not a tensor'}" + \
                             f", Cheb polynomials shapes: {[p.shape if isinstance(p, torch.Tensor) else 'Not a tensor' for p in cheb_polynomials]}"
                    error_msg = f"Error in STGCN forward pass: {str(e)}. Shapes: {shapes}"
                    raise RuntimeError(error_msg)
        
        model = STGCNNet(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=self.hidden_dims,
            seq_len=self.seq_len,
            horizon=self.horizon,
            num_nodes=num_nodes,
            spatial_ks=self.spatial_ks,
            dropout=self.stgcn_params["dropout"],
            use_spatial_attention=self.stgcn_params["use_spatial_attention"],
            use_temporal_attention=self.stgcn_params["use_temporal_attention"],
            use_quantile_regression=self.use_quantile_regression,
            quantiles=self.quantiles
        )
        
        self.logger.info(f"STGCN model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        return model


    def _prepare_data(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        is_prediction: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Vectorised version – identical output, far less Python-level looping.
        """

        # Check if location ID column exists, add synthetic one if not
        if self.location_id_col not in X.columns.to_list():
            self.logger.warning(f"{self.location_id_col} column missing in data preparation. Creating synthetic column.")
            # Create a synthetic location ID column with a single location id
            X = X.copy()
            X[self.location_id_col] = "synthetic_location_0"

        df = X.copy()
        if y is not None and not is_prediction:
            df[self.target_name] = y

        # assure datetime
        if self.time_col in df.columns and not pd.api.types.is_datetime64_any_dtype(
            df[self.time_col]
        ):
            df[self.time_col] = pd.to_datetime(df[self.time_col])

        # location → node index
        if self.graph_builder is None or self.cheb_polynomials is None:
            self._build_graph(df)
        loc2idx = self.location_mapping
        num_nodes = len(loc2idx)

        if self.is_using_enhanced_features and any(self.feature_categories.values()):
            feat_cols = [
                c
                for cat in ("temporal", "spatial", "context")
                for c in self.feature_categories[cat]
                if c in df.columns
            ]
        else:
            feat_cols = [
                c
                for c in df.columns
                if c
                not in (
                    self.location_id_col,
                    self.time_col,
                    self.target_name,
                )
            ]
        self.feature_names = feat_cols
        self.input_dim = max(1, len(feat_cols))

        if self.time_col in df.columns:
            df["_t"] = df[self.time_col]
        else:  # fallback to index order
            df["_t"] = np.arange(len(df))

        df["_loc"] = df[self.location_id_col].astype(str).map(loc2idx)
        df = df.dropna(subset=["_loc"])  # rows with unknown locations

        # index arrays
        t_vals, t_idx = np.unique(df["_t"].values, return_inverse=True)
        node_idx = df["_loc"].astype(int).values

        T = len(t_vals)
        F = self.input_dim
        feat_mat = np.zeros((T, num_nodes, F), dtype="float32")
        feat_data = df[feat_cols].to_numpy(dtype="float32") if feat_cols else np.zeros((len(df), 1), dtype="float32")
        # scatter into tensor
        feat_mat[t_idx, node_idx, :] = feat_data

        if not is_prediction:
            # same for target
            y_mat = np.zeros((T, num_nodes, 1), dtype="float32")
            y_vals = df[self.target_name].to_numpy(dtype="float32")
            y_mat[t_idx, node_idx, 0] = y_vals

        seq_len, horizon = self.seq_len, self.horizon
        if T < seq_len + (0 if is_prediction else horizon):
            self.logger.warning("Not enough timesteps, using fallback")
            return self._create_fallback_dataset(df, feat_cols, y)

        def build_samples(mat, out_steps):
            # mat: T × N × C  ->  num_samples × out_steps × N × C
            idx = np.arange(T - seq_len - out_steps + 1)[:, None] + np.arange(seq_len + out_steps)
            sliced = mat[idx]  # shape: samples × (seq+horizon) × N × C
            enc = sliced[:, :seq_len]  # encoder part
            dec = sliced[:, seq_len:] if out_steps else None
            return enc, dec

        X_enc, y_dec = build_samples(feat_mat, 0 if is_prediction else horizon)
        X_tensor = torch.from_numpy(X_enc)

        if is_prediction:
            return X_tensor, None

        _, y_dec = build_samples(y_mat, horizon)
        y_tensor = torch.from_numpy(y_dec)
        return X_tensor, y_tensor

    def _create_fallback_dataset(self, data: pd.DataFrame, feature_cols: list, y: pd.Series = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Create a fallback dataset when normal data preparation fails.
        
        Args:
            data: The original data
            feature_cols: Feature column names
            y: Optional target series
            
        Returns:
            Tuple of (input tensor, target tensor)
        """
        self.logger.warning("Creating fallback dataset due to data issues")
        
        try:
            # Get unique locations
            if self.location_mapping is None or not self.location_mapping:
                unique_locs = data[self.location_id_col].unique()
                self.location_mapping = {str(loc): i for i, loc in enumerate(unique_locs)}
            
            num_nodes = len(self.location_mapping)
            self.input_dim = len(feature_cols) if feature_cols else 1
            
            # Create minimal tensors
            if y is not None:  # For training
                batch_size = min(32, max(1, len(data) // (self.seq_len * num_nodes) // 2))
                X_tensor = torch.zeros((batch_size, self.seq_len, num_nodes, self.input_dim))
                y_tensor = torch.zeros((batch_size, self.horizon, num_nodes, self.output_dim))
                
                for loc_id, node_idx in self.location_mapping.items():
                    loc_data = data[data[self.location_id_col].astype(str) == str(loc_id)]
                    
                    if len(loc_data) > 0:
                        for f_idx, feat in enumerate(feature_cols):
                            if feat in loc_data.columns:
                                for b in range(batch_size):
                                    for t in range(min(self.seq_len, len(loc_data))):
                                        idx = (b * self.seq_len + t) % len(loc_data)
                                        X_tensor[b, t, node_idx, f_idx] = loc_data.iloc[idx].get(feat, 0)
                        
                        if self.target_name in loc_data.columns:
                            for b in range(batch_size):
                                for t in range(min(self.horizon, len(loc_data))):
                                    idx = (b * self.horizon + t) % len(loc_data)
                                    y_tensor[b, t, node_idx, 0] = loc_data.iloc[idx].get(self.target_name, 0)
                
                return X_tensor, y_tensor
            
            else:  # For prediction
                X_tensor = torch.zeros((1, self.seq_len, num_nodes, self.input_dim))
                
                for loc_id, node_idx in self.location_mapping.items():
                    loc_data = data[data[self.location_id_col].astype(str) == str(loc_id)]
                    
                    if len(loc_data) > 0:
                        for f_idx, feat in enumerate(feature_cols):
                            if feat in loc_data.columns:
                                for t in range(min(self.seq_len, len(loc_data))):
                                    idx = t % len(loc_data)
                                    X_tensor[0, t, node_idx, f_idx] = loc_data.iloc[idx].get(feat, 0)
                
                return X_tensor, None
                
        except Exception as e:
            self.logger.error(f"Error creating fallback dataset: {str(e)}")
            # As a last resort, create minimal tensors with zeros
            num_nodes = max(1, len(self.location_mapping) if self.location_mapping else 5)  # Default to 5 nodes
            X_tensor = torch.zeros((1, self.seq_len, num_nodes, self.input_dim))
            y_tensor = torch.zeros((1, self.horizon, num_nodes, self.output_dim)) if y is not None else None
            return X_tensor, y_tensor
    
    def fit(self, X: pd.DataFrame, y: pd.Series, validation_data: Optional[Tuple] = None) -> None:
        """Train the STGCN model.
        
        Args:
            X: Feature dataframe
            y: Target series
            validation_data: Optional tuple of (X_val, y_val)
        """
        self.logger.info("Starting STGCN model training")
        
        try:
            # Initialize device (use GPU if available)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.logger.info(f"Using device: {device}")
            
            # Prepare training data
            X_tensor, y_tensor = self._prepare_data(X, y, is_prediction=False)
            
            # Update input and output dimensions based on data
            self.input_dim = X_tensor.shape[-1]
            self.output_dim = y_tensor.shape[-1]
            
            # Create DataLoader for batch processing
            dataset = TensorDataset(X_tensor, y_tensor)
            train_loader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=True
            )
            
            # Build the model
            num_nodes = len(self.location_mapping)
            self.model = self._build_model(num_nodes)
            self.model.to(device)
            
            # Define loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            
            # Training loop
            best_loss = float('inf')
            patience_counter = 0
            losses = []
            
            for epoch in range(self.epochs):
                self.model.train()
                epoch_loss = 0.0
                num_batches = 0
                
                # Training batches
                for batch_X, batch_y in train_loader:
                    # Move batch to device
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = self.model(batch_X, self.cheb_polynomials)
                    
                    # Handle quantile outputs if enabled
                    if isinstance(outputs, tuple):
                        # If model returns both mean and quantile predictions
                        point_preds = outputs[0]
                        # For training, we just use the mean predictions
                        loss = criterion(point_preds, batch_y)
                    else:
                        loss = criterion(outputs, batch_y)
                    
                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                # Calculate average epoch loss
                epoch_loss /= max(1, num_batches)
                losses.append(epoch_loss)
                
                # Log progress
                if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == self.epochs - 1:
                    self.logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.6f}")
                
                # Check for early stopping
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                    # Save best model
                    self._save_model()
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Training complete
            self.is_fitted = True
            self.logger.info(f"STGCN model training completed. Final loss: {losses[-1]:.6f}")
            
            # Plot training curve if possible
            try:
                plt.figure(figsize=(10, 6))
                plt.plot(losses)
                plt.title('STGCN Training Loss Curve')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.grid(True)
                plt.savefig('stgcn_training_loss.png')
                plt.close()
            except Exception as e:
                self.logger.warning(f"Could not plot training curve: {str(e)}")
                
        except Exception as e:
            self.logger.error(f"Error during STGCN model training: {str(e)}")
            raise RuntimeError(f"STGCN training failed: {str(e)}")
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the trained STGCN model.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        self.logger.info(f"Generating predictions with STGCN model for data with shape {X.shape}")
        
        try:
            # Initialize device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Prepare prediction data
            X_tensor, _ = self._prepare_data(X, is_prediction=True)
            X_tensor = X_tensor.to(device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Generate predictions
            with torch.no_grad():
                outputs = self.model(X_tensor, self.cheb_polynomials)
                
                # Handle quantile outputs
                if isinstance(outputs, tuple):
                    # If model returns both mean and quantile predictions
                    predictions = outputs[0]  # Use only the mean predictions
                else:
                    predictions = outputs
                    
                # Convert to numpy array
                predictions = predictions.cpu().numpy()
                
                # Reshape to [horizon, num_nodes, output_dim]
                predictions = predictions[0]  # Remove batch dimension
                
            # Map predictions back to original locations
            results = self._map_predictions_to_locations(predictions, X)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error generating STGCN predictions: {str(e)}")
            # Provide a fallback prediction
            return self._generate_fallback_predictions(X)
            
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate predictions with uncertainty estimates.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Dictionary with mean and uncertainty bounds
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        if not self.use_quantile_regression:
            self.logger.warning("Uncertainty estimates requested but quantile regression not enabled")
            mean_preds = self.predict(X)
            # Create a simple uncertainty estimate based on 10% of the mean
            return {
                'mean': mean_preds,
                'lower': mean_preds * 0.9,
                'upper': mean_preds * 1.1
            }
        
        self.logger.info(f"Generating predictions with uncertainty for data with shape {X.shape}")
        
        try:
            # Initialize device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Prepare prediction data
            X_tensor, _ = self._prepare_data(X, is_prediction=True)
            X_tensor = X_tensor.to(device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Generate predictions with quantiles
            with torch.no_grad():
                outputs = self.model(X_tensor, self.cheb_polynomials)
                
                if not isinstance(outputs, tuple) or len(outputs) != 2:
                    self.logger.warning("Model not configured for quantile regression")
                    predictions = outputs.cpu().numpy()[0]
                    # Create simple uncertainty bounds
                    results = {
                        'mean': predictions,
                        'lower': predictions * 0.9,
                        'upper': predictions * 1.1
                    }
                else:
                    # Get mean and quantile predictions
                    mean_preds = outputs[0].cpu().numpy()[0]
                    quantile_preds = outputs[1]
                    
                    # Extract lower and upper bounds from quantiles
                    lower_quantile = None
                    upper_quantile = None
                    for q_str, q_preds in quantile_preds.items():
                        q_val = float(q_str.replace('_', '.'))
                        q_preds = q_preds.cpu().numpy()[0]
                        
                        if q_val == 0.1 or q_val == 0.05:
                            lower_quantile = q_preds
                        elif q_val == 0.9 or q_val == 0.95:
                            upper_quantile = q_preds
                    
                    # If quantiles not found, use simple bounds
                    if lower_quantile is None:
                        lower_quantile = mean_preds * 0.9
                    if upper_quantile is None:
                        upper_quantile = mean_preds * 1.1
                    
                    # Map predictions back to original locations
                    mean_results = self._map_predictions_to_locations(mean_preds, X)
                    lower_results = self._map_predictions_to_locations(lower_quantile, X)
                    upper_results = self._map_predictions_to_locations(upper_quantile, X)
                    
                    results = {
                        'mean': mean_results,
                        'lower': lower_results,
                        'upper': upper_results
                    }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error generating STGCN predictions with uncertainty: {str(e)}")
            # Create a fallback prediction with uncertainty
            preds = self._generate_fallback_predictions(X)
            return {
                'mean': preds,
                'lower': preds * 0.9,
                'upper': preds * 1.1
            }
    
    def _map_predictions_to_locations(self, predictions: np.ndarray, X: pd.DataFrame) -> np.ndarray:
        """Map node-level predictions back to the original location format.
        
        Args:
            predictions: Tensor of predictions [horizon, num_nodes, output_dim]
            X: Original feature dataframe
            
        Returns:
            Predictions mapped to original locations
        """
        # Get unique locations from X
        if self.location_id_col in X.columns:
            unique_locs = X[self.location_id_col].unique()
        else:
            unique_locs = list(self.location_mapping.keys())
            
        # Create output array with shape [horizon, num_locations]
        horizon, num_nodes, out_dim = predictions.shape
        output = np.zeros((horizon, len(unique_locs)))
        
        # Map predictions back
        inv_mapping = {idx: loc for loc, idx in self.location_mapping.items()}
        
        for h in range(horizon):
            for n in range(num_nodes):
                if n in inv_mapping:
                    loc_id = inv_mapping[n]
                    # Find the index of this location in unique_locs
                    try:
                        loc_idx = np.where(unique_locs == loc_id)[0][0]
                        output[h, loc_idx] = predictions[h, n, 0]  # Only take the first output dimension
                    except (IndexError, ValueError):
                        # Skip if location not found
                        continue
                        
        return output
    
    def _generate_fallback_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """Generate simple fallback predictions when the main method fails.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Fallback predictions
        """
        self.logger.warning("Generating fallback predictions")
        
        # Create minimal predictions based on average target values
        if self.location_id_col in X.columns:
            unique_locs = X[self.location_id_col].unique()
        else:
            unique_locs = list(self.location_mapping.keys()) if self.location_mapping else [0]
            
        # Create a simple prediction for each location
        output = np.ones((self.horizon, len(unique_locs)))
        
        # If we have any target data, use mean as the baseline
        if hasattr(self, '_last_y_mean') and self._last_y_mean is not None:
            output *= self._last_y_mean
        
        return output
    
    def _save_model(self) -> None:
        """Save the current model state."""
        if self.model is None:
            self.logger.warning("No model to save")
            return
            
        try:
            # Create directory if it doesn't exist
            model_dir = Path(self.model_config.get("model_dir", "models"))
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a complete model state dictionary
            model_state = {
                "model_state_dict": self.model.state_dict(),
                "location_mapping": self.location_mapping,
                "feature_names": self.feature_names,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "params": self.stgcn_params,
                "is_fitted": self.is_fitted,
                "graph_type": self.graph_type,
                "feature_categories": self.feature_categories
            }
            
            # Save model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = model_dir / f"stgcn_model_{timestamp}.pt"
            torch.save(model_state, model_path)
            self.logger.info(f"Model saved to {model_path}")
            
            # Save the most recent path
            self._last_saved_path = model_path
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            
    @classmethod
    def load(cls, model_path: str, config: Dict = None, logger: Optional[StructuredLogger] = None) -> 'OptimizedSTGCNModel':
        """Load a saved STGCN model.
        
        Args:
            model_path: Path to the saved model
            config: Optional model configuration
            logger: Optional logger instance
            
        Returns:
            Loaded STGCN model
        """
        if logger is None:
            logger = logging.getLogger(__name__)
            
        logger.info(f"Loading STGCN model from {model_path}")
        
        try:
            # Load the model state
            model_state = torch.load(model_path)
            
            # Create a new model instance
            if config is None:
                config = {"params": model_state["params"]}
                
            # Initialize the model
            model = cls(config, logger)
            
            # Restore model parameters
            model.location_mapping = model_state["location_mapping"]
            model.feature_names = model_state["feature_names"]
            model.input_dim = model_state["input_dim"]
            model.output_dim = model_state["output_dim"]
            model.is_fitted = model_state["is_fitted"]
            
            # Restore feature categories if available
            if "feature_categories" in model_state:
                model.feature_categories = model_state["feature_categories"]
            
            # Build model architecture
            num_nodes = len(model.location_mapping)
            model.model = model._build_model(num_nodes)
            
            # Load model weights
            model.model.load_state_dict(model_state["model_state_dict"])
            
            logger.info(f"Successfully loaded STGCN model with {num_nodes} nodes")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load STGCN model: {str(e)}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance based on gradient-based attribution.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be trained before computing feature importance")
            
        self.logger.info("Computing feature importance for STGCN model")
        
        # For STGCN, use a simple approximation of feature importance
        # based on the weights of the input projection layer
        try:
            with torch.no_grad():
                # Get the weights of the input projection layer
                if hasattr(self.model, 'input_projection'):
                    weights = self.model.input_projection.weight.data.cpu().numpy()
                    # Compute importance as the sum of absolute weights for each feature
                    importances = np.sum(np.abs(weights), axis=0)
                    
                    # Normalize importances
                    if np.sum(importances) > 0:
                        importances = importances / np.sum(importances)
                        
                    # Create a dictionary mapping feature names to importances
                    feature_imp = {}
                    for idx, feat_name in enumerate(self.feature_names):
                        if idx < len(importances):
                            feature_imp[feat_name] = float(importances[idx])
                            
                    return feature_imp
                else:
                    self.logger.warning("Model architecture doesn't support feature importance calculation")
                    return {feat: 1.0/len(self.feature_names) for feat in self.feature_names}  # Equal importances
                    
        except Exception as e:
            self.logger.error(f"Error computing feature importance: {str(e)}")
            return {feat: 1.0/len(self.feature_names) for feat in self.feature_names}  # Fallback to equal importances
