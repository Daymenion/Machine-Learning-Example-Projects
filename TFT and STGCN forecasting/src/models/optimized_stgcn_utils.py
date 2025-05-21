#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
import time
import warnings

class OptimizedGraphBuilder:
    """A robust graph builder class for spatial-temporal graph models.
    
    This class builds adjacency matrices for graph neural networks with
    multiple methods for graph construction and robust error handling.
    """
    
    def __init__(self, graph_type: str = "distance"):
        """Initialize the graph builder.
        
        Args:
            graph_type: Type of graph to build ("distance", "correlation", "adaptive")
        """
        self.graph_type = graph_type
        self.location_mapping = {}  # Maps location IDs to node indices
        self.adjacency_matrix = None
        self.normalized_adj = None
        self.logger = logging.getLogger(__name__)
        
    def build_graph_from_locations(self, location_data: pd.DataFrame,
                                  location_id_col: str = "location_id",
                                  threshold: float = 0.1) -> np.ndarray:
        """Build a distance-based graph from location coordinates.
        
        Args:
            location_data: DataFrame with location information including coordinates
            location_id_col: Column name for location IDs
            threshold: Distance threshold for creating edges
            
        Returns:
            Adjacency matrix as a numpy array
        """
        self.logger.info("Building distance-based graph from location data")
        
        # Check if we have coordinates
        if "latitude" not in location_data.columns or "longitude" not in location_data.columns:
            self.logger.warning("No coordinate columns found in location data. Creating fallback graph.")
            return self._build_fallback_graph(location_data, location_id_col)
            
        # Get unique locations and create mapping
        unique_locations = location_data[location_id_col].unique()
        self.location_mapping = {str(loc): i for i, loc in enumerate(unique_locations)}
        num_locations = len(unique_locations)
        
        # Prepare coordinates for each location
        coords = np.zeros((num_locations, 2))
        mask = np.zeros(num_locations, dtype=bool)  # To track valid coordinates
        
        # Extract coordinates for each location
        for loc_id, node_idx in self.location_mapping.items():
            loc_data = location_data[location_data[location_id_col].astype(str) == str(loc_id)]
            
            # Use the first non-null coordinates
            loc_coords = loc_data[["latitude", "longitude"]].dropna().values
            if len(loc_coords) > 0:
                coords[node_idx] = loc_coords[0]
                mask[node_idx] = True
            
        # Check if we have enough valid coordinates
        valid_count = np.sum(mask)
        if valid_count < 0.5 * num_locations:
            self.logger.warning(f"Only {valid_count}/{num_locations} locations have valid coordinates. Creating fallback graph.")
            return self._build_fallback_graph(location_data, location_id_col)
            
        # Calculate distance matrix only for valid coordinates
        valid_indices = np.where(mask)[0]
        valid_coords = coords[valid_indices]
        
        # Calculate pairwise distances using haversine for valid coordinates
        valid_distances = self._haversine_distance(valid_coords, valid_coords)
        # Apply threshold to create adjacency matrix
        valid_adjacency = np.zeros((len(valid_indices), len(valid_indices)))
        # Normalize distances to [0, 1] for thresholding
        max_dist = np.max(valid_distances) if np.max(valid_distances) > 0 else 1.0
        valid_distances_normalized = valid_distances / max_dist
        # Create edges where normalized distance is below threshold
        valid_adjacency = (valid_distances_normalized < threshold).astype(float)
        np.fill_diagonal(valid_adjacency, 0)  # No self-loops
        
        # Expand to full adjacency matrix
        adjacency = np.zeros((num_locations, num_locations))
        for i, vi in enumerate(valid_indices):
            for j, vj in enumerate(valid_indices):
                adjacency[vi, vj] = valid_adjacency[i, j]
                
        # Ensure the graph is connected
        if not self._is_connected(adjacency):
            self.logger.warning("Graph is not connected. Adding minimum edges to ensure connectivity.")
            adjacency = self._ensure_connectivity(adjacency, valid_distances, valid_indices)
            
        self.logger.info(f"Distance-based graph created with {int(np.sum(adjacency))} edges for {num_locations} nodes")
        self.adjacency_matrix = adjacency
        return adjacency
        
    def build_graph_from_correlation(self, time_series_data: pd.DataFrame,
                                    location_id_col: str = "location_id",
                                    target_col: str = "target",
                                    time_col: str = "timestamp",
                                    threshold: float = 0.5) -> np.ndarray:
        """Build a correlation-based graph from time series data.
        
        Args:
            time_series_data: DataFrame with time series data
            location_id_col: Column name for location IDs
            target_col: Column name for target variable
            time_col: Column name for time index
            threshold: Correlation threshold for creating edges
            
        Returns:
            Adjacency matrix as a numpy array
        """
        self.logger.info("Building correlation-based graph from time series data")
        
        # Check if required columns exist
        if target_col not in time_series_data.columns:
            self.logger.warning(f"Target column {target_col} not found. Creating fallback graph.")
            return self._build_fallback_graph(time_series_data, location_id_col)
            
        # Get unique locations and create mapping
        unique_locations = time_series_data[location_id_col].unique()
        self.location_mapping = {str(loc): i for i, loc in enumerate(unique_locations)}
        num_locations = len(unique_locations)
        
        try:
            # Create a location-time matrix of the target variable
            pivot_data = time_series_data.pivot_table(
                index=time_col,
                columns=location_id_col,
                values=target_col,
                aggfunc='mean'
            ).fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Calculate correlation matrix
            correlation_matrix = pivot_data.corr().abs().values
            
            # Apply threshold to create adjacency matrix
            adjacency = (correlation_matrix > threshold).astype(float)
            np.fill_diagonal(adjacency, 0)  # No self-loops
            
            # Update location mapping to match pivot table columns
            if isinstance(pivot_data.columns, pd.Index):
                column_locs = [str(col) for col in pivot_data.columns]
                self.location_mapping = {str(loc): i for i, loc in enumerate(column_locs)}
            
            # Ensure the graph is connected
            if not self._is_connected(adjacency):
                self.logger.warning("Graph is not connected. Adding minimum edges to ensure connectivity.")
                # Add minimum spanning tree edges to ensure connectivity
                sorted_indices = np.argsort(correlation_matrix, axis=None)
                adjacency = self._add_min_edges_until_connected(adjacency, correlation_matrix, sorted_indices)
            
            self.logger.info(f"Correlation-based graph created with {int(np.sum(adjacency))} edges for {num_locations} nodes")
            self.adjacency_matrix = adjacency
            return adjacency
            
        except Exception as e:
            self.logger.error(f"Error building correlation graph: {str(e)}. Creating fallback graph.")
            return self._build_fallback_graph(time_series_data, location_id_col)
    
    def build_adaptive_graph(self, base_adjacency: Optional[np.ndarray] = None) -> np.ndarray:
        """Build an adaptive adjacency matrix that can be learned during training.
        
        Args:
            base_adjacency: Base adjacency matrix to initialize with
            
        Returns:
            Adaptive adjacency matrix
        """
        if base_adjacency is None:
            if self.adjacency_matrix is None:
                raise ValueError("No adjacency matrix available. Build a base graph first.")
            base_adjacency = self.adjacency_matrix
            
        # Create an adaptive adjacency with the same sparsity pattern
        adaptive_adjacency = base_adjacency.copy()
        
        self.logger.info(f"Created adaptive graph with {int(np.sum(adaptive_adjacency > 0))} edges")
        self.adjacency_matrix = adaptive_adjacency
        return adaptive_adjacency
        
    def normalize_graph(self, adjacency: Optional[np.ndarray] = None) -> np.ndarray:
        """Normalize the graph Laplacian for GCN operations.
        
        Args:
            adjacency: Adjacency matrix to normalize (uses stored one if None)
            
        Returns:
            Normalized adjacency matrix: D^(-1/2) A D^(-1/2)
        """
        if adjacency is None:
            if self.adjacency_matrix is None:
                raise ValueError("No adjacency matrix available. Build a graph first.")
            adjacency = self.adjacency_matrix
            
        try:
            # Add self-loops
            adj_with_loops = adjacency.copy()
            np.fill_diagonal(adj_with_loops, 1.0)
            # Calculate degree matrix
            degrees = np.sum(adj_with_loops, axis=1)
            # Create D^(-1/2)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                d_inv_sqrt = np.power(degrees, -0.5)
                d_inv_sqrt[np.isinf(d_inv_sqrt) | np.isnan(d_inv_sqrt)] = 0.0
                
            # Create diagonal matrix
            d_inv_sqrt_mat = np.diag(d_inv_sqrt)
            # Calculate normalized adjacency: D^(-1/2) A D^(-1/2)
            normalized_adj = np.matmul(np.matmul(d_inv_sqrt_mat, adj_with_loops), d_inv_sqrt_mat)
            
            self.normalized_adj = normalized_adj
            return normalized_adj
            
        except Exception as e:
            self.logger.error(f"Error normalizing graph: {str(e)}. Using identity matrix.")
            identity = np.eye(adjacency.shape[0])
            self.normalized_adj = identity
            return identity
            
    def compute_cheb_polynomials(self, k: int) -> List[np.ndarray]:
        """Compute the Chebyshev polynomials up to order k.
        
        Args:
            k: Maximum order of Chebyshev polynomials
            
        Returns:
            List of Chebyshev polynomial matrices
        """
        if self.normalized_adj is None:
            raise ValueError("Normalized adjacency not computed. Call normalize_graph first.")
            
        n = self.normalized_adj.shape[0]
        cheb_polynomials = []
        # T_0(L) = I
        cheb_polynomials.append(np.eye(n))
        # T_1(L) = L
        cheb_polynomials.append(self.normalized_adj)
        # T_k(L) = 2L*T_{k-1}(L) - T_{k-2}(L)
        for i in range(2, k):
            cheb_k = 2 * np.matmul(self.normalized_adj, cheb_polynomials[i-1]) - cheb_polynomials[i-2]
            cheb_polynomials.append(cheb_k)
        return cheb_polynomials
    
    def to_torch_sparse(self, adj_matrix: np.ndarray) -> torch.Tensor:
        """Convert adjacency matrix to PyTorch sparse tensor.
        
        Args:
            adj_matrix: Adjacency matrix to convert
            
        Returns:
            PyTorch sparse tensor
        """
        # Get indices of non-zero elements
        indices = np.where(adj_matrix != 0)
        values = adj_matrix[indices]
        
        # Convert to torch indices and values
        indices = torch.LongTensor(np.vstack(indices))
        values = torch.FloatTensor(values)
        
        # Create sparse tensor
        sparse_tensor = torch.sparse.FloatTensor(
            indices, values, 
            torch.Size(adj_matrix.shape)
        )
        return sparse_tensor
    
    def _build_fallback_graph(self, data: pd.DataFrame, location_id_col: str) -> np.ndarray:
        """Build a fallback graph when the main graph building method fails.
        
        Args:
            data: DataFrame with location information
            location_id_col: Column name for location IDs
            
        Returns:
            Adjacency matrix for a fully connected graph
        """
        self.logger.warning("Building fallback fully-connected graph")
        
        # Get unique locations
        unique_locations = data[location_id_col].unique()
        num_locations = len(unique_locations)
        # Create mapping
        self.location_mapping = {str(loc): i for i, loc in enumerate(unique_locations)}
        # Create a fully connected graph
        adjacency = np.ones((num_locations, num_locations))
        np.fill_diagonal(adjacency, 0)  # No self-loops
        
        self.logger.info(f"Created fallback fully-connected graph with {num_locations} nodes")
        self.adjacency_matrix = adjacency
        return adjacency
        
    def _haversine_distance(self, coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
        """Calculate haversine distance between sets of coordinates.
        
        Args:
            coords1: Array of [lat, lon] coordinates
            coords2: Array of [lat, lon] coordinates
            
        Returns:
            Distance matrix in kilometers
        """
        # Convert latitude and longitude to radians
        coords1_rad = np.radians(coords1)
        coords2_rad = np.radians(coords2)
        # Differences in coordinates
        lat1, lon1 = coords1_rad[:, 0][:, np.newaxis], coords1_rad[:, 1][:, np.newaxis]
        lat2, lon2 = coords2_rad[:, 0], coords2_rad[:, 1]
        # Haversine formula components
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        # Calculate distance
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        r = 6371  # Earth radius in kilometers
        return r * c
        
    def _is_connected(self, adjacency: np.ndarray) -> bool:
        """Check if the graph is connected using BFS.
        
        Args:
            adjacency: Adjacency matrix
            
        Returns:
            True if graph is connected, False otherwise
        """
        n = adjacency.shape[0]
        if n <= 1:
            return True
            
        # Start BFS from first node
        visited = np.zeros(n, dtype=bool)
        queue = [0]
        visited[0] = True
        while queue:
            node = queue.pop(0)
            neighbors = np.where(adjacency[node] > 0)[0]
            
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
                    
        # Check if all nodes were visited
        return np.all(visited)
        
    def _ensure_connectivity(self, adjacency: np.ndarray, distances: np.ndarray, valid_indices: np.ndarray) -> np.ndarray:
        """Ensure graph connectivity by adding minimum edges.
        
        Args:
            adjacency: Adjacency matrix
            distances: Distance matrix
            valid_indices: Indices of valid nodes
            
        Returns:
            Connected adjacency matrix
        """
        n = adjacency.shape[0]
        modified_adj = adjacency.copy()
        # Apply Kruskal's algorithm for minimum spanning tree on valid nodes
        # Sort edges by distance
        edge_list = []
        for i in range(len(valid_indices)):
            for j in range(i+1, len(valid_indices)):
                vi, vj = valid_indices[i], valid_indices[j]
                edge_list.append((vi, vj, distances[i, j]))
                
        # Sort edges by distance
        edge_list.sort(key=lambda x: x[2])
        # Union-find data structure
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
            
        def union(x, y):
            parent[find(x)] = find(y)
            
        # Apply Kruskal's algorithm
        for u, v, _ in edge_list:
            if find(u) != find(v):
                modified_adj[u, v] = modified_adj[v, u] = 1
                union(u, v)
                
        return modified_adj
        
    def _add_min_edges_until_connected(self, adjacency: np.ndarray, weights: np.ndarray, sorted_indices: np.ndarray) -> np.ndarray:
        """Add minimum edges until graph is connected, sorted by weight.
        
        Args:
            adjacency: Adjacency matrix
            weights: Weight matrix (e.g., correlation)
            sorted_indices: Sorted indices of weights
            
        Returns:
            Connected adjacency matrix
        """
        n = adjacency.shape[0]
        modified_adj = adjacency.copy()
        
        # Keep adding edges until the graph is connected
        while not self._is_connected(modified_adj) and len(sorted_indices) > 0:
            # Get the next highest correlation edge
            idx = sorted_indices[-1]
            sorted_indices = sorted_indices[:-1]
            # Convert flat index to 2D
            i, j = idx // n, idx % n
            # Add edge if it doesn't create a self-loop
            if i != j:
                modified_adj[i, j] = modified_adj[j, i] = 1
                
        return modified_adj
