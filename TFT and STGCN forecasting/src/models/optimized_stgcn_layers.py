#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class ChebGraphConv(nn.Module):
    """Chebyshev graph convolution layer.
    
    Implements spatial graph convolution using Chebyshev polynomials
    approximation of the graph Laplacian.
    
    Reference: Defferrard et al. "Convolutional Neural Networks on Graphs
    with Fast Localized Spectral Filtering", NeurIPS 2016.
    """
    
    def __init__(self, in_channels: int, out_channels: int, K: int):
        """Initialize ChebGraphConv layer.
        
        Args:
            in_channels: Number of input features
            out_channels: Number of output features
            K: Order of Chebyshev polynomial
        """
        super(ChebGraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.weights = nn.Parameter(torch.FloatTensor(K, in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize weights with Xavier uniform and biases with zeros."""
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.bias)
        
    def forward(self, x: torch.Tensor, cheb_polynomials: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass through the Chebyshev graph convolution layer.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, num_nodes, in_channels]
            cheb_polynomials: List of Chebyshev polynomials of the graph Laplacian
            
        Returns:
            Output tensor of shape [batch_size, seq_len, num_nodes, out_channels]
        """
        batch_size, seq_len, num_nodes, _ = x.shape
        x = x.reshape(batch_size * seq_len, num_nodes, self.in_channels)
        output = torch.zeros(batch_size * seq_len, num_nodes, self.out_channels, device=x.device)
        
        for k in range(self.K):
            T_k = cheb_polynomials[k]  # [num_nodes, num_nodes]
            
            if isinstance(T_k, torch.sparse.Tensor):
                x_k = torch.sparse.mm(T_k, x.reshape(batch_size * seq_len * num_nodes, self.in_channels))
                x_k = x_k.reshape(batch_size * seq_len, num_nodes, self.in_channels)
            else:
                x_k = torch.matmul(T_k, x)
                
            output += torch.einsum('bni,ioc->bno', x_k, self.weights[k])
        
        output += self.bias
        
        output = output.reshape(batch_size, seq_len, num_nodes, self.out_channels)
        return output

class SpatialAttentionLayer(nn.Module):
    """Spatial Attention Layer for enhancing STGCN models.
    
    This layer applies self-attention across the spatial dimension (nodes)
    to capture the dynamic spatial dependencies.
    """
    
    def __init__(self, in_channels: int, num_nodes: int):
        """Initialize Spatial Attention layer.
        
        Args:
            in_channels: Number of input features
            num_nodes: Number of nodes in the graph
        """
        super(SpatialAttentionLayer, self).__init__()
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        
        self.W_q = nn.Linear(in_channels, in_channels)
        self.W_k = nn.Linear(in_channels, in_channels)
        self.W_v = nn.Linear(in_channels, in_channels)
        
        self.W_out = nn.Linear(in_channels, in_channels)        
        self.layer_norm = nn.LayerNorm([num_nodes, in_channels])
        self.scale = torch.sqrt(torch.FloatTensor([in_channels])).item()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, num_nodes, in_channels]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, num_nodes, in_channels]
        """
        batch_size, seq_len, num_nodes, _ = x.shape
        
        # Reshape to process each time step
        x_reshaped = x.reshape(-1, num_nodes, self.in_channels)  # [batch_size * seq_len, num_nodes, in_channels]
        
        # Apply linear transformations
        q = self.W_q(x_reshaped)  # [batch_size * seq_len, num_nodes, in_channels]
        k = self.W_k(x_reshaped)  # [batch_size * seq_len, num_nodes, in_channels]
        v = self.W_v(x_reshaped)  # [batch_size * seq_len, num_nodes, in_channels]
        
        # Calculate attention scores
        attn = torch.matmul(q, k.permute(0, 2, 1))  # [batch_size * seq_len, num_nodes, num_nodes]
        attn = attn / self.scale
        
        # Apply softmax to get attention weights
        attn = F.softmax(attn, dim=-1)
        
        # Weighted aggregation of values
        output = torch.matmul(attn, v)  # [batch_size * seq_len, num_nodes, in_channels]
        output = self.W_out(output)  # Linear projection
        
        # Reshape back
        output = output.reshape(batch_size, seq_len, num_nodes, self.in_channels)
        
        # Residual connection and layer normalization
        output = self.layer_norm(x + output)
        
        return output

class TemporalConvLayer(nn.Module):
    """Temporal Convolution Layer for STGCN models.
    
    This layer applies 1D convolution across the temporal dimension
    with gated linear units for better gradient flow.
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, dilation: int = 1,
                 dropout: float = 0.3):
        """Initialize Temporal Convolution Layer.
        
        Args:
            in_channels: Number of input features
            out_channels: Number of output features
            kernel_size: Size of the convolution kernel
            dilation: Dilation rate for causal convolution
            dropout: Dropout rate
        """
        super(TemporalConvLayer, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Calculate padding to maintain sequence length
        self.padding = self._get_causal_padding(kernel_size, dilation)
        
        # Convolution filter
        self.filter_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding
        )
        
        # Gating mechanism
        self.gate_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding
        )
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection if input and output dimensions differ
        self.use_residual = (in_channels != out_channels)
        if self.use_residual:
            self.residual_conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1
            )
        
    def _get_causal_padding(self, kernel_size: int, dilation: int) -> int:
        """Calculate padding for causal convolution.
        
        Args:
            kernel_size: Size of the convolution kernel
            dilation: Dilation rate for causal convolution
            
        Returns:
            Required padding size
        """
        effective_k_size = (kernel_size - 1) * dilation + 1
        return (effective_k_size - 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal convolution.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, num_nodes, in_channels]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, num_nodes, out_channels]
        """
        batch_size, seq_len, num_nodes, _ = x.shape
        
        x = self.layer_norm(x)
        
        # Reshape for 1D convolution: [batch_size * num_nodes, in_channels, seq_len]
        x_reshaped = x.permute(0, 2, 3, 1).reshape(-1, self.in_channels, seq_len)
        filter_out = self.filter_conv(x_reshaped)
        gate_out = self.gate_conv(x_reshaped)
        
        # Apply GLU activation: filter_out ⊗ σ(gate_out)
        x_glu = filter_out * torch.sigmoid(gate_out)
        
        # Apply residual connection if needed
        if self.use_residual:
            residual = self.residual_conv(x_reshaped)
            x_glu = x_glu + residual
        
        # Apply dropout
        x_glu = self.dropout(x_glu)
        
        # Reshape back: [batch_size, seq_len, num_nodes, out_channels]
        # Note: We need to trim the sequence to handle causal padding
        x_out = x_glu[:, :, :seq_len].reshape(batch_size, num_nodes, self.out_channels, seq_len)
        x_out = x_out.permute(0, 3, 1, 2)
        
        return x_out

class TemporalAttentionLayer(nn.Module):
    """Temporal Attention Layer for enhancing STGCN models.
    
    This layer applies self-attention across the temporal dimension
    to capture long-range temporal dependencies.
    """
    
    def __init__(self, in_channels: int, seq_len: int):
        """Initialize Temporal Attention Layer.
        
        Args:
            in_channels: Number of input features
            seq_len: Length of the input sequence
        """
        super(TemporalAttentionLayer, self).__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        
        # Parameter matrices for calculating attention
        self.W_q = nn.Linear(in_channels, in_channels)
        self.W_k = nn.Linear(in_channels, in_channels)
        self.W_v = nn.Linear(in_channels, in_channels)
        
        # Output projection
        self.W_out = nn.Linear(in_channels, in_channels)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm([seq_len, in_channels])
        
        # Scaling factor
        self.scale = torch.sqrt(torch.FloatTensor([in_channels])).item()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal attention.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, num_nodes, in_channels]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, num_nodes, in_channels]
        """
        batch_size, seq_len, num_nodes, _ = x.shape
        
        # Reshape to process each node separately
        x_reshaped = x.permute(0, 2, 1, 3)  # [batch_size, num_nodes, seq_len, in_channels]
        x_reshaped = x_reshaped.reshape(-1, seq_len, self.in_channels)  # [batch_size * num_nodes, seq_len, in_channels]
        
        # Apply linear transformations
        q = self.W_q(x_reshaped)  # [batch_size * num_nodes, seq_len, in_channels]
        k = self.W_k(x_reshaped)  # [batch_size * num_nodes, seq_len, in_channels]
        v = self.W_v(x_reshaped)  # [batch_size * num_nodes, seq_len, in_channels]
        
        # Calculate attention scores
        attn = torch.matmul(q, k.permute(0, 2, 1))  # [batch_size * num_nodes, seq_len, seq_len]
        attn = attn / self.scale
        
        # Apply softmax to get attention weights
        attn = F.softmax(attn, dim=-1)
        
        # Apply causal masking (optional)
        # mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        # attn = attn * mask.unsqueeze(0)
        # attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-9)
        
        # Weighted aggregation of values
        output = torch.matmul(attn, v)  # [batch_size * num_nodes, seq_len, in_channels]
        output = self.W_out(output)  # Linear projection
        
        # Residual connection and layer normalization
        output = self.layer_norm(x_reshaped + output)
        
        # Reshape back
        output = output.reshape(batch_size, num_nodes, seq_len, self.in_channels)
        output = output.permute(0, 2, 1, 3)  # [batch_size, seq_len, num_nodes, in_channels]
        
        return output

class STGCNBlock(nn.Module):
    """Spatio-Temporal Graph Convolutional Block.
    
    This block combines temporal convolution with spatial graph convolution
    in a sandwich-like structure: [Temporal Conv] - [Spatial GCN] - [Temporal Conv]
    
    Reference: Yu et al. "Spatio-Temporal Graph Convolutional Networks: 
    A Deep Learning Framework for Traffic Forecasting", IJCAI 2018.
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 spatial_kernel_size: int, temporal_kernel_size: int,
                 num_nodes: int, dropout: float = 0.3,
                 use_spatial_attention: bool = True,
                 use_temporal_attention: bool = True):
        """Initialize the STGCN Block.
        
        Args:
            in_channels: Number of input features
            out_channels: Number of output features
            spatial_kernel_size: Size of the spatial convolution kernel (Chebyshev order)
            temporal_kernel_size: Size of the temporal convolution kernel
            num_nodes: Number of nodes in the graph
            dropout: Dropout rate
            use_spatial_attention: Whether to use spatial attention
            use_temporal_attention: Whether to use temporal attention
        """
        super(STGCNBlock, self).__init__()
        self.use_spatial_attention = use_spatial_attention
        self.use_temporal_attention = use_temporal_attention
        
        # First temporal convolution: input -> hidden
        self.temp_conv1 = TemporalConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=temporal_kernel_size,
            dropout=dropout
        )
        
        # Spatial graph convolution
        self.graph_conv = ChebGraphConv(
            in_channels=out_channels,
            out_channels=out_channels,
            K=spatial_kernel_size
        )
        
        # Second temporal convolution: hidden -> output
        self.temp_conv2 = TemporalConvLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=temporal_kernel_size,
            dropout=dropout
        )
        
        # Optional attention layers
        if use_spatial_attention:
            self.spatial_attention = SpatialAttentionLayer(
                in_channels=out_channels,
                num_nodes=num_nodes
            )
            
        if use_temporal_attention:
            # Note: We use a fixed sequence length here
            # This will be dynamically adjusted during forward pass
            self.temporal_attention = TemporalAttentionLayer(
                in_channels=out_channels,
                seq_len=100  # Placeholder value, will be overridden
            )
            
        # Output layer normalization and dropout
        self.layer_norm = nn.LayerNorm([out_channels])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, cheb_polynomials: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass through the STGCN block.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, num_nodes, in_channels]
            cheb_polynomials: List of Chebyshev polynomials of the graph Laplacian
            
        Returns:
            Output tensor of shape [batch_size, seq_len, num_nodes, out_channels]
        """
        # First temporal convolution
        out = self.temp_conv1(x)
        # Spatial graph convolution
        out = self.graph_conv(out, cheb_polynomials)
        # Apply spatial attention if enabled
        if self.use_spatial_attention:
            out = self.spatial_attention(out)
            
        # Second temporal convolution
        out = self.temp_conv2(out)
            
        # Apply temporal attention if enabled
        if self.use_temporal_attention:
            # Create a new temporal attention layer with correct sequence length
            _, seq_len, _, _ = out.shape
            temporal_attention = TemporalAttentionLayer(
                in_channels=out.shape[-1],
                seq_len=seq_len
            ).to(out.device)
            out = temporal_attention(out)
            
        # Apply final normalization
        batch_size, seq_len, num_nodes, _ = out.shape
        out = out.reshape(-1, out.shape[-1])
        out = self.layer_norm(out)
        out = self.dropout(out)
        out = out.reshape(batch_size, seq_len, num_nodes, -1)
        
        return out
