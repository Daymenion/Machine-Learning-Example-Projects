# Ride Demand Forecasting Project

## Overview

This project implements an advanced demand forecasting system for ride-sharing services. It predicts hourly passenger demand across multiple locations, enabling optimized vehicle allocation and surge pricing strategies. The system employs machine learning models trained on historical ride data, weather conditions, temporal patterns, and spatial features to make accurate predictions.

## Solution Architecture

The solution follows a modular data science pipeline architecture:

1. **Data Processing**: Enriches raw trip data with temporal, spatial, and external features
2. **Feature Engineering**: Creates advanced features including time-based aggregations, weather impact factors, and spatial relationships
3. **Model Training**: Implements gradient boosting decision trees model, Temporal Fusion Transformer and Spatio-Temporal Graph Convolutional Network for demand prediction. (Lack of time, there are some bugs in TFT and STGCN that need to be solved, so the results are not available in the results section.)
4. **Evaluation**: Comprehensive metrics and visualizations to analyze model performance
5. **Pricing Optimization**: Uses forecasts to recommend optimal surge pricing strategies

```
Raw Data → Preprocessor → Feature Engineer → Model Training → Evaluation → Pricing Optimization
```

## Feature Engineering Approach

Our feature engineering process creates a rich set of predictive signals:

- **Temporal Features**: Hour of day, day of week, month, holidays, seasonality indicators
- **Spatial Features**: Location clusters, distance bands, area demand profiles
- **Weather Impact**: Temperature, precipitation, humidity, and their interactions with demand
- **Aggregation Features**: Rolling windows, exponential weighted means for historical patterns
- **Interaction Features**: Cross-features capturing complex relationships between variables

We implemented careful feature selection to avoid data leakage, particularly by excluding features that could introduce future information (like exponential weighted means with specific alpha values).

## Model Architectures

We developed three advanced model architectures for demand forecasting, each with unique strengths:

### 1. LightGBM Model

The primary forecasting engine uses **LightGBM**, a gradient boosting framework optimized for performance and accuracy:

- **Model Type**: Gradient Boosting Decision Trees (GBDT)
- **Key Parameters**:
  - `max_depth`: 12 (controls tree depth)
  - `num_leaves`: 4096 (controls model complexity)
  - `learning_rate`: 0.01 (controls training stability)
  - `n_estimators`: 1000 (number of boosting iterations)
  - Regularization: L1/L2 regularization to prevent overfitting

The LightGBM model was trained using careful cross-validation and hyperparameter optimization to maximize performance while maintaining generalization.

### 2. Optimized Spatial-Temporal Graph Convolutional Network (STGCN)

We implemented an advanced **STGCN** model to explicitly capture spatial and temporal relationships in the demand data:

- **Architecture**: Multi-layer network with specialized components:

  - **Spatial Module**: Graph convolutional layers with Chebyshev polynomial approximation to model complex spatial dependencies between locations
  - **Temporal Module**: 1D convolutional layers to capture temporal patterns across various time scales
  - **Spatial-Temporal Attention**: Custom attention mechanisms that focus on the most relevant spatial and temporal signals
  - **Dynamic Graph Construction**: Adaptive adjacency matrices that evolve based on observed demand patterns
- **Key Innovations**:

  - **Spatial Attention Layer**: Learns to weight connections between locations based on their demand correlation patterns
  - **Temporal Attention Layer**: Focuses on the most relevant historical time points for prediction
  - **Optimized Graph Builder**: Constructs the spatial graph dynamically based on multiple similarity metrics
  - **Skip Connections**: Enables better gradient flow through this complex architecture

The STGCN model excels at capturing complex non-linear relationships between locations and shows superior performance for areas with high spatial dependency.

### 3. Temporal Fusion Transformer (TFT)

We also developed a **Temporal Fusion Transformer** model, a state-of-the-art architecture for multi-horizon time series forecasting:

- **Architecture Components**:

  - **Variable Selection Networks**: Automatically select the most relevant features for each prediction step
  - **Gated Residual Networks**: Control information flow through the network
  - **Temporal Self-Attention Layers**: Capture long-range dependencies in time series data
  - **Multi-Head Attention**: Allows the model to focus on different aspects of the input data
  - **Interpretable Attention Weights**: Provides insights into which features and time periods most influence predictions
- **Key Innovations**:

  - **Static-Dynamic Integration**: Effectively combines time-invariant features with temporal patterns
  - **Quantile Forecasting**: Produces prediction intervals to quantify uncertainty
  - **Categorical Embedding**: Efficiently handles categorical variables like location IDs
  - **Future-Known Features**: Special handling for variables like holidays known in advance

The TFT model provides both accurate point forecasts and reliable uncertainty estimates, making it valuable for understanding prediction confidence.

## Results and Performance

The model demonstrates strong predictive performance across different locations and time periods:

- **Overall RMSE**: Low error rates in predicting demand volume
- **Temporal Performance**: Consistent accuracy across different hours of the day and days of the week
- **Spatial Performance**: Reliable predictions across different location clusters
- **Feature Importance**: Weather conditions, time of day, and recent historical demand emerged as strong predictors

The weekly hourly forecast visualization provides clear insights into demand patterns, highlighting peak times and seasonal variations that inform business strategy.

## Key Findings

1. **Data Leakage Prevention**: We identified and excluded potential data leakage from exponentially weighted mean features that may incorporate future information
2. **Temporal Patterns**: Demand shows distinct patterns by hour of day and day of week that are highly predictable
3. **Weather Impact**: Weather conditions significantly influence demand, with specific interaction effects during peak hours
4. **Spatial Clustering**: Grouping locations into clusters improved prediction accuracy for areas with similar demand patterns

## Future Work

1. **Advanced Models**: Explore spatial-temporal graph convolutional networks (STGCN) for capturing complex spatial-temporal dynamics
2. **Feature Evolution**: Implement dynamic feature selection to adapt to changing patterns over time
3. **Uncertainty Quantification**: Enhance prediction intervals to better capture demand volatility
4. **Real-time Processing**: Develop streaming data processing for near real-time demand forecasting
5. **Multi-objective Optimization**: Balance multiple business objectives in pricing strategies

## Usage

To run the full pipeline:

```bash
python runner.py run-all
```

To run specific components:

```bash
python runner.py train-models  # Train demand forecasting models
python runner.py evaluate-models  # Evaluate model performance
python runner.py optimize-pricing  # Generate pricing recommendations
```

## Dependencies

- requirements.txt

## Configuration

The system is configured through `config.yaml`, which controls data paths, feature engineering settings, model parameters, and evaluation metrics.
