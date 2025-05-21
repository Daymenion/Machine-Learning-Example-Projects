# Ride Demand Forecasting System - Run Summary

**Run ID:** 20250516_143115

**Timestamp:** 2025-05-16T14:40:15.761619

## Data Summary

- Train set size: 23895
- Validation set size: 5120
- Test set size: 5121
- Number of features: 273

## Model Evaluation

| Model | RMSE | MAE | MAPE | R² |
|-------|------|-----|------|-----|
| lightgbm | 2.3831 | 1.8157 | 47.60% | 0.5715 |

**Best model:** lightgbm

## Pricing Optimization

- Mean surge multiplier: 1.00
- Maximum surge multiplier: 1.00
- Total expected revenue: $307119.50
- Revenue improvement vs. baseline: -8.71%

## Visualizations

- [lightgbm_error_distribution](visualizations\lightgbm_error_distribution.png)
- [lightgbm_feature_importance](visualizations\lightgbm_feature_importance.png)
- [lightgbm_time_series](visualizations\lightgbm_time_series.png)
- [lightgbm_weekly_hourly_forecast](visualizations\lightgbm_weekly_hourly_forecast.png)
- [surge_multiplier_heatmap](visualizations\surge_multiplier_heatmap.png)
