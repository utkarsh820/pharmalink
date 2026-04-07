# Pharmalink - Pharmaceutical Demand Forecasting and Supply Chain Optimization

## Project Context
Efficient inventory management in the pharmaceutical sector is critical for ensuring patient access to medication while minimizing carrying costs. This project utilizes monthly prescription data integrated with supply chain variables to build a robust demand forecasting system.

## Business Problem
Pharmacists often face the dual challenge of stockouts for high-demand drugs and overstocking for low-turnover items. This project aims to:
* Identify underlying demand patterns and seasonal trends.
* Quantify the impact of lead times on stock availability.
* Provide data-driven insights to optimize replenishment cycles across diverse geographical locations.

## Dataset Specifications
The analysis is based on a multi-dimensional dataset containing:
* **Temporal Data**: Monthly demand and delivery lead times.
* **Geographic Data**: City and pharmacy-level identifiers.
* **Product Data**: Specific drug categories and unit costs.
* **Target Variable**: Monthly prescriptions (Demand).

## Analytical Approach
1.  **Demand Profiling**: Classifying drugs based on demand volatility (Stable vs. Intermittent).
2.  **Temporal Analysis**: Decomposing series into trend, seasonality, and residual components.
3.  **Supply-Demand Gap Analysis**: Mapping inventory levels against actual demand to identify systemic shortages.
4.  **Feature Engineering**: Developing lag variables and rolling statistics for predictive modeling.

## Key Performance Indicators
* **Forecast Accuracy**: Measured via WAPE (Weighted Absolute Percentage Error).
* **Service Level**: Percentage of demand met without stockouts.
* **Inventory Turnover**: Efficiency of stock utilization relative to demand spikes.

