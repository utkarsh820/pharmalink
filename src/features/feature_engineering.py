import pandas as pd
import numpy as np

def build_features(df):
    # Ensure datetime
    df['date'] = pd.to_datetime(df['date'])

    # =========================
    # 1. Time-based features
    # =========================
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # =========================
    # 2. Sort for time-series ops
    # =========================
    df = df.sort_values(by=['pharmacy_id', 'drug', 'date'])

    # =========================
    # 3. Lag features
    # =========================
    df['demand_lag_1'] = df.groupby(['pharmacy_id', 'drug'])['demand'].shift(1)
    df['demand_lag_7'] = df.groupby(['pharmacy_id', 'drug'])['demand'].shift(7)

    # =========================
    # 4. Rolling statistics
    # =========================
    df['rolling_mean_7'] = df.groupby(['pharmacy_id', 'drug'])['demand'].transform(lambda x: x.shift(1).rolling(7).mean())
    df['rolling_std_7'] = df.groupby(['pharmacy_id', 'drug'])['demand'].transform(lambda x: x.shift(1).rolling(7).std())

    # =========================
    # 5. Stock-demand features
    # =========================
    df['stock_gap'] = df['stock'] - df['demand']
    df['stock_ratio'] = df['stock'] / (df['demand'] + 1)
    df['stockout_flag'] = (df['stock'] < df['demand']).astype(int)

    # =========================
    # 6. Lead time features
    # =========================
    df['demand_per_lead_time'] = df['demand'] / (df['lead_time_days'] + 1)
    df['urgency_score'] = (df['demand'] / (df['stock'] + 1)) * df['lead_time_days']

    # =========================
    # 7. Trend features
    # =========================
    df = df.sort_values(by=['pharmacy_id', 'drug', 'date'])

    # 2. Group by the entities and shift the demand by 1 month to create the lag
    df['demand_lag_1'] = df.groupby(['pharmacy_id', 'drug'])['demand'].shift(1)


    df['demand_diff_1'] = df['demand'] - df['demand_lag_1']
    df['demand_growth_rate'] = df['demand_diff_1'] / (df['demand_lag_1'] + 1)
    # =========================
    # 8. Cost-based features
    # =========================
    df['revenue_estimate'] = df['demand'] * df['cost']
    df['inventory_value'] = df['stock'] * df['cost']

    # =========================
    # 9. City encoding (one-hot)
    # =========================
    df = pd.get_dummies(df, columns=['city'], drop_first=True)

    # =========================
    # 10. Pharmacy-level aggregations
    # =========================
    pharmacy_stats = df.groupby('pharmacy_id')['demand'].agg(['mean', 'std']).reset_index()
    pharmacy_stats.columns = ['pharmacy_id', 'pharmacy_avg_demand', 'pharmacy_demand_std']

    df = df.merge(pharmacy_stats, on='pharmacy_id', how='left')

    # =========================
    # 11. Interaction features
    # =========================
    df['lead_time_stock_gap'] = df['lead_time_days'] * df['stock_gap']

    # =========================
    # 12. Risk flags
    # =========================
    df['high_risk_flag'] = ((df['stock'] < df['demand']) & (df['lead_time_days'] > 3)).astype(int)
    df['overstock_flag'] = (df['stock'] > df['demand'] * 1.5).astype(int)

    # =========================
    # 13. Handle missing values (from lag/rolling)
    # =========================
    df = df.fillna(0)
    return df

def main():
    df = pd.read_parquet("data/processed/final_pharmalink_dataset.parquet")
    print("loading successfull")
    df = build_features(df)
    print("build sucessful")
    df.to_parquet("data/processed/pharma_features.parquet", index=False)
    print(f"saved to: data/processed/pharma_features.parquet")
    return print(df.head())

if __name__ == "__main__":
    main()



