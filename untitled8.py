# -*- coding: utf-8 -*-
"""Untitled8.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1BK49fO2gW_mnp7JzGbtbloEF1N9VQ09R
"""

!pip install scikit-learn


import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Assuming your data is in a CSV file named 'transactions.csv'
df = pd.read_csv('/content/1b.csv') # Load your data into a pandas DataFrame named 'df'

def preprocess_data(df):
    """Preprocess the dataset for anomaly detection."""
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Convert date to datetime format
    df['Days_Since_First'] = (df['Date'] - df['Date'].min()).dt.days  # Convert date to numerical value

    # Selecting features for the model
    feature_columns = ['Amount', 'Balance', 'Days_Since_First']
    df_features = df[feature_columns].dropna()  # Drop NaN values

    return df, df_features

def train_anomaly_detection_model(data):
    """Train an optimized Isolation Forest model."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    model = IsolationForest(n_estimators=200, contamination=0.1, random_state=42, bootstrap=True, n_jobs=-1)
    model.fit(X_scaled)

    return model, scaler

def detect_forged_transactions(df, data, model, scaler):
    """Detect forged transactions and add the 'is_forged' label."""
    X_scaled = scaler.transform(data)
    predictions = model.predict(X_scaled)

    # Isolation Forest labels anomalies as -1
    df.loc[data.index, 'is_forged'] = (predictions == -1).astype(int)

    return df

# Preprocess data
df, df_features = preprocess_data(df)

# Train model
model, scaler = train_anomaly_detection_model(df_features)

# Detect forged transactions
results = detect_forged_transactions(df, df_features, model, scaler)

# Save results to CSV in the current directory
output_file = "detected_forged_transactions.csv"
results.to_csv(output_file, index=False)

print(f"Detection complete. {results['is_forged'].sum()} forged transactions detected.")
print(f"Results saved in '{output_file}'")
