# Ensure necessary libraries are installed
try:
    import pandas as pd
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    import numpy as np
except ModuleNotFoundError as e:
    print(f"Missing dependency: {e.name}. Please install it using 'pip install {e.name}'")
    exit(1)  # Stop execution if dependencies are missing

# Load CSV file
df = pd.read_csv('/content/1b.csv')

def preprocess_data(df):
    """Preprocess the dataset for anomaly detection."""
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Days_Since_First'] = (df['Date'] - df['Date'].min()).dt.days
    feature_columns = ['Amount', 'Balance', 'Days_Since_First']
    df_features = df[feature_columns].dropna()
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
    df.loc[data.index, 'is_forged'] = (predictions == -1).astype(int)
    return df

# Run the model
df, df_features = preprocess_data(df)
model, scaler = train_anomaly_detection_model(df_features)
results = detect_forged_transactions(df, df_features, model, scaler)

# Save results
output_file = "detected_forged_transactions.csv"
results.to_csv(output_file, index=False)
print(f"Detection complete. {results['is_forged'].sum()} forged transactions detected.")
print(f"Results saved in '{output_file}'")
