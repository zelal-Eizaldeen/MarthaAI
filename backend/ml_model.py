import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

PATH_TO_DATA='../data'
# Load the data
sample_ads=pd.read_csv(f'{PATH_TO_DATA}/meta_data.csv')
# Convert into data
sample_ads['date'] = pd.to_datetime(sample_ads['date'])

# Feature engineering
sample_ads['hour'] = sample_ads['date'].dt.hour
sample_ads['dayofweek'] = sample_ads['date'].dt.dayofweek
sample_ads['campaign_encoded'] = sample_ads['campaign'].astype('category').cat.codes

features = ['ad_spend', 'impressions', 'clicks', 'ctr', 'cpc', 'hour', 'dayofweek', 'campaign_encoded']
X = sample_ads[features]
y = sample_ads['conversions']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(f"Test RMSE: {root_mean_squared_error(y_test, y_pred):.2f}")
# After model prediction
ml_results_df = pd.DataFrame({
    'Actual_Conversions': y_test,
    'Predicted_Conversions': y_pred
})
# Save to CSV
ml_results_df.to_csv(f"{PATH_TO_DATA}/rf_predictions.csv", index=False)


# Calculate RMSE
rmse = root_mean_squared_error(y_test, y_pred)

# Add RMSE row
rmse_row = pd.DataFrame({
    'Actual_Conversions': ['RMSE'],
    'Predicted_Conversions': [f"{rmse:.2f}"]
})

# Combine both
export_df = pd.concat([ml_results_df, rmse_row], ignore_index=True)

# Save to CSV
export_df.to_csv(f"{PATH_TO_DATA}/rf_predictions_with_rmse.csv", index=False)

# Insights
# Feature importance
feature_importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("🔍 Top Features Influencing Conversions:")
print(feature_importance)

feature_importance_df = pd.DataFrame({
    'Feature': feature_importance.index,
    'Importance': feature_importance.values
})

feature_importance_df.to_csv(f"{PATH_TO_DATA}/rf_feature_importance.csv", index=False)