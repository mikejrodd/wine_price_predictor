import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from mapping_dict import map_varieties
from predict import model_prediction
import os
import json


# Construct the full path to the file
file_path_rmse = os.path.join(os.path.dirname(__file__), 'model_rmse.txt')
file_path_country_region = os.path.join(os.path.dirname(__file__), 'country_region_mapping.json')
file_path_slope_intercept = os.path.join(os.path.dirname(__file__), 'score_trend_params.txt')
file_path_outlier_regions = os.path.join(os.path.dirname(__file__), 'outlier_regions.txt')
file_path_price_per_point = os.path.join(os.path.dirname(__file__), 'price_per_point.txt')

# Load average price per point for score trend adjustment
with open(file_path_price_per_point, 'r') as f:
    average_price_per_point = float(f.read())
    
# Open the file for RMSE
with open(file_path_rmse) as f:
    rmse = float(f.read())

# Open the file for outlier regions
with open(file_path_outlier_regions) as f:
    outlier_regions = f.read()

# Load slope and intercept for score trend adjustment
with open(file_path_slope_intercept, 'r') as f:
    lines = f.readlines()
    slope = float(lines[0].strip())
    intercept = float(lines[1].strip())

# Open the file for country-region mapping
with open(file_path_country_region) as f:
    country_region_dict = json.load(f)

# Load your model and other necessary components
model_pipeline = joblib.load('model.pkl')
# Ensure outlier_regions and average_price_per_point are loaded

# Load the test data
df_test = pd.read_csv('winemag-data_10MB.csv')
df_test['price'].fillna(df_test['price'].median(), inplace=True)

# Preprocessing
df_test['country'].fillna('Unknown', inplace=True)
df_test['region_1'].fillna('Unknown', inplace=True)
df_test['variety'].fillna('Unknown', inplace=True)
df_test['wine_type'] = df_test['variety'].apply(map_varieties)

# Run predictions
predictions = []
for index, row in df_test.iterrows():
    predicted_price = model_prediction(row['country'], row['region_1'], row['points'], row['variety'], row['wine_type'], outlier_regions, average_price_per_point, model_pipeline, intercept, slope)
    predictions.append(predicted_price)

# Calculate RMSE
actual_prices = df_test['price'].values  # Use the actual prices from the filtered data
rmse_test = np.sqrt(mean_squared_error(actual_prices, predictions))
print("Recalculated RMSE: ", rmse_test)