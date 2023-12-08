import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib
import json
from scipy.stats import norm
from scipy.stats import linregress
from mapping_dict import map_varieties

# Load dataset
df = pd.read_csv('winemag_data.csv')

# Preprocessing
df = df[(df['price'] <= 75) & (df['price'] > 25)]
df['country'].fillna('Unknown', inplace=True)
df['region_1'].fillna('Unknown', inplace=True)
df['variety'].fillna('Unknown', inplace=True)
df['wine_type'] = map_varieties(df['variety'])  # Ensure this function is defined correctly

# Perform linear regression between points and price
slope, intercept, r_value, p_value, std_err = linregress(df['points'], df['price'])

# Save the slope and intercept to a file
with open('score_trend_params.txt', 'w') as f:
    f.write(f"{slope}\n{intercept}\n")

# Using the original 'points' feature
features = df[['country', 'region_1', 'wine_type', 'variety']]
target = df['price']

# One-hot encoding, including 'variety'
column_transformer = ColumnTransformer(
    [("ohe", OneHotEncoder(handle_unknown='ignore'), ['country', 'region_1', 'wine_type', 'variety'])],
    remainder='passthrough'
)

# Creating a dictionary of countries to regions
country_region_dict = df.groupby('country')['region_1'].unique().apply(list).to_dict()

# Save the dictionary as a JSON file
with open('country_region_mapping.json', 'w') as f:
    json.dump(country_region_dict, f)

# Model creation
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
model_pipeline = make_pipeline(column_transformer, RandomForestRegressor(max_features=100))

model_pipeline.fit(X_train, y_train)

# Calculate the mean price for each region
mean_prices_by_region = df.groupby('region_1')['price'].mean()

# Calculate RMSE
y_pred = model_pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Save the model and RMSE to files
joblib.dump(model_pipeline, 'model.pkl')
with open('model_rmse.txt', 'w') as f:
    f.write(str(rmse))

# Calculate the difference in price for each point
df['price_per_point'] = df['price'] / df['points']
average_price_per_point = df['price_per_point'].mean()

# Save the average_price_per_point to a file
with open('price_per_point.txt', 'w') as f:
    f.write(str(average_price_per_point))

# # Create outlier lists
# outlier_regions = mean_prices_by_region[mean_prices_by_region > 50].index.tolist()

# # Save the outlier lists to files
# with open('outlier_regions.txt', 'w') as file:
#     for region in outlier_regions:
#         file.write(region + '\n')