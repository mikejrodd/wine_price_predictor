import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib
import json
import os
import plotly.graph_objects as go
from mapping_dict import map_varieties

# Initialize session state for storing plot data and a counter
if 'plot_data' not in st.session_state:
    empty_bars = [("", 0, 'black', 0)] * 5  # Initialize with 5 empty entries
    st.session_state['plot_data'] = empty_bars
    st.session_state['bar_counter'] = 0
    st.session_state['total_inputs'] = 0

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

# Load model and country-region mapping
model_pipeline = joblib.load('model.pkl')

# Get unique grape varieties from the map_varieties function
unique_varieties = list(map_varieties())

def model_prediction(country, region, score, grape_variety, wine_type, outlier_regions, average_price_per_point):
    # Predict the baseline price without considering the score
    input_data = pd.DataFrame([[country, region, wine_type, grape_variety]], columns=['country', 'region_1', 'wine_type', 'variety'])
    baseline_prediction = model_pipeline.predict(input_data)[0]

    # Standard score-based adjustment
    reference_score = 91
    score_adjustment = (score - reference_score) * average_price_per_point
    adjusted_prediction = baseline_prediction + score_adjustment

    # Check for outlier region and apply a premium if applicable
    outlier_premium_factor = 1.5  # Define the premium factor for outlier regions
    if region in outlier_regions:
        adjusted_prediction *= outlier_premium_factor

    # Additional adjustments for specific score ranges
    if score < 80:
        # Apply a penalty for wines scored under 75
        penalty_factor = 0.4  # Define the penalty factor (less than 1)
        adjusted_prediction *= penalty_factor
    elif score < 84 and score > 79:
        # Apply a penalty for wines scored under 75
        penalty_factor = 0.6  # Define the penalty factor (less than 1)
        adjusted_prediction *= penalty_factor
    elif score < 87 and score > 83:
        # Apply a penalty for wines scored under 75
        penalty_factor = 0.8  # Define the penalty factor (less than 1)
        adjusted_prediction *= penalty_factor
    elif score > 99:
        # Apply a reward for wines scored over 99
        reward_factor = 1.2  # Define the reward factor (greater than 1)
        adjusted_prediction *= reward_factor

    # Penalty if country and region are the same
    if country == region:
        same_place_penalty_factor = 0.7  # Define the penalty factor for same country and region
        adjusted_prediction *= same_place_penalty_factor

    # Ensure the prediction is not negative
    adjusted_prediction = max(adjusted_prediction, 0)

    return adjusted_prediction

# Function to create bar plot using Plotly
def create_bar_plot(data):
    # Extract data for plotting
    labels = [item[0] for item in data]
    predictions = [max(0, item[1]) for item in data]
    colors = [item[2] for item in data]

    # Insert a line break after the second parenthesis in each label
    split_labels = [label.replace(") ", ")<br>", 1) for label in labels]

    # Set text for each bar based on whether the label is empty or not
    text = [f"${p:.1f}" if label else '' for label, p in zip(split_labels, predictions)]

    # Create a Plotly bar chart
    fig = go.Figure(data=[go.Bar(
        x=split_labels,  # Use the split labels with line break for x-axis
        y=predictions,
        text=text,
        textposition='outside',  # Position of the text
        marker_color=colors
    )])

    # Update layout
    fig.update_layout(
        title="Predicted Prices for World Wines",
        title_x=0.37, 
        yaxis=dict(title='USD ($)'),
        yaxis_range=[0,120],
        showlegend=False
    )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

# Streamlit interface
st.markdown(
    """
    <h3 style='text-align: center;'>Enter the country of origin, region, and critic score of your wine to predict the ideal market price</h1>
    """,
    unsafe_allow_html=True
)

# Country selection
default_country_index = sorted(country_region_dict.keys()).index('US') if 'US' in country_region_dict else 0
selected_country = st.selectbox("Country:", sorted(country_region_dict.keys()), index=default_country_index)
st.session_state['selected_country'] = selected_country

# Region selection
default_region_index = country_region_dict[selected_country].index('Lake Michigan Shore') if 'Lake Michigan Shore' in country_region_dict[selected_country] else 0
selected_region = st.selectbox("Region:", country_region_dict[selected_country], index=default_region_index)

# Grape variety selection
default_variety_index = sorted(unique_varieties).index('Riesling') if 'Riesling' in unique_varieties else 0
grape_variety = st.selectbox("Grape Variety:", sorted(unique_varieties), index=default_variety_index)
score = st.slider("Score:", min_value=75, max_value=100, value=90)

# Prediction and plotting
col1, col2, col3, col4, col5 = st.columns([1, 1, 1.7, 1, 1])

with col1:
    if st.button('Predict Price'):
        wine_type = map_varieties(grape_variety)  # Map grape variety to wine type
        predicted_price = model_prediction(selected_country, selected_region, score, grape_variety, wine_type, outlier_regions, average_price_per_point)
        predicted_price = float(predicted_price)
        wine_color = {'red': 'plum', 'white': 'palegoldenrod', 'rose': 'mistyrose'}.get(wine_type, 'black')
        label = f"{grape_variety.title()} ({score}-pts) {selected_region}, {selected_country}"

        # Check if this wine is already in the plot_data
        existing_labels = [item[0] for item in st.session_state['plot_data']]
        if label in existing_labels:
            st.warning("This wine is already priced below.")
        else:
            if st.session_state['total_inputs'] < 5:
                st.session_state['plot_data'][st.session_state['bar_counter']] = (label, predicted_price, wine_color, rmse)
                st.session_state['bar_counter'] += 1
            else:
                st.session_state['plot_data'].append((label, predicted_price, wine_color, rmse))
            
            st.session_state['total_inputs'] += 1

with col5:
    if st.button('Reset Graph', key='reset'):
        empty_bars = [("", 0, 'black', 0)] * 5  # Reset to 5 empty bars
        st.session_state['plot_data'] = empty_bars
        st.session_state['bar_counter'] = 0
        st.session_state['total_inputs'] = 0

# Always draw the plot
if st.session_state['plot_data']:
    create_bar_plot(st.session_state['plot_data'])