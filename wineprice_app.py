import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib
import json
from mapping_dict import map_varieties

# Initialize session state for storing plot data and a counter
if 'plot_data' not in st.session_state:
    empty_bars = [("", 0, 'black', 0)] * 5  # Initialize with 5 empty entries
    st.session_state['plot_data'] = empty_bars
    st.session_state['bar_counter'] = 0
    st.session_state['total_inputs'] = 0

# Load model, RMSE, and country-region mapping
model_pipeline = joblib.load('model.pkl')
with open('model_rmse.txt') as f:
    rmse = float(f.read())
with open('country_region_mapping.json') as f:
    country_region_dict = json.load(f)

# Get unique grape varieties from the map_varieties function
unique_varieties = list(map_varieties())

# Function to make price prediction
def model_prediction(country, region, score, grape_variety):
    wine_type = map_varieties(grape_variety)
    input_data = pd.DataFrame([[score, country, region, wine_type]], columns=['points', 'country', 'region_1', 'wine_type'])
    price_prediction = model_pipeline.predict(input_data)
    return price_prediction[0], wine_type

# Function to create bar plot
def create_bar_plot(data):
    plt.figure(figsize=(14, 8))
    for idx, (label, prediction, color, rmse) in enumerate(data):
        prediction = max(0, prediction)
        plt.bar(idx, prediction, color=color, capsize=5)
        label_y = max(3, prediction + 2)
        plt.text(idx, label_y, f"${prediction:.1f}", ha='center', fontsize=18, fontweight='bold')
    
    plt.xticks(range(len(data)), [item[0] for item in data], rotation=75, fontsize=12)

    # Set the default y-axis range to 0 to 100
    plt.ylim(0, 100)

    plt.ylabel('USD ($)', fontsize=18)
    plt.title(f"Predicted Prices for World Wines", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=16)
    st.pyplot(plt)

# Streamlit interface
st.markdown(
    """
    <h3 style='text-align: center;'>Enter the country of origin, region, and critic score of your wine to predict the ideal market price</h1>
    """,
    unsafe_allow_html=True
)

# Country selection
selected_country = st.selectbox("Country:", sorted(country_region_dict.keys()), index=sorted(country_region_dict.keys()).index(st.session_state.get('selected_country', list(country_region_dict.keys())[0])))
st.session_state['selected_country'] = selected_country

# Region selection
selected_region = st.selectbox("Region:", country_region_dict[selected_country])

# Grape variety selection
grape_variety = st.selectbox("Grape Variety:", sorted(unique_varieties))
score = st.slider("Score:", min_value=70, max_value=100, value=90)

# Prediction and plotting
col1, col2, col3, col4, col5 = st.columns([1, 1, 1.7, 1, 1])

with col1:
    if st.button('Predict Price'):
        predicted_price, wine_type = model_prediction(selected_country, selected_region, score, grape_variety)
        predicted_price = float(predicted_price)
        wine_color = {'red': 'plum', 'white': 'palegoldenrod', 'rose': 'mistyrose'}.get(wine_type, 'black')
        label = f"{score}-pt {grape_variety.title()} from {selected_region}, {selected_country}"
        
        # Check if the maximum number of bars (5) has been reached, and add a new one
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