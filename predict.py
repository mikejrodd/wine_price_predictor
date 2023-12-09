import pandas as pd
import os
import joblib

def model_prediction(country, province, region, score, grape_variety, wine_type, outlier_regions, average_price_per_point, model_pipeline, intercept, slope):
    # Adjust input data to include province
    input_data = pd.DataFrame([[country, province, region, wine_type, grape_variety]], columns=['country', 'province', 'region_1', 'wine_type', 'variety'])
    baseline_prediction = model_pipeline.predict(input_data)[0]

    # Standard score-based adjustment
    reference_score = 89
    score_adjustment = (score - reference_score) * average_price_per_point
    adjusted_prediction = baseline_prediction + score_adjustment

    # Check for outlier region and apply a premium if applicable
    outlier_premium_factor = 1.5
    if region in outlier_regions:
        adjusted_prediction *= outlier_premium_factor

    # Additional adjustments for specific score ranges
    if score < 80:
        penalty_factor = 0.85
        adjusted_prediction *= penalty_factor
    elif 79 < score < 84:
        penalty_factor = 0.95
        adjusted_prediction *= penalty_factor
    elif 83 < score < 87:
        penalty_factor = 1
        adjusted_prediction *= penalty_factor
    elif 96 < score < 100:
        penalty_factor = 1.2
        adjusted_prediction *= penalty_factor
    elif score > 99:
        reward_factor = 1.3
        adjusted_prediction *= reward_factor

    # Penalty if country and region are the same
    if country == region:
        same_place_penalty_factor = 0.95
        adjusted_prediction *= same_place_penalty_factor

    # Ensure the prediction is not negative
    adjusted_prediction = max(adjusted_prediction, 0)

    return adjusted_prediction
