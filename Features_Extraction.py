import pandas as pd
import numpy as np
import os
from scipy.stats import skew, kurtosis

# Load the dataset
# Replace 'battery_data.csv' with the path to your dataset
data = pd.read_csv(os.getcwd() + '/Datasets/HNEI_Processed/Final Database.csv')

# Ensure the dataset has a column identifying each battery
if 'Battery_ID' not in data.columns:
    raise ValueError("The dataset must contain a 'Battery_ID' column.")

# Function to extract statistical features for a group of cycles within a single battery
def extract_features_grouped(df):
    """
    Extract statistical features for every 50-cycle group within a single battery.
    
    Parameters:
        df (DataFrame): Battery data with features like cycle index, discharge time, etc.
        
    Returns:
        grouped_features_df (DataFrame): DataFrame with statistical features for each group.
    """
    # List of columns to extract features from
    feature_columns = [
        'Discharge Time (s)', 
        'Decrement 3.6-3.4V (s)', 
        'Max. Voltage Dischar. (V)', 
        'Min. Voltage Charg. (V)', 
        'Time at 4.15V (s)', 
        'Time constant current (s)', 
        'Charging time (s)', 
        'Total time (s)'
    ]
    
    # Create a new column for grouping (every 50 cycles)
    df['Cycle_Group'] = df['Cycle_Index'] // 50
    
    # Initialize a list to store feature dictionaries for each group
    feature_list = []
    
    # Group by 'Cycle_Group' and calculate features
    grouped = df.groupby('Cycle_Group')
    for group, group_data in grouped:
        stats = {'Cycle_Group': group}  # Store the group identifier
        for col in feature_columns:
            if col in group_data.columns:
                #stats[f'{col}_mean'] = group_data[col].mean()
                #stats[f'{col}_std'] = group_data[col].std()
                #stats[f'{col}_min'] = group_data[col].min()
                #stats[f'{col}_max'] = group_data[col].max()
                stats[f'{col}_skew'] = skew(group_data[col].dropna())
                stats[f'{col}_kurtosis'] = kurtosis(group_data[col].dropna())
        # Add RUL if available
        if 'RUL' in group_data.columns:
            stats['RUL'] = group_data['RUL'].iloc[-1]  # Take the last RUL in the group
        feature_list.append(stats)
    
    # Convert the list of feature dictionaries to a DataFrame
    grouped_features_df = pd.DataFrame(feature_list)
    return grouped_features_df

# Process each battery separately
battery_features = []

# Group the data by Battery_ID
batteries = data['Battery_ID'].unique()
for battery_id in batteries:
    print(f"Processing features for Battery_ID: {battery_id}")
    battery_data = data[data['Battery_ID'] == battery_id]
    battery_features_df = extract_features_grouped(battery_data)
    battery_features_df['Battery_ID'] = battery_id  # Add battery identifier to the features
    battery_features.append(battery_features_df)

# Combine all battery features into a single DataFrame
all_features_df = pd.concat(battery_features, ignore_index=True)

# Save the extracted features to a CSV file
output_csv_path = 'battery_features_two.csv'
all_features_df.to_csv(output_csv_path, index=False)

print(f"Features for all batteries extracted and saved to '{output_csv_path}'")
