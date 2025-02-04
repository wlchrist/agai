import pandas as pd

# Load SWE data file
swe_file = "/scratch/project/hackathon/data/SnowpackPredictionChallenge/input_data/swe_data/SWE_values_all.csv"
df_swe = pd.read_csv(swe_file)  # Adjust row count as needed

# Load station info file
station_file = "/scratch/project/hackathon/data/SnowpackPredictionChallenge/input_data/swe_data/Station_Info.csv"
df_stations = pd.read_csv(station_file)  # Adjust row count as needed

# Ensure column names match (adjust if necessary)
swe_features = ['Date', 'SWE', 'Latitude', 'Longitude']
station_features = ['Station', 'Latitude', 'Longitude']

# Select relevant columns
df_swe = df_swe[swe_features]
df_stations = df_stations[station_features]

# Merge on Latitude (and Longitude if needed)
df_merged = df_swe.merge(df_stations, on=['Latitude'], how='left')

# Drop rows where no matching station was found
df_merged = df_merged.dropna(subset=['Station'])

# Print the results
for _, row in df_merged.iterrows():
    print(f"{row['Date']} - {row['Station']}: SWE {row['SWE']}, Latitude {row['Latitude']}")
