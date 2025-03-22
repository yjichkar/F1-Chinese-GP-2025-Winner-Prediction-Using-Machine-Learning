# F1-Chinese-GP-2025-Winner-Prediction-Using-Machine-Learning
Predict the winner of the 2025 Chinese F1 Grand Prix using historical race data and machine learning! This project leverages the fastf1 library to fetch 2024 Chinese GP race data, combines it with hypothetical 2025 qualifying times, and applies a Gradient Boosting Regressor to predict race lap times. The driver with the fastest predicted lap time is declared the winner. Built with Python, Pandas, and Scikit-learn, this project is ideal for F1 enthusiasts and data scientists looking to explore sports analytics. Contributions are welcome to enhance features, improve prediction accuracy, or extend to other races!

Breakdown of the Description
Opening Hook: "Predict the winner of the 2025 Chinese F1 Grand Prix using historical race data and machine learning!" grabs attention by stating the project’s purpose in an engaging way.
Methodology Overview: Explains the use of fastf1 for data fetching, the combination of 2024 race data with 2025 qualifying times, and the use of a Gradient Boosting Regressor to predict lap times.
Prediction Logic: Clarifies that the winner is determined by the fastest predicted lap time, giving a clear idea of the approach.
Tools Used: Mentions Python, Pandas, and Scikit-learn, which helps users understand the tech stack.
Target Audience: Appeals to "F1 enthusiasts and data scientists," broadening its appeal.
Call to Action: Invites contributions to improve the project or extend it to other races, encouraging collaboration.
# F1 Chinese GP 2025 Winner Prediction Using Machine Learning

Predict the winner of the 2025 Chinese F1 Grand Prix using historical race data and machine learning! This project leverages the `fastf1` library to fetch 2024 Chinese GP race data, combines it with hypothetical 2025 qualifying times, and applies a Gradient Boosting Regressor to predict race lap times. The driver with the fastest predicted lap time is declared the winner. Built with Python, Pandas, and Scikit-learn, this project is ideal for F1 enthusiasts and data scientists looking to explore sports analytics. Contributions are welcome to enhance features, improve prediction accuracy, or extend to other races!

pip install fastf1
# prompt: genrate a code for china 2025 qualiflying top 10 drivers
import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
# Load FastF1 2024 Chinese GP race session
session_2024 = fastf1.get_session(2024, 5, "R")  # Chinese GP is typically round 5
session_2024.load()

# Extract lap times
laps_2024 = session_2024.laps[["Driver", "LapTime"]].copy()
laps_2024.dropna(subset=["LapTime"], inplace=True)
laps_2024["LapTime (s)"] = laps_2024["LapTime"].dt.total_seconds()

# 2025 Qualifying Data for Chinese GP (replace with your actual data)
qualifying_2025 = pd.DataFrame({
    "Driver": ["Lando Norris", "Oscar Piastri", "Max Verstappen", "George Russell", "Yuki Tsunoda",
               "Alexander Albon", "Charles Leclerc", "Lewis Hamilton", "Pierre Gasly", "Carlos Sainz Jr."],
    "QualifyingTime (s)": [1.20, 1.22, 1.18, 1.21, 1.23,
                           1.24, 1.20, 1.22, 1.23, 1.21]
})

# Map full names to FastF1 3-letter codes for Chinese GP
driver_mapping = {
    "Lando Norris": "NOR", "Oscar Piastri": "PIA", "Max Verstappen": "VER", "George Russell": "RUS",
    "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB", "Charles Leclerc": "LEC", "Lewis Hamilton": "HAM",
    "Pierre Gasly": "GAS", "Carlos Sainz Jr.": "SAI"
}
qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)

# Merge 2025 Qualifying Data with 2024 Race Data
merged_data = qualifying_2025.merge(laps_2024, left_on="DriverCode", right_on="Driver")

# Use only "QualifyingTime (s)" as a feature
X = merged_data[["QualifyingTime (s)"]]
y = merged_data["LapTime (s)"]

# Check if X is empty
if X.shape[0] == 0:
    raise ValueError("Dataset is empty after preprocessing. Check data sources!")

# Train Gradient Boosting Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=39)
model.fit(X_train, y_train)

# Predict using 2025 qualifying times

predicted_lap_times = model.predict(qualifying_2025[["QualifyingTime (s)"]])
qualifying_2025["PredictedRaceTime (s)"] = predicted_lap_times

# Rank drivers by predicted race time
qualifying_2025 = qualifying_2025.sort_values(by="PredictedRaceTime (s)")

# Print final prediction
print("\n=== Predicted 2025 Chinese GP Top 10 ===\n")
print(qualifying_2025[["Driver", "PredictedRaceTime (s)"]].head(10))

# Evaluate Model
y_pred = model.predict(X_test)
print(f"Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")
