# 🏎️ F1 Chinese GP 2025 Winner Prediction Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![FastF1](https://img.shields.io/badge/FastF1-3.0+-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

Predict the winner of the 2025 Chinese Formula 1 Grand Prix using historical race data and machine learning techniques. This project demonstrates how sports analytics can be applied to motorsports.

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Methodology](#-methodology)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)

## 🌟 Project Overview

This machine learning project predicts the winner of the 2025 Chinese Grand Prix by:
1. Fetching historical 2024 Chinese GP race data using `fastf1`
2. Combining with hypothetical 2025 qualifying times
3. Training a Gradient Boosting Regressor model
4. Predicting race lap times based on qualifying performance
5. Ranking drivers by predicted lap time to determine the winner

## ✨ Features

- **Data Collection**: Automatically fetches real F1 race data
- **Machine Learning**: Uses Gradient Boosting algorithm for accurate predictions
- **Visualization**: Generates bar charts of predicted results
- **Customizable**: Easy to modify for different races or seasons
- **Performance Metrics**: Provides MAE score for model evaluation

## 💻 Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/F1-Chinese-GP-2025-Winner-Prediction-Using-Machine-Learning.git
cd F1-Chinese-GP-2025-Winner-Prediction-Using-Machine-Learning
