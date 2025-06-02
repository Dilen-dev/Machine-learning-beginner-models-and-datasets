# Salary Prediction Using Linear Regression

This project demonstrates a simple machine learning model that predicts salaries based on years of experience using Linear Regression.

## 📄 File Overview

- `Salaries_prediction_model.py`: A standalone Python script that:
  - Loads a dataset (CSV format) with experience and salary data.
  - Splits data into training and testing sets.
  - Trains a Linear Regression model using scikit-learn.
  - Makes predictions on test data.
  - Visualizes both training and test results with matplotlib.

## 📊 Dataset Format

The script expects a CSV file named `Salary_Data.csv` with the following columns:

| YearsExperience | Salary     |
|-----------------|------------|
| 1.1             | 39343.00   |
| 2.0             | 46205.00   |
| ...             | ...        |

Make sure the CSV file is in the same directory as the script or adjust the file path.

## 🚀 How to Run

1. Install required packages (if not already installed):
   ```bash
   pip install pandas numpy matplotlib scikit-learn
