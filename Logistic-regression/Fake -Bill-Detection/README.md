# Fake Bill Detection Using Logistic Regression

This project builds a logistic regression model to classify whether a bill is genuine or fake based on various visual features.

## 📄 Script Overview

- `Deciding_genuine_bills.py`: A Python script that:
  - Loads and cleans a dataset of bill characteristics
  - Handles missing values
  - Scales features
  - Trains a logistic regression model
  - Evaluates performance with accuracy, confusion matrix, and classification report

## 📊 Dataset

The script uses a CSV file named fake_bills.csv with features such as:

| is_genuine | margin_low | margin_up | length | left_margin | ... |
|------------|-------------|-----------|--------|-------------|-----|

The target variable is is_genuine (1 = real, 0 = fake).

Update the dataset path as needed:
```python
dataset = pd.read_csv("path/to/fake_bills.csv")
