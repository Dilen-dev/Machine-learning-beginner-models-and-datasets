
# Medical Insurance Bill Prediction

This project uses linear regression to predict individual medical insurance charges based on demographic and lifestyle factors.

## 📄 Script Overview

- `Insurance_bill_prediction.py`: A Python script that:
  - Loads and preprocesses the dataset (including encoding categorical features)
  - Trains a multiple linear regression model
  - Evaluates the model performance with R² score
  - Visualizes prediction results with a scatter plot

## 📊 Dataset

The script uses a CSV file named insurance.csv, expected to have columns like:

| age | sex | bmi | children | smoker | region | charges |
|-----|-----|-----|----------|--------|--------|---------|

Replace the file path if needed:
```python
insurance = pd.read_csv("your/path/to/insurance.csv")
````

## 🔍 Features Used

* Age
* Sex
* BMI
* Number of Children
* Smoker/Non-smoker
* Region

## 🧪 Model & Preprocessing

* LabelEncoding & OneHotEncoding for categorical data
* Dummy variable trap handled by removing one encoded column
* Scikit-learn Linear Regression
* Train-test split: 80/20

## 📈 Output

* Model coefficients and intercept
* R² score for performance
* Scatter plot comparing actual vs. predicted charges

## ✅ Requirements

* Python 3.x
* pandas
* numpy
* seaborn
* matplotlib
* scikit-learn

## 🚀 How to Run

1. Install dependencies:

   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn
   ```

2. Update the path to your dataset if necessary.

3. Run the script:

   ```bash
   python Insurance_bill_prediction.py
   ```
