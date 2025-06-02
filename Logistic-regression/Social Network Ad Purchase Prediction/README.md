# Social Network Ad Purchase Prediction

This project uses Logistic Regression to predict whether a user purchases a product based on social media ad data, including age, estimated salary, and gender.

## 📄 Script Overview

- Loads and preprocesses the dataset
- Encodes categorical data (gender)
- Splits data into training and test sets
- Trains a logistic regression model
- Evaluates the model using accuracy, confusion matrix, and classification report

## 📊 Dataset

The script uses a CSV file named Social_Network_Ads.csv with columns like:

| User ID | Gender | Age | EstimatedSalary | Purchased |
|---------|--------|-----|------------------|-----------|

The target variable is Purchased (1 = bought the product, 0 = did not).

Make sure to update the file path if needed:
```python
social_net_ads = pd.read_csv("path/to/Social_Network_Ads.csv")
````

## 🔍 Features Used

* Age
* Estimated Salary
* Gender (converted to numeric)

## 🧪 Model & Processing

* Logistic Regression using scikit-learn
* Categorical encoding with pd.get\_dummies()
* Train-test split: 77% training / 23% testing
* Optional: Feature scaling using StandardScaler (currently commented out)

## ✅ Output

* Precision, recall, and F1-score
* Confusion matrix
* Accuracy score

## 🛠 Requirements

* Python 3.x
* pandas
* matplotlib (optional)
* scikit-learn

## 🚀 How to Run

1. Install dependencies:

   ```bash
   pip install pandas matplotlib scikit-learn
   ```

2. Ensure the dataset path is correct.

3. Run the script:

   ```bash
   python your_script_name.py
   ```


