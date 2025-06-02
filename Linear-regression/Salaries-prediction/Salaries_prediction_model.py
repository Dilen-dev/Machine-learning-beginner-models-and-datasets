#------Step 1 Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#------Step 2 Loading the dataset
salaries = pd.read_csv("C:/Users/lenyo/Documents/Datasets/Salary linear regression/Salary_dataset.csv")
x = salaries.iloc[:,:-1].values
y = salaries.iloc[:, -1].values

#print(salaries.head())

#------Step 3 Data visualization
#sns.heatmap(salaries.corr())
#plt.show()

#------Step 6 Splitting the dataset into Training and Testing(we jump step 4 and 5 because we did not have non-numeric data)
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#------Step 7 Fitting the model with Train data
model = LinearRegression()
model.fit(x_train, y_train)

#------Step 8 Predicting the Test set using the trained model
y_pred = model.predict(x_test)
#print(y_pred)

#------Step 9 Calculating the coefficients and intercept
print(model.coef_)
print(model.intercept_)

#------Step 10 Evaluating the model
print(r2_score(y_test, y_pred))

#------Step 11 Making the scatterplot
sns.scatterplot(x = y_test, y = y_pred, marker = "+")
plt.show()
