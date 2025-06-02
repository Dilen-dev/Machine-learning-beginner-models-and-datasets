#------Step 1 Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#------Step 2 Loading the datasets
insurance = pd.read_csv("C:/Users/lenyo/Documents/Datasets/Medical linear regression/insurance.csv")
x = insurance.iloc[:,:-1].values
y = insurance.iloc[:, -1].values

#print(insurance.head())

#------Step 3 Data visualization
#sns.heatmap(insurance.select_dtypes(include=[np.number]).corr(), annot = True)
#plt.show()

#------Step 4 Encoding non-numeric
labelencoder = LabelEncoder()
categorical_indices = [1,4,5]

for i in categorical_indices:
    x[:, i] = labelencoder.fit_transform(x[:, i])

col_Tran = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), categorical_indices)], remainder = "passthrough")
x = col_Tran.fit_transform(x)

#------Step 5 Avoiding Dummy variable Trap
x = x[:,1:]

#------Step 6 Splitting the Dataset into Training and Testing data
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

#------Step 7 Fitting the model with Training data
model = LinearRegression()
model.fit(x_train,y_train)

#------Step 8 Predicting the Test data using the trained model
y_pred = model.predict(x_test)
#print(y_pred)

#------Step 9 Calculating the coefficients and the intercept
print(model.coef_)
print(model.intercept_)

#------Step 10 Evaluating the model
print(r2_score(y_test,y_pred))

#------Step 11 Creating a Scatterplot
sns.scatterplot(x = y_test, y = y_pred, marker = "+")
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()