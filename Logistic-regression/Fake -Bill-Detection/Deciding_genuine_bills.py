#------Step 1 Importing the neceessary libraries
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

#------Step 2 Loading the data set  
dataset = pd.read_csv("C:/Users/lenyo/Documents/Datasets/Logistic_regression/fake_bills.csv")
#print(dataset.shape)
#print(dataset.info())

#------Step 3 addressing the missing values
#print(np.mean(dataset["margin_low"]))
dataset["margin_low"].fillna((dataset["margin_low"].mean()), inplace = True)
#print(dataset.info())

#------Step 4 defining the dependent and independent variables
x = dataset.iloc[:,1:].values
#print(x)
y = dataset.iloc[:,0].values
#print(y)



#------Step 5 Dividing the dataset into Train and test data
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.25, random_state=42)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#------Step 6 Making an instance of the model and fitting it with training set
log_model = LogisticRegression()
log_model.fit(x_train,y_train)

#------Step 7 Predicting the test set
y_pred = log_model.predict(x_test)

#------Step 8 Evaluating the Accuracy of the model using confusion matrix,classification_report and Accuracy_score
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
