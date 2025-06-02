#------Step1. Importing Necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

#------Step 2 Loading the data set
social_net_ads = pd.read_csv("C:/Users/lenyo/Documents/Datasets/Logistic_regression/Social_Network_Ads.csv")
#print(social_net_ads.head())

#------Step 3 Converting categorical data into numerical 
social_net_ads = pd.get_dummies(social_net_ads, columns = ["Gender"], drop_first = True)
#print(social_net_ads.info())

#------Step 4 Selecting dependent and independent variables
x = social_net_ads.iloc[:,1:-1].values
y = social_net_ads.iloc[:,-1].values

#------Step 5 splitting the dataset into train and test sets
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.23, random_state = 42)
sc = StandardScaler()
#x_train= sc.fit_transform(x_train)
#x_test= sc.fit_transform(x_test)

#------Step 6 Making an intsance of the model and fitting the train set
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train,y_train)

#------Step 7 predicting the output on the test set
y_pred = logisticRegr.predict(x_test)

#------Step 8 determining the accuracy of the model
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test, y_pred))