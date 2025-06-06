#------Step 1 Importing necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer

#------Step 2 Loading the dataset
used_cars = pd.read_csv("C:/Users/lenyo/Documents/Datasets/Random forest/Used cars.csv")
#print(used_cars.shape)
#print(used_cars.info())
#print(used_cars.head())
x = used_cars.iloc[:,:-1].values
y = used_cars.iloc[:,-1].values


#------Step 3 Encoding the non-numerical values
labelencoder = LabelEncoder()
categorical_columns = [0,4,5,6]

for i in categorical_columns:
    x[:,i] = labelencoder.fit_transform(x[:,i])

column_transformer = ColumnTransformer(transformers = [('encoder', OneHotEncoder(),categorical_columns)], remainder = 'passthrough')
x = column_transformer.fit_transform(x)

#------Step 4 Splitting the data into train and test data
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.3, random_state = 0)

#------Step 5 Instantiating the model and traininng it
mod = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=5, random_state=0)
mod.fit(x_train,y_train)

#------Step 6 Predicting the test set
y_pred = mod.predict(x_test)
#print(y_pred)

#------Step 7 Evaluating the accuracy
print(r2_score(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))
