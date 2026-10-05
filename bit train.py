#Importing the basic librarires for analysis

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("ggplot")  #using style ggplot


from mpl_toolkits.mplot3d import Axes3D
import datetime as dt
import plotly.graph_objects as go
import plotly.express as px


#Importing the dataset
df =pd.read_csv(r"C:\Users\91821\Downloads\Final code of Bit Coin Prediction Using Machine Learning\Bitcoin.csv")


# look the data set
df.head()


# looking the shape DataSet
df.shape


#Checking the dtypes of all the columns

df.info()


#Conversion data type column - Date from object to Datetime

df["Date"]=pd.to_datetime(df["Date"])

#checking null value 
df.isna().sum()

# look  describe data set
df.describe().round(2)


#add columns Day 
df["day"]=df['Date'].dt.day_name()


# how much repeat the Day  in the dataset
plt.figure(figsize=(10, 7))
sns.lineplot(x=df["day"], y=df["Bit_price"])
plt.show()
'''
plt.figure(figsize=(10,7))
sns.lineplot(df["day"],df["Bit_price"])
plt.show()
'''


#Importing the basic librarires for building model


from sklearn.linear_model import LinearRegression  
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error ,mean_squared_error, median_absolute_error,confusion_matrix,accuracy_score,r2_score

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler ,PolynomialFeatures,minmax_scale,MaxAbsScaler ,LabelEncoder

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectPercentile
from sklearn.neural_network import MLPRegressor

from sklearn.svm import SVR

#  add column Year , month and date
df['Year']=df['Date'].dt.year
df['Month']=df['Date'].dt.month


#drop column Day, Date
df.drop(columns=["Date","day"],inplace=True)
#df.drop(columns=["date"],inplace=True)

df.head()


#Defined X value and y value , and split the data train

X = df.drop(columns="Volume")           
y = df.drop(columns="Bit_price")    # y = quality


# split the data train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("X Train : ", X_train.shape)
print("X Test  : ", X_test.shape)
print("Y Train : ", y_train.shape)
print("Y Test  : ", y_test.shape)

from sklearn import datasets, linear_model, metrics
digits = datasets.load_digits()
X = digits.data
y = digits.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=1)
reg = linear_model.LogisticRegression()

reg.fit(X_train, y_train)
# making predictions on the testing set
y_pred = reg.predict(X_test)

# comparing actual response values (y_test) with predicted response values (y_pred)
print("Linear Regression model accuracy(in %):",
metrics.accuracy_score(y_test, y_pred)*100)


