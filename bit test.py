import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as py
import datetime as dt

data = pd.read_csv(r"C:\Users\91821\Downloads\Final code of Bit Coin Prediction Using Machine Learning\Bitcoin.csv")
data

data.head

data.head()

data.tail()


data.info()


print(data.columns)
#data.corr()['Close']

numeric_data = data.select_dtypes(include=['float64', 'int64'])
corr_close = numeric_data.corr()['Close']
print(corr_close)

data.drop(['Volume', 'Asset_ID'], axis = 1, inplace = True) # dropping the unnecessary features
data


data.isnull() # checking if any value in the dataset is null

data.duplicated().sum # checking if there is any null value on any date


data.isnull().sum() # number of null values of every feature


data.isnull().any() # rechecking if there are any null values in any feature

data.shape # shape of the dataset

data.head() # top 5 rows of the dataset

#add columns Date 
data["Date"]=data["Date"]


# how much repeat the Date and Bit_price in the dataset
plt.figure(figsize=(10, 7))
sns.lineplot(x=data["day"], y=data["Bit_price"])
plt.show()

'''
plt.figure(figsize=(10,7))
sns.lineplot(data["Date"],data["Bit_price"])
plt.show()
'''


#add columns day
data["day"]=data["day"]

# how much repeat the day and Bit_price in the dataset
plt.figure(figsize=(10, 7))
sns.lineplot(x=data["day"], y=data["Bit_price"])
plt.show()

'''
plt.figure(figsize=(10,7))
sns.lineplot(data["day"],data["Bit_price"])
plt.show()
'''

#add columns High
data["High"]=data["High"]


# how much repeat the High and Low in the dataset
plt.figure(figsize=(10, 7))
sns.lineplot(x=data["High"], y=data["Low"])
plt.show()

'''
plt.figure(figsize=(10,7))
sns.lineplot(data["High"],data["Low"])
plt.show()
'''


#add columns Open
data["Open"]=data["Open"]


# how much repeat the  Open and Close in the dataset
plt.figure(figsize=(10, 7))
sns.lineplot(x=data["Open"], y=data["Close"])
plt.show()

'''
plt.figure(figsize=(10,7))
sns.lineplot(data["Open"],data["Close"])
plt.show()
'''


#add columns Adj Close
data["Adj Close"]=data["Adj Close"]


# how much repeat the Adj Close  in the dataset

plt.figure(figsize=(10,7))
sns.countplot(data["Adj Close"])
plt.show()

 






