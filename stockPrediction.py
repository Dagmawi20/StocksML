#import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

df = pd.read_csv('APPL_Historical_Data.csv')

print(df.head())

#plotting data from csv

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

df['Date'] = pd.to_datetime(df.Date,format='%m/%d/%Y')
df.index = df['Date']
print(df.iloc[:,1])
print(df.iloc[:,2])
plt.figure(figsize=(12,6))
plt.plot(df.iloc[:,1])
plt.show()


#implementing Long Short Term Memory ML model for APPl stock for the past year

#input gate : adds information
#forget gate: removal of non-required information
#output gate: shows selected output of model

#importing libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

#orgnaize dataframe created from APPL stock information
data = df.sort_index(ascending=True,axis=0)
print(data)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date','Close'])