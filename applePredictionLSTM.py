import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yfin

yfin.pdr_override()

#get stock data of Apple from 01/01/2019 to 09/17/2021
df = pdr.get_data_yahoo('AAPL', start='2019-01-01', end='2021-09-17')
# find thape of our dataframe of Apple stock
print(df.shape)
#plot the closing price of Apple over our given time period
plt.figure(figsize=(16,8))
plt.title('AAPL closing price history')
plt.plot(df['Close'])
plt.xlabel('Date',fontsize = 18)
plt.ylabel('Closing Price in USD', fontsize = 18)


#split our df to create new df of only closing column
close_price_df = df.filter(['Close'])
dataset_cp_df = close_price_df.values
print(len(dataset_cp_df))

#number of rows allows for us to train the LSTM model using 80% of our data
training_data_size = math.ceil(len(dataset_cp_df)*.80)

# scale the closing price data, preprocssing transformation to input before adding to neural network
# range is between 0 and 1 inclusive
scaler = MinMaxScaler(feature_range=(0,1))
#transform data using scaler, 
scaled_data = scaler.fit_transform(dataset_cp_df)

# create scaled training data set
train_data = scaled_data[0:training_data_size,:]
#split data into x_train and y_train data sets
#x train is independent variables 
#y train is dependent variables
x_train = []
y_train = []

for i in range(60,len(train_data)):
  x_train.append(train_data[i-60:i,0])
  y_train.append(train_data[i,0])

#convert x_train and y_train to numpy arrays to use in LSTM
x_train, y_train = np.array(x_train),np.array(y_train)

#reshape x_train so LSTM expects 3D data and we are in a 2D data form
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
print(x_train.shape)

#build LSTM model
model = Sequential()
model.add(LSTM(50,return_sequences=True,input_shape = (x_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


# compile LSTM model checks how model does on training
model.compile(optimizer='adam',loss='mean_squared_error')
#begin training our model, epochs is number of iterations 
model.fit(x_train,y_train, batch_size=1,epochs=1)

# create testing data using scaled values from 83 to 178
tesing_data  = scaled_data[training_data_size-60: ,:]
# now create data sets for x and y
x_test,y_test = [],dataset_cp_df[training_data_size:,:]
for i in range(60,len(tesing_data)):
  x_test.append(tesing_data[i-60:i,0])
  

  # convert data to numpy arrays
x_test = np.array(x_test)
# reshpae our data to make it 3D
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
# get predicted values from our models
predictions = model.predict(x_test)
# inverse transform data from model to check they are same as y values(actual values)
predictions = scaler.inverse_transform(predictions)


#Check evalutaion of our model by checking RMSE: root mean squared error
rmse = np.sqrt(np.mean(predictions-y_test)**2)
print(rmse)

#plot data 
train = close_price_df[:training_data_size]
valid = close_price_df[training_data_size:]
valid['Predictions']= predictions
plt.figure(figsize=(16,8))
plt.title('LSTM model predictions')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Prices in USD',fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','Validation','Predictions'],loc ='lower right')
plt.show()

#Predict stock price for 9/20/2021
future_prices = pdr.get_data_yahoo('AAPL', start='2017-01-01', end='2021-09-17')
new_df = future_prices.filter(['Close']) 
#get the last 60 day closing price
last_two_months = new_df[-60:].values
#scale values between 0 and 1
last_two_months_scaled = scaler.transform(last_two_months)
#create an empty list
future_test = []
future_test .append(last_two_months_scaled)
#convert to numpy array
future_test = np.array(future_test)
future_test = np.reshape(future_test,(future_test.shape[0],future_test.shape[1],1))
#use our model
prediction_prices = model.predict(future_test)
prediction_prices = scaler.inverse_transform(prediction_prices)
print(prediction_prices)

#correct Price
future_prices_check = pdr.get_data_yahoo('AAPL', start='2021-09-19', end='2021-09-23')
print(future_prices_check['Close'][0])