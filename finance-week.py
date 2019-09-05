# import the libraries used
import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame
import numpy as np
import math

# scikit libraries
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# set up start and end dates for the stock prices to pull
start = datetime.datetime(2011, 1 ,1)
end = datetime.datetime(2019, 9, 4)

df = web.DataReader("AAPL", 'yahoo', start, end)
df.tail()
# print(df.tail())

# grab the closing column
close_px = df['Adj Close']

# set up the rolling mean/moving average and plotting it
mavg = close_px.rolling(window=100).mean()

# matplot 
import matplotlib.pyplot as plt
from matplotlib import style

# Adjusting the size of matplotlib
import matplotlib as mpl
mpl.rc('figure', figsize=(8, 7))
mpl.__version__

# Adjusting the style of matplotlib
style.use('ggplot')

close_px.plot(label='AAPL')
mavg.plot(label='mavg')
plt.legend()
# plt.show()

# showing return deviation
plt.figure(2)
rets = close_px / close_px.shift(1)
rets.plot(label='return')
# plt.show()


# engineering features (High Low Percentage and Percentage Change)
dfreg = df.loc[:, ['Adj Close', 'Volume']]
dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
dfreg['PCT_change'] =(df['Close'] - df['Open']) / df['Open'] * 100.0

# some pre-processing and cross validation
# Drop missing values
dfreg.dropna(inplace=True)

# separating out 1 percent of data to forecast
forecast_out = int(math.ceil(0.01 * len(dfreg)))

# pull out the label to predict Adjusted Close
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)

X = np.array(dfreg.drop(['label'], 1))

# feature scaling for X so all features have the same distribution for linear regression
X = preprocessing.scale(X)

# data series for late X (to evaluate model) and early X (to train model)
X_lately = X[-forecast_out]
X = X[:-forecast_out]

# separate label and identify as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]

# MODEL GENERATION
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Linear regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)
# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)

# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)

# KNN Regression
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X_train, y_train)

confidencereg = clfreg.score(X_test, y_test)
print(confidencereg)