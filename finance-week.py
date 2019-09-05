# import the libraries used
import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame

# scikit libraries
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


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
plt.show()

# showing return deviation
plt.figure(2)
rets = close_px / close_px.shift(1)
rets.plot(label='return')
plt.show()

# Linear regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)

# Quadratic regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_traing, y_train)

# Quadratic regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)

# KNN Regression
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X_train, y_train)

# need to back up to 'Predicting Stocks Price' in https://towardsdatascience.com/in-12-minutes-stocks-analysis-with-pandas-and-scikit-learn-a8d8a7b50ee7