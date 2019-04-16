from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from numpy import array
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pymysql
import pandas
import ArticleHelper
import numpy
from sklearn.preprocessing import MinMaxScaler

def linear(date, price):
    lin = LinearRegression()
    lin.fit(date, price)

    plt.plot(date[:,0], price, label='real values')
    plt.plot(lin.predict(date), label='linear regression')
    #plt.show()

    kfold = model_selection.KFold(n_splits=10)
    results = model_selection.cross_val_score(lin, date, price, cv=kfold, scoring='neg_mean_absolute_error')
    print("Lin MAE: {0} ({1})".format(results.mean(), results.std()))
    results = model_selection.cross_val_score(lin, date, price, cv=kfold, scoring='neg_mean_squared_error')
    print("Lin MSE: {0} ({1})".format(results.mean(), results.std()))
    results = model_selection.cross_val_score(lin, date, price, cv=kfold, scoring='r2')
    print("Lin R^2: {0} ({1})".format(results.mean(), results.std()))

def SVR_gen(date, price):

    rbfSVR = SVR(kernel='rbf', C=1e3, gamma=0.1)

    index = []
    for n in range(len(date)):
        index.append(n)

    tscv = TimeSeriesSplit(n_splits=10)

    # for train_index, test_index in tscv.split(date):
    #     X_train, X_test = date[train_index], date[test_index]
    #     y_train, y_test = price[train_index], price[test_index]
    #
    #     scaler = MinMaxScaler(feature_range=(-1, 1))
    #     X_train = scaler.fit_transform(X_train)
    #     X_test = scaler.fit_transform(X_test)
    #     y_train = scaler.fit_transform(y_train.reshape(-1, 1))
    #     y_test = scaler.fit_transform(y_test.reshape(-1, 1))
    #
    #     y_train = y_train.flatten()
    #     y_test = y_test.flatten()
    #
    #     rbfSVR.fit(X_train, y_train)
    #     plt.scatter(index[0:len(X_train)], y_train, color='black', label='train data')
    #     plt.scatter(index[len(X_train):len(X_train) + len(X_test)], y_test, color='blue', label='test data')
    #     plt.plot(index[len(X_train):len(X_train) + len(X_test)], y_test, color='red', label='real values')
    #     plt.plot(index[0:len(X_train)], rbfSVR.predict(X_train), color='yellow', label='rbf')
    #     plt.plot(index[len(X_train):len(X_train) + len(X_test)], rbfSVR.predict(X_test), color='green', label='rbf prediction')
    #
    #     plt.xlabel('Date')
    #     plt.ylabel('Price')
    #
    #     plt.legend()
    #     plt.show()

    #date, price = ArticleHelper.createTimeseries(date,price)n
    #rbfSVR.fit(date, price)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    date = scaler.fit_transform(date)
    price = scaler.fit_transform(price.reshape(-1,1))
    price = price.flatten()

    kfold = model_selection.KFold(n_splits=10)
    results = model_selection.cross_val_score(rbfSVR, date, price, cv=kfold, scoring='neg_mean_absolute_error')
    print("SVR MAE: {0} ({1})".format(results.mean(), results.std()))
    results = model_selection.cross_val_score(rbfSVR, date, price, cv=kfold, scoring='neg_mean_squared_error')
    print("SVR MSE: {0} ({1})".format(results.mean(), results.std()))
    results = model_selection.cross_val_score(rbfSVR, date, price, cv=kfold, scoring='r2')
    print("SVR R^2: {0} ({1})".format(results.mean(), results.std()))

    return


def ML(date, price, features):
    date, price = ArticleHelper.createTimeseries(date,price)
    x_train, y_train, x_test, y_test = ArticleHelper.createLSTMarrays(date, price)

    model = Sequential()
    model.add(LSTM(units=features,  input_shape=(50, features), return_sequences=True))
    model.add(Dense(units=len(date[0])))
    model.add(Dense(units=1))

    model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])


    model.fit(x_train,y_train)
    
    print(model.evaluate(x_test))

gcloudcon = pymysql.connect(host='127.0.0.1',
                            database='store',
                            user='SQLUser',
                            password='3dWHUFePz9dHkFn')
cur = gcloudcon.cursor()

#NewsArticles = pandas.read_sql_query('SELECT * FROM articles WHERE article IS NOT NULL', con=gcloudcon)
StockData = pandas.read_sql_query('SELECT * FROM dji', con=gcloudcon)

baselineX, baselineY = ArticleHelper.formatData(StockData, gcloudcon, False)
x, y = ArticleHelper.formatData(StockData, gcloudcon, True)

# print("Linear regression based off close prices")
# linear(hold, price)
# print("Linear regression close prices and news articles")
# linear(date, price)
print("SVR based off close prices")
SVR_gen(baselineX, baselineY)
print("SVR close prices and news articles")
SVR_gen(x, y)
# print("Neural network based off close prices")
# ML(hold, price, 6)
# print("Neural network based off close prices and news articles")
# ML(date, price, 31)

