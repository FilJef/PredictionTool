from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from numpy import array
from datetime import datetime
from keras.callbacks import EarlyStopping
from sklearn.model_selection import TimeSeriesSplit
from sklearn import model_selection
import matplotlib.pyplot as plt
import pymysql
import pandas
import ArticleHelper
import numpy


def plotoutput(X_train, y_train, X_test, y_test, model, graphtitle, length):

    index = []
    for n in range(length):
        index.append(n)

    X_train, y_train, X_test, y_test = ArticleHelper.scaleDataset(X_train, y_train, X_test, y_test)

    if graphtitle[0] is 'S':
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train, epochs=100, batch_size=1)

    plt.scatter(index[0:len(X_train)], y_train, color='black', label='train data')
    plt.scatter(index[len(X_train):len(X_train) + len(X_test)], y_test, color='blue', label='test data')
    plt.plot(index[0:len(X_train) + len(X_test)], numpy.append(y_train, y_test), color='red', label='real values')
    plt.plot(index[len(X_train):len(X_train) + len(X_test)], model.predict(X_test),
             color='green', label='model prediction')

    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title(graphtitle)
    plt.legend()
    plt.show()


def SVR_gen(x, y, title):

    rbfSVR = SVR(kernel='rbf', C=1e3, gamma=0.1)

    tscv = TimeSeriesSplit(n_splits=10)

    for train_index, test_index in tscv.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        plotoutput(X_train, y_train, X_test, y_test, rbfSVR, title, len(x))

    kfold = model_selection.KFold(n_splits=10)
    results = model_selection.cross_val_score(rbfSVR, x, y, cv=kfold, scoring='neg_mean_absolute_error')
    print("SVR MAE: {0} ({1})".format(results.mean(), results.std()))
    results = model_selection.cross_val_score(rbfSVR, x, y, cv=kfold, scoring='neg_mean_squared_error')
    print("SVR MSE: {0} ({1})".format(results.mean(), results.std()))
    results = model_selection.cross_val_score(rbfSVR, x, y, cv=kfold, scoring='r2')
    print("SVR R^2: {0} ({1})".format(results.mean(), results.std()))


def neuralnet(x , y, title):

    model = Sequential()
    model.add(Dense(units=12, activation='relu', input_dim=len(x[0])))
    model.add(Dense(units=1, activation='relu'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    length = len(x)
    split = int((length/100) * 70)
    X_train = x[0:split]
    y_train = y[0:split]
    X_test = x[split+1:length]
    y_test = y[split+1:length]

    plotoutput(X_train, y_train, X_test, y_test, model, title, length)


def LSTM(x, y, features, title):

    length = len(x)

    x_train, y_train, x_test, y_test = ArticleHelper.createLSTMarrays(x, y)

    model = Sequential()
    model.add(LSTM(features, (50, features), return_sequences=True))
    model.add(Dense(units=len(date[0])))
    model.add(Dense(units=1))

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    plotoutput(X_train, y_train, X_test, y_test, model, title, length)

    print(model.evaluate(x_test))

gcloudcon = pymysql.connect(host='127.0.0.1',
                            database='store',
                            user='SQLUser',
                            password='3dWHUFePz9dHkFn')
cur = gcloudcon.cursor()

StockData = pandas.read_sql_query('SELECT * FROM dji', con=gcloudcon)

lookback = 3

# baselineX, baselineY = ArticleHelper.formatData(StockData, gcloudcon, False, lookback)
# x, y = ArticleHelper.formatData(StockData, gcloudcon, True, lookback)
#
# title = "SVR based off close prices"
# print(title)
# SVR_gen(baselineX, baselineY, title)
# title = "SVR close prices and news articles"
# print(title)
# SVR_gen(x, y, title)
#
# baselineX, baselineY = ArticleHelper.formatData(StockData, gcloudcon, False, lookback)
# x, y = ArticleHelper.formatData(StockData, gcloudcon, True, lookback)
#
# title = "Neural network based off close prices"
# print(title)
# neuralnet(baselineX, baselineY, title)
# title = "Neural network based off close prices and news articles"
# print(title)
# neuralnet(x, y, title)

baselineX, baselineY = ArticleHelper.formatData(StockData, gcloudcon, False, 1)
x, y = ArticleHelper.formatData(StockData, gcloudcon, True, 1)

title = "LSTM Neural network based off close prices"
print(title)
LSTM(baselineX, baselineY, 1, title)
title = "LSTM Neural network based off close prices and news articles"
print(title)
LSTM(x, y, 31, title)