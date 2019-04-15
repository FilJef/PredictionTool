from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
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

    # Turn the price array into a 1D array so sklearn doesnt shout at me
    price = price.flatten()

    tscv = TimeSeriesSplit(n_splits=10)

    for train_index, test_index in tscv.split(date):
        X_train, X_test = date[train_index], date[test_index]
        y_train, y_test = price[train_index], price[test_index]

        rbfSVR.fit(X_train, y_train)
        plt.scatter(X_train[:,0], y_train, color='black', label='train data')
        plt.scatter(X_test[:,0], y_test, color='blue', label='test data')
        plt.plot(X_test[:,0], y_test, color='red', label='real values')
        plt.plot(X_train[:,0], rbfSVR.predict(X_train), color='yellow', label='rbf')
        plt.plot(X_test[:,0], rbfSVR.predict(X_test), color='green', label='rbf prediction')

        plt.xlabel('Date')
        plt.ylabel('Price')

        plt.legend()
        #plt.show()

    #rbfSVR.fit(date, price)
    kfold = model_selection.KFold(n_splits=10)
    results = model_selection.cross_val_score(rbfSVR, date, price, cv=kfold, scoring='neg_mean_absolute_error')
    print("SVR MAE: {0} ({1})".format(results.mean(), results.std()))
    results = model_selection.cross_val_score(rbfSVR, date, price, cv=kfold, scoring='neg_mean_squared_error')
    print("SVR MSE: {0} ({1})".format(results.mean(), results.std()))
    results = model_selection.cross_val_score(rbfSVR, date, price, cv=kfold, scoring='r2')
    print("SVR R^2: {0} ({1})".format(results.mean(), results.std()))

    return


def ML(date, price):

    model = Sequential()
    model.add(LSTM(units=len(date[0]),  input_shape=(len(date), len(date[0]))))
    model.add(Dense(units=len(date[0])))
    model.add(Dense(units=1))

    model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])


    date = date.reshape(1, len(date), len(date[0]))
    price = price.flatten()
    price = price.reshape(1, len(price))

    print(model.input_shape)
    print(date.shape)
    print(model.output_shape)
    print(price.shape)
    model.summary()
    model.fit(date, price)

    loss_and_metrics = model.evaluate(date, price, batch_size=128)
    print(loss_and_metrics)

gcloudcon = pymysql.connect(host='127.0.0.1',
                            database='store',
                            user='SQLUser',
                            password='3dWHUFePz9dHkFn')
cur = gcloudcon.cursor()

#NewsArticles = pandas.read_sql_query('SELECT * FROM articles WHERE article IS NOT NULL', con=gcloudcon)
StockData = pandas.read_sql_query('SELECT * FROM dji', con=gcloudcon)

date = numpy.array(object)
price = numpy.array(object)

date, price = ArticleHelper.formatData(StockData, gcloudcon)


#Just dates and price
hold = date[:,0]
hold = hold.reshape(-1, 1)
print("Linear regression based off close prices")
#value = linear(hold, price)
print("Linear regression close prices and news articles")
#value = linear(date, price)
print("SVR based off close prices")
#value = SVR_gen(hold, price)
print("SVR close prices and news articles")
#value = SVR_gen(date, price)
print("Neural network based off close prices")
ML(hold, price)
print("Neural network based off close prices and news articles")
ML(date, price)

