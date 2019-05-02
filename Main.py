from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from numpy import array
from datetime import datetime
from keras.callbacks import CSVLogger
from sklearn.model_selection import TimeSeriesSplit
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras import metrics
import matplotlib.pyplot as plt
import pymysql
import pandas
import ArticleHelper
import numpy


def plotoutput(X_train, y_train, X_test, y_test, model, graphtitle, length, run):
    neuralnetwork = False
    index = []
    for n in range(length):
        index.append(n)

    if graphtitle[0] is 'S':
        model.fit(X_train, y_train)
    else:
        neuralnetwork = True
        csv_logger = CSVLogger('results/' + graphtitle[0:4] + '-' + str(run) +'-training.log')
        model.fit(X_train, y_train, epochs=75, batch_size=10, callbacks=[csv_logger], verbose = 0)

    plt.scatter(index[0:len(X_train)], y_train, color='black', label='train data')
    plt.scatter(index[len(X_train):len(X_train) + len(X_test)], y_test, color='blue', label='test data')
    plt.plot(index[0:len(X_train) + len(X_test)], numpy.append(y_train, y_test), color='red', label='real values')
    plt.plot(index[len(X_train):len(X_train) + len(X_test)], model.predict(X_test),
             color='green', label='model prediction')

    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title(graphtitle + ' ' + str(run) + '/10')
    plt.legend()
    plt.show()

    if neuralnetwork:
        results = pandas.read_csv('results/' + graphtitle[0:4] + '-' + str(run) +'-training.log')
        f, axarr = plt.subplots(2, sharex=True)
        f.suptitle('Loss and accuracy metrics')
        axarr[0].plot(results['epoch'], results['loss'])
        axarr[1].plot(results['epoch'], results['acc'])
        plt.show()

        scores = model.evaluate(X_test, y_test, verbose=0)

        for i in range(len(model.metrics_names)):
            print("{0}: {1}".format(model.metrics_names[i], scores[i] * 100))


def SVR_gen(x, y, title):

    rbfSVR = SVR(kernel='rbf', C=1e3, gamma=0.1)

    tscv = TimeSeriesSplit(n_splits=10)
    i = 1

    for train_index, test_index in tscv.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train, y_train, X_test, y_test = ArticleHelper.scaleDataset(X_train, y_train, X_test, y_test)

        plotoutput(X_train, y_train, X_test, y_test, rbfSVR, title, len(x), i)

        i = i + 1

    x, y = ArticleHelper.scaleKfold(x, y.reshape(-1, 1))
    skKfold(x, y)


def skKfold(x, y):
    rbfSVR = SVR(kernel='rbf', C=1e3, gamma=0.1)

    kfold = model_selection.KFold(n_splits=10)
    results = model_selection.cross_val_score(rbfSVR, x, y, cv=kfold, scoring='neg_mean_absolute_error')
    print("SVR MAE: {0} (+/- {1})".format(results.mean(), results.std()))

    results = model_selection.cross_val_score(rbfSVR, x, y, cv=kfold, scoring='neg_mean_squared_error')
    print("SVR MSE: {0} (+/-  {1})".format(results.mean(), results.std()))

    results = model_selection.cross_val_score(rbfSVR, x, y, cv=kfold, scoring='r2')
    print("SVR R^2: {0} (+/-  {1})".format(results.mean(), results.std()))


def neuralnet(x , y, title):

    length = len(x)
    split = int((length/100) * 70)
    X_train = x[0:split]
    y_train = y[0:split]
    X_test = x[split+1:length]
    y_test = y[split+1:length]

    X_train, y_train, X_test, y_test = ArticleHelper.scaleDataset(X_train, y_train, X_test, y_test)

    model = createNN(len(x[0]))

    plotoutput(X_train, y_train, X_test, y_test, model, title, length, 10)

    model = createNN(len(x[0]))

    Kfold(x, y, model)


def createNN(length):

    model = Sequential()
    model.add(Dense(units=12, activation='relu', input_dim=length))
    model.add(Dense(units=1, activation='relu'))
    model.compile(loss='mae', optimizer='adam', metrics=['accuracy', 'mae', 'mse'])

    return model


def lstm(x, y, features, title):

    length = len(x)

    X_train, y_train, X_test, y_test = ArticleHelper.createLSTMarrays(x, y)

    model = createLSTM(features)

    plotoutput(X_train, y_train, X_test, y_test, model, title, length, 10)

    model = createLSTM()

    Kfold(x, y, model)


def createLSTM(features):
    model = Sequential()
    model.add(LSTM(units=features, input_shape=(50, features)))
    model.add(Dense(units=features))
    model.add(Dense(units=1))

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    return model


def Kfold(X, Y, model):

    Wsave = model.get_weights()

    kfold = KFold(n_splits=10)

    cvscores = []
    for train, test in kfold.split(X, Y):
        # Reset the weights to start
        model.set_weights(Wsave)

        X_train, X_test = X[train],X[test]
        y_train, y_test = Y[train], Y[test]
        # scale datasets
        X_train,y_train, X_test, y_test = ArticleHelper.scaleDataset(X_train, y_train, X_test, y_test)
        # Fit the model
        model.fit(X_train, y_train, epochs=75, batch_size=10, verbose=0)
        # evaluate the model
        scores = model.evaluate(X_test, y_test, verbose=0)
        # #print("{0}: {1}".format(model.metrics_names[1], scores[1] * 100))
        # for i in range(len(model.metrics_names)):
        #     print("{0}: {1}".format(model.metrics_names[i], scores[i] * 100))
        cvscores.append(scores)

    cvscores = array(cvscores)
    for i in range(len(cvscores[0])):
        print("{0}: {1}(+/- {2})".format(model.metrics_names[i], numpy.mean(cvscores[:, i]), numpy.std(cvscores[:, i])))


gcloudcon = pymysql.connect(host='127.0.0.1',
                            database='store',
                            user='SQLUser',
                            password='3dWHUFePz9dHkFn')
cur = gcloudcon.cursor()

StockData = pandas.read_sql_query('SELECT * FROM dji', con=gcloudcon)

lookback = 3

#baselineX, baselineY = ArticleHelper.formatData(StockData, gcloudcon, False, lookback)
x, y = ArticleHelper.formatData(StockData, gcloudcon, True, lookback)
#
# title = "SVR based off close prices"
# print(title)
# SVR_gen(baselineX, baselineY, title)
title = "SVR close prices and news articles"
print(title)
SVR_gen(x, y, title)

# baselineX, baselineY = ArticleHelper.formatData(StockData, gcloudcon, False, lookback)
x, y = ArticleHelper.formatData(StockData, gcloudcon, True, lookback)
#
# title = "Neural network based off close prices"
# print(title)
# neuralnet(baselineX, baselineY, title)
title = "Neural network based off close prices and news articles"
print(title)
neuralnet(x, y, title)

baselineX, baselineY = ArticleHelper.formatData(StockData, gcloudcon, False, 1)
x, y = ArticleHelper.formatData(StockData, gcloudcon, True, 1)

title = "LSTM Neural network based off close prices"
print(title)
lstm(baselineX, baselineY, 1, title)
title = "LSTM Neural network based off close prices and news articles"
print(title)
lstm(x, y, 31, title)