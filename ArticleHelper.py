import pandas
import numpy
import Generic_parser
from numpy import array
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer


def scaleDataset(X_train, y_train, X_test, y_test):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = scaler.fit_transform(y_train.reshape(-1, 1))
    y_test = scaler.transform(y_test.reshape(-1, 1))
    
    missingImputer(X_train, X_test)

    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    return X_train, y_train, X_test, y_test


def missingImputer(X_train, X_test):
    # fill missing values with mean
    X_train = pandas.DataFrame(X_train)
    X_train = X_train.replace(-1, numpy.NAN)
    X_train = X_train.values

    X_test = pandas.DataFrame(X_test)
    X_test = X_test.replace(-1, numpy.NAN)
    X_test = X_test.values

    imputer = Imputer()
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.fit_transform(X_test)

    return X_train, X_test


def missingMean(X_train, X_test):
    # fill missing values with mean
    X_train = pandas.DataFrame(X_train)
    X_train = X_train.replace(-1, numpy.NAN)
    X_train.fillna(X_train.mean, inplace=True)
    X_train = X_train.values

    X_test = pandas.DataFrame(X_test)
    X_test = X_test.replace(-1, numpy.NAN)
    X_test.fillna(X_test.mean, inplace=True)
    X_test = X_test.values
    
    return X_train, X_test


def scaleKfold(x, y):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    x = scaler.fit_transform(x)
    y = scaler.fit_transform(y)
    return x, y


def formatData(StockData, gcloudcon, getArticles, lookback):

    flatStockData = StockData.values

    x = None

    if getArticles is True:
        x = [0] * 31
    else:
        x = [0] * 1

    y = [0] * 1

    for n in range(0, len(StockData) - 1):

        LMheadline = []
        LMarticle = []
        isArticles = False
        
        # Sentiment of the news articles
        if getArticles is True:
            query = "SELECT * FROM articles WHERE date >= '{0}' AND date < '{1}'"\
                .format(flatStockData[n, 0], flatStockData[n + 1, 0])
            NewsArticles = pandas.read_sql_query(query, con=gcloudcon)
            NewsArticles = NewsArticles.values

            for i in range(len(NewsArticles)):
                isArticles = True
                LMheadline.append(Generic_parser.get_data(NewsArticles[i, 1].upper()))

                # Check there is an article body, is not just 0 it
                if NewsArticles[i, 2] is not "":
                    LMarticle.append(Generic_parser.get_data(NewsArticles[i, 2].upper()))
                else:
                    LMarticle.append([0] * 17)

        data = None
        
        #append sentiment values to price
        if isArticles is True:
            LMheadline = averageArray(LMheadline)
            LMarticle = averageArray(LMarticle)
            LM = LMheadline[2:len(LMheadline)] + LMarticle[2:len(LMarticle)]
            data = [flatStockData[n, 4]] + LM
        elif getArticles is False:
            data = flatStockData[n, 4]
        else:
            data = [flatStockData[n, 4]] + ([0] * 30)

        x = numpy.vstack((x, data))
        y = numpy.vstack((y, flatStockData[n + 1, 4]))


    # remove the null rows we instantiated at the start
    x = numpy.delete(x, (0), axis=0)
    y = numpy.delete(y, (0), axis=0)

    x, y = createTimeseries(x, y, lookback)

    return x, y


def createTimeseries(inX, inY, lookback):
    x = []
    y = []

    for i in range(lookback, len(inX)):
        hold = []
        for j in range(lookback):
            hold = numpy.append(hold, (inX[i - j]))

        x.append(hold)
        y.append(inY[i])

    return array(x), array(y).flatten()

# Found on github
def averageArray(array):
    ncols = len(array[0])
    nrows = len(array)
    # Sum all elements in each column:
    results = ncols * [0]  # sums per column, afterwards avgs
    for col in range(ncols):
        for row in range(nrows):
            results[col] += array[row][col]
    # Then calculate averages:
    # * nrows is also number of elements in every col:
    nelem = float(nrows)
    results = [s / nelem for s in results]
    return results


def createLSTMarrays(x, y):

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    count = 0
    while count + 50 < len(x):
        xhold = []
        yhold = []
        for i in range(50):
            xhold.append(x[i + count])
            yhold.append(y[i + count])

        count += 50
        if count + 50 < len(x):
            x_train.append(xhold)
            y_train.append(yhold)
        else:
            x_test.append(xhold)
            y_test.append(yhold)

    return array(x_train), array(y_train), array(x_test), array(y_test)
