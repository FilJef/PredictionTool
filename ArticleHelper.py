import pandas
import numpy
import Generic_parser
from numpy import array
def formatData(StockData, gcloudcon):

    StockData = StockData.values
    x = [0] * 31
    y = StockData[0, 1]

    for n in range(1, len(StockData) - 1):
        NewsArticles = pandas.read_sql_query(
            "SELECT * FROM articles WHERE date >= '{0}' AND date < '{1}'".format(StockData[n, 0], StockData[n + 1, 0]),
            con=gcloudcon)
        NewsArticles = NewsArticles.values

        date = n

        LMheadline = []
        LMarticle = []
        isArticles = False
        # Sentiment of the news articles
        for i in range(len(NewsArticles)):
            isArticles = True
            LMheadline.append(Generic_parser.get_data(NewsArticles[i, 1].upper()))

            # Check there is an article body, is not just 0 it
            if NewsArticles[i, 2] is not "":
                LMarticle.append(Generic_parser.get_data(NewsArticles[i, 2].upper()))
            else:
                LMarticle.append([0] * 17)

        if isArticles is True:
            LMheadline = averageArray(LMheadline)
            LMarticle = averageArray(LMarticle)
            LM = LMheadline[2:len(LMheadline)] + LMarticle[2:len(LMarticle)]
            date = [date] + LM
            x = numpy.vstack((x, date))
        else:
            date = [date] + ([0] * 30)
            x = numpy.vstack((x, date))

        y = numpy.vstack((y, StockData[n, 1]))
    return x, y

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
    x_test =[]
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
        