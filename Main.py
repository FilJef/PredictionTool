from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from numpy import array
import pymysql
import pandas
import Generic_parser
import numpy

class Article(object):

    def __init__(self, df):
        data = [0] * 5
        x = 0
        for item in df[1].iteritems():
            data[x] = item[1]
            x += 1
        
        self.date = data[0]
        self.headline = data[1]
        self.article = data[2]
        self.link = data[3]
        self.sentiment = data[4]


gcloudcon = pymysql.connect(host='127.0.0.1',
                            database='store',
                            user='SQLUser',
                            password='3dWHUFePz9dHkFn')
cur = gcloudcon.cursor()

NewsArticles = pandas.read_sql_query('SELECT * FROM articles WHERE article IS NOT NULL', con=gcloudcon)
StockData = pandas.read_sql_query('SELECT * FROM dji', con=gcloudcon,index_col='date')
#abc = NewsArticles.iloc[::-1]
Xclose = numpy.array(object)
X = numpy.array(object)
Y = numpy.array(object)

StockData = StockData.values
X = StockData[0, 3]
Y = StockData[1, 3]
for n in range(1, len(StockData) - 1):
    X = numpy.vstack((X, StockData[n, 3]))
    Y = numpy.vstack((Y, StockData[n + 1, 3]))

        
scaler = MinMaxScaler(feature_range=(-1, 1))

X = scaler.fit_transform(X)
Y = scaler.fit_transform(Y)
model = Sequential()
#Y = Y[:, 0]
#print(X.shape)

lengthX = len(X)
samples = list()
samples.append(X)
X = array(samples)
X = X.reshape((-1, lengthX, 1))

lengthY = len(Y)
Y = Y.reshape((1, lengthY))

print(X.shape)
print(Y.shape)

model.add(LSTM(4, input_shape=(lengthX, 1)))
model.add(Dense(units=20, activation='relu'))
model.add(Dense(units=1, activation='relu'))

model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
print(model.input_shape)
print(model.output_shape)
model.fit(X, Y, batch_size=20, epochs=1000)

