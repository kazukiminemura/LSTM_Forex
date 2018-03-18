import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt



# load GBP/USD daily data
df = pd.read_csv('gbpusd_2015_2018.csv', index_col=0)


def _load_data(data, n_prev = 100):
    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY


def train_test_split(df, test_size=0.1, n_prev = 100):
    ntrn = round(len(df) * (1 - test_size))
    ntrn = int(ntrn)
    X_train, y_train = _load_data(df.iloc[0:ntrn], n_prev)
    X_test, y_test = _load_data(df.iloc[ntrn:], n_prev)

    return (X_train, y_train), (X_test, y_test)


def only_test_data(df, n_prev=100):
    X_test, y_test = _load_data(df.iloc[0:], n_prev)

    return (X_test, y_test)

length_of_sequences = 100
#(X_train, y_trian), (X_test, y_test) = train_test_split(df[["Close"]], n_prev = length_of_sequences)
#print(len(X_test))

(X_test, y_test) = only_test_data(df[["Price"]], n_prev = length_of_sequences)
X_test = X_test[::-1]
y_test = y_test[::-1]
#ntrn = len(X_test)
#X_test = X_test[ntrn - 400:]
#y_test = y_test[ntrn - 400:]
#print len(y_test)
#print y_test.shape

from keras.models import model_from_json
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM


#in_out_neurons = 1
#hidden_neurons = 300

#model = Sequential()
#model.add(LSTM(hidden_neurons, batch_input_shape=(None, length_of_sequences, in_out_neurons), return_sequences=False))
#model.add(Dense(in_out_neurons))
#model.add(Activation("linear"))
#model.compile(loss="mean_squared_error", optimizer="sgd")
#model.fit(X_train, y_trian, batch_size=600, nb_epoch=15, validation_split=0.05)


#json_string = model.to_json()
#open('LSTM_model.json', 'w').write(json_string)
#model.save_weights('LSTM_weights.h5')


### read model from json
json_string = open('LSTM_model.json').read()
model = model_from_json(json_string)

model.load_weights('LSTM_weights.h5')

predicted = model.predict(X_test)

dataf = pd.DataFrame(predicted[100:])
dataf.columns = ["predict"]
dataf["input"] = y_test[100:].astype(float)
dataf.plot(figsize = (15,5))
plt.show()
