# This part of the code is inspired from the tutorials in Machine Learning Mastery https://machinelearningmastery.com/
import numpy as np
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import TimeDistributed
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.core import Dense, Activation, Dropout
import warnings
warnings.filterwarnings('ignore')

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

def stateful_cut(arr, batch_size, T_after_cut):
    if len(arr.shape) != 3:
        # N: Independent sample size,
        # T: Time length,
        # m: Dimension
        print("ERROR: please format arr as a (N, T, m) array.")

    N = arr.shape[0]
    T = arr.shape[1]

    # We need T_after_cut * nb_cuts = T
    nb_cuts = int(T / T_after_cut)
    if nb_cuts * T_after_cut != T:
        print("ERROR: T_after_cut must divide T")

    # We need batch_size * nb_reset = N
    # If nb_reset = 1, we only reset after the whole epoch, so no need to reset
    nb_reset = int(N / batch_size)
    if nb_reset * batch_size != N:
        print("ERROR: batch_size must divide N")

    # Cutting (technical)
    cut1 = np.split(arr, nb_reset, axis=0)
    cut2 = [np.split(x, nb_cuts, axis=1) for x in cut1]
    cut3 = [np.concatenate(x) for x in cut2]
    cut4 = np.concatenate(cut3)
    return(cut4)

# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

    s = int(len(data)/n_in)*n_in
    
    data = data[0:s,:]
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def data_prep(dataset, look_back, look_ahead, split_ratio=0.7): 
 
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset.reshape(-1, 1))

    # Split the dataset into training and testing 
    train_size = int(len(dataset) * split_ratio)
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    # reshape into X=t and Y=t+1
    train = series_to_supervised(train, look_back,look_ahead)
    test = series_to_supervised(test, look_back,look_ahead) 

    X_train,y_train=train.values[:,0:look_back],train.values[:,(look_back+look_ahead-1)]
    X_test,y_test=test.values[:,0:look_back],test.values[:,(look_back+look_ahead-1)]

    # reshape input to be [samples, time steps, features]

    X_train = np.reshape(X_train, (1,X_train.shape[0], X_train.shape[1]))
    X_test = np.reshape(X_test, (1,X_test.shape[0], X_test.shape[1]))

    y_train = np.reshape(y_train, (1, y_train.shape[0],1))
    y_test = np.reshape(y_test, (1,y_test.shape[0], 1))

    print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

    return X_train, y_train, X_test, y_test
  

def model_RNN(batch_size, dim_in, dim_out):
    model = Sequential()
    model.add(SimpleRNN(batch_input_shape=(batch_size, None, dim_in),return_sequences=True, units=100, stateful=True))
    model.add(Dropout(0.3))
    model.add(SimpleRNN(batch_input_shape=(batch_size, None, dim_in),return_sequences=True, units=50, stateful=True))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(activation='linear', units=dim_out)))
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.99, beta_2=0.99)
    model.compile(loss='mse', optimizer= opt,metrics=['accuracy'])
    
    return model

def model_stateless_RNN(model, dim_in, dim_out):
    model_stateless = Sequential()
    model_stateless.add(SimpleRNN(input_shape=(None, dim_in),return_sequences=True, units=100))
    model_stateless.add(Dropout(0.3))
    model_stateless.add(SimpleRNN(input_shape=(None, dim_in),return_sequences=True, units=50))
    model_stateless.add(Dropout(0.3))
    model_stateless.add(TimeDistributed(Dense(activation='linear', units=dim_out)))
    model_stateless.compile(loss='mse', optimizer='adam')
    model_stateless.set_weights(model.get_weights())
    return model_stateless

def model_LSTM(batch_size, dim_in, dim_out):
    model = Sequential()
    model.add(LSTM(batch_input_shape=(batch_size, None, dim_in),return_sequences=True, units=100, stateful=True))
    model.add(Dropout(0.3))
    model.add(LSTM(batch_input_shape=(batch_size, None, dim_in),return_sequences=True, units=50, stateful=True))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(activation='linear', units=dim_out)))
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.99, beta_2=0.99)
    model.compile(loss='mse', optimizer= opt,metrics=['accuracy'])
    
    return model

def model_stateless_LSTM(model, dim_in, dim_out):
    model_stateless = Sequential()
    model_stateless.add(LSTM(input_shape=(None, dim_in),return_sequences=True, units=100))
    model_stateless.add(Dropout(0.3))
    model_stateless.add(LSTM(input_shape=(None, dim_in),return_sequences=True, units=50))
    model_stateless.add(Dropout(0.3))
    model_stateless.add(TimeDistributed(Dense(activation='linear', units=dim_out)))
    model_stateless.compile(loss='mse', optimizer='adam')
    model_stateless.set_weights(model.get_weights())
    return model_stateless