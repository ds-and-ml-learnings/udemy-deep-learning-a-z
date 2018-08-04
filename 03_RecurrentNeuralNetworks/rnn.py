from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import pandas_datareader.data as web
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime
import numpy as np
import pandas as pd
import math






def get_dataset(stock='GOOG', start_tuple, end_tuple):
    start = datetime(2013, 1, 1)
    end = datetime(2017, 12, 31)
    dataset = web.DataReader(stock, 'iex', start, end)
    dataset.reset_index(inplace=True) ## Changing the index value
    dataset.loc[:, 'open'].values.reshape(-1,1)

    return dataset

def get_standarized(start, end):
    return MinMaxScaler(feature_range=(start, end))


def train_processing():
    training_set = get_dataset('GOOG', (2013, 1, 1), (2017, 12 ,31))
    sc = get_standarized(0,1)
    training_set_scaled = sc.fit_transform(training_set)

    X_train = []
    y_train = []
    [X_train.append(training_set_scaled[i-60:i, 0]) for i in range(60, 949)]
    [y_train.append(training_set_scaled[i, 0]) for i in range(60, 949)]
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return [X_train, y_train, sc]

def test_processing(dataset_train):
    dataset_test = get_dataset('GOOG', (2018, 1, 1), (2018, 5, 20))

    dataset_total = pd.concat([dataset_train['open'], dataset_test['open']], axis=0)

    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    inputs = inputs.reshape(-1,1)
    sc = get_standarized(0,1)
    inputs = sc.transform(inputs)

    X_test = []
    for i in range(60, 156): # To-date dates + 60
        X_test.append(inputs[i-60:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return [X_test,current_stock_price]

def build_classifier():
    # Initialisizing the RNN
    regressor = Sequential()
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(.2))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(.2))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(.2))
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(.2))
    regressor.add(Dense(units=1)) # The stock price we are prediciting is only one value
    regressor.compile(optimizer='Adam', loss='mean_squared_error')

    return regressor

def return_score():
    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    rmse = math.sqrt(mean_squared_error(current_stock_price, predicted_stock_price))
    current_stock_mean = np.mean(current_stock_price)
    return rmse/current_stock_mean


def start_rnn():
    X_train, y_train, sc = train_processing()
    X_test, current_stock = test_processing(X_train)





X_train, y_train = train_processing()
X_test, current_stock_price = test_processing()

classifier = KerasClassifier(build_fn=build_classifier, batch_size=30, epochs=500)
parameters = {
    'batch_size': [20, 25, 30],
    'epochs': [100,300,500],
}
grid_search = GridSearchCV(param_grid=parameters,
                            scoring='neg_mean_squared_error',
                            cv=10,
                            n_jobs=-1)

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracies = grid_search.best_score_
print('{0} were the best parameters to use'.format(best_parameters))
print('{0} was the accuracies'.format(best_accuracies))

start_rnn()

regressor = build_classifier()
return_score(regressor, X_test, current_stock_price)
