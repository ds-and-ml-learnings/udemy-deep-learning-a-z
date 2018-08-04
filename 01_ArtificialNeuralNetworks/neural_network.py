import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def preprocessing():
    df = pd.read_csv("../archive/Churn_Modelling.csv")
    y = df['Exited']
    Xs = df.iloc[:, 3:13]

    # Preprocessing the DATA
    labelencoder_X_geo = LabelEncoder()
    Xs.loc[:, "Geography"] = labelencoder_X_geo.fit_transform(Xs.loc[:, "Geography"])
    labelencoder_X_gender = LabelEncoder()
    Xs.loc[:, "Gender"] = labelencoder_X_gender.fit_transform(Xs.loc[:, "Gender"])
    onehotencoder = OneHotEncoder(categorical_features=[Xs.columns.get_loc("Geography")])
    Xs = onehotencoder.fit_transform(Xs).toarray()
    Xs = Xs[:, 1:]

    # Splitting the data set into training and testing
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=0)

    # Feature Scaling
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    return [X_train, X_test, y_train, y_test]


def build_classifier(activation):
    classifier = Sequential()
    classifier.add(Dense(units=5, kernel_initializer='uniform', activation='softsign', input_dim=11))
    classifier.add(Dense(units=5, kernel_initializer='uniform', activation='softsign'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

    classifier.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


X_train, X_test, y_train, y_test = preprocessing()
classifier = KerasClassifier(build_fn=build_classifier, batch_size=30, epochs=500)
parameters = {
    'activation': ['relu', 'softsign', 'softplus', 'tanh']
}
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10,
                           n_jobs=-1)

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracies = grid_search.best_score_
print('{0} were the best parameters to use'.format(best_parameters))
print('{0} was the accuracies'.format(best_accuracies))


'''
Parameters selected were
optimizer=rmsprop
batch_size=30
epochs=500
activation=softsign
units=5
'''
