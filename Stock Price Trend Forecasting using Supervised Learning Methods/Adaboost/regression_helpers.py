# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC, SVR
from sklearn.qda import QDA
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.grid_search import GridSearchCV
# from Neural_Network import NeuralNet

def load_dataset(path_directory, symbol): 
    
    path = os.path.join(path_directory, symbol)

    out = pd.read_csv(path, index_col=2, parse_dates=[2])
    out.drop(out.columns[0], axis=1, inplace=True)

    return [out]    

def count_missing(dataframe):
    
    return (dataframe.shape[0] * dataframe.shape[1]) - dataframe.count().sum()

    
def addFeatures(dataframe, adjclose, returns, n):
   
    
    return_n = adjclose[9:] + "Time" + str(n)
    dataframe[return_n] = dataframe[adjclose].pct_change(n)
    
    roll_n = returns[7:] + "RolMean" + str(n)
    dataframe[roll_n] = pd.rolling_mean(dataframe[returns], n)

    exp_ma = returns[7:] + "ExponentMovingAvg" + str(n)
    dataframe[exp_ma] = pd.ewma(dataframe[returns], halflife=30)
    
def mergeDataframes(datasets):
   
    return pd.concat(datasets)

    
def applyTimeLag(dataset, lags, delta):
    
    maxLag = max(lags)

    columns = dataset.columns[::(2*max(delta)-1)]
    for column in columns:
        newcolumn = column + str(maxLag)
        dataset[newcolumn] = dataset[column].shift(maxLag)

    return dataset.iloc[maxLag:-1, :]
   
def prepareDataForClassification(dataset, start_test):
    
    le = preprocessing.LabelEncoder()
    
    dataset['UpDown'] = dataset['Return_Out']
    dataset.UpDown[dataset.UpDown >= 0] = 'Up'
    dataset.UpDown[dataset.UpDown < 0] = 'Down'
    dataset.UpDown = le.fit(dataset.UpDown).transform(dataset.UpDown)
    
    features = dataset.columns[1:-1]
    X = dataset[features]    
    y = dataset.UpDown    
    
    X_train = X[X.index < start_test]
    y_train = y[y.index < start_test]    
    
    X_test = X[X.index >= start_test]    
    y_test = y[y.index >= start_test]
    
    return X_train, y_train, X_test, y_test    

def prepareDataForModelSelection(X_train, y_train, start_validation):
    
    X = X_train[X_train.index < start_validation]
    y = y_train[y_train.index < start_validation]    
    
    X_val = X_train[X_train.index >= start_validation]    
    y_val = y_train[y_train.index >= start_validation]   
    
    return X, y, X_val, y_val

  
def performClassification(X_train, y_train, X_test, y_test, method, parameters={}):
    

    print('Performing ' + method + ' Classification...')
    print('Size of train set: ', X_train.shape)
    print('Size of test set: ', X_test.shape)
    print('Size of train set: ', y_train.shape)
    print('Size of test set: ', y_test.shape)
    

    classifiers = [
    
        AdaBoostRegressor(),
        AdaBoostClassifier(**parameters)(),
        GradientBoostingClassifier(n_estimators=100),
    ]

    scores = []

    for classifier in classifiers:
        scores.append(benchmark_classifier(classifier, \
            X_train, y_train, X_test, y_test))

    print(scores)

def benchmark_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    return accuracy

# REGRESSION
    
def getFeatures(X_train, y_train, X_test, num_features):
    ch2 = SelectKBest(chi2, k=5)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    return X_train, X_test

def performRegression(dataset, split, symbol, output_dir):
   

    features = dataset.columns[1:]
    index = int(np.floor(dataset.shape[0]*split))
    train, test = dataset[:index], dataset[index:]
    print('Size of train set: ', train.shape)
    print('Size of test set: ', test.shape)

    out_params = (symbol, output_dir)

    output = dataset.columns[0]

    predicted_values = []

    classifiers = [
        BaggingRegressor(),
        AdaBoostRegressor(),
        GradientBoostingRegressor(),
    ]

    for classifier in classifiers:

        predicted_values.append(benchmark_model(classifier, \
            train, test, features, output, out_params))

    maxiter = 1000
    batch = 150

    # classifier = NeuralNet(50, learn_rate=0.1)

    # predicted_values.append(benchmark_model(classifier, \
    #     train, test, features, output, out_params, \
    #     fine_tune=False, maxiter=maxiter, SGD=True, batch=batch, rho=0.9))
    

    print('-'*100)

    mean_squared_errors = []

    r2_scores = []
    # i = 0
    for pred in predicted_values:
        # print (pred)
        # if i > 10:
        #     break
        # i+=1;
        mean_squared_errors.append(mean_squared_error(test[output].as_matrix(), \
            pred))
        r2_scores.append(r2_score(test[output].as_matrix(), pred))

    print ("MSE : ",mean_squared_errors)

    return mean_squared_errors, r2_scores

def benchmark_model(model, train, test, features, output, \
    output_params, *args, **kwargs):
    '''
        Performs Training and Testing of the Data on the Model.
    '''

    print('-'*80)
    model_name = model.__str__().split('(')[0].replace('Regressor', ' Regressor')
    print(model_name)

    
    symbol, output_dir = output_params

    model.fit(train[features].as_matrix(), train[output].as_matrix(), *args, **kwargs)
    predicted_value = model.predict(test[features].as_matrix())

    plt.plot(test[output].as_matrix(), color='g', ls='-', label='Actual Value')
    plt.plot(predicted_value, color='b', ls='--', label='predicted_value Value')

    plt.xlabel('Number of Set')
    plt.ylabel('Output Value')

    plt.title(model_name)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, str(symbol) + '_' \
        + model_name + '.png'), dpi=100)
    #plt.show()
    plt.clf()

    return predicted_value
