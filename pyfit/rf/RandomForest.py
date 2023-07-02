"""
GOAL: 
1) import random forest, and just call with train, test, n
2) cross validation version
3) scoring
4) should probably be a class so we can do confusion matrix and score plots etc
4.a) can we make these classes in lib that you inherit from and potentially override? 
"""
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy import stats
from random import randint
import seaborn as sn
from time import time

def randomForestRegression(X, y, n, sweep=False, cross_val=False):
    if sweep:
        n = rank_num_trees(X,y,n)[0][0]

    model = RandomForestRegressor(n_estimators=n)
    model.fit(X,y)
    return model

def randomForestClassification(X, y, n, sweep=False, cross_val=False):
    if sweep:
        n = rank_num_trees(X,y,n)[0][0]

    model = RandomForestClassifier(n_estimators=n)
    model.fit(X, y)
    return model


def rank_num_trees(X,y,max_n):
    trees = []
    for n in range(1,max_n+1):
        score = score_cross_val(X,y,n)[0]
        trees.append((n, score))

    trees = sorted(trees, lambda x: x[1], descending=True)
    return trees

# 10 fold cross-val 
def score_cross_val(X,y,n):
    RANDOM_SEED = randint(0, 100)
    kf = KFold(10, True, RANDOM_SEED)

    '''train/test split, train model, get errors'''
    residual_list = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model = RandomForestRegressor(n_estimators=n)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        score = model.score(y_pred, y_test)
        residual_list.append(score)

    '''return cv_err and stderr'''
    cv_score = np.mean(residual_list)
    stderr = stats.sem(np.array(residual_list))
    return cv_score, stderr

# 10-fold
def mse_cross_val(X, y, n):
    RANDOM_SEED = randint(0, 100)
    kf = KFold(10, True, RANDOM_SEED)

    '''train/test split, train model, get errors'''
    residual_list = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model = RandomForestRegressor(n_estimators=n)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
        residual = mean_squared_error(y_pred, y_test)
        residual_list.append(residual)
        
    '''return cv_err and stderr'''
    cv_score = np.mean(residual_list)
    stderr = stats.sem(np.array(residual_list))
    return cv_score, stderr
    

def mse_cross_val_with_pred_distribution(X, y, n):
    RANDOM_SEED = randint(0, 100)
    kf = KFold(10, True, RANDOM_SEED)

    '''train/test split, train model, get errors'''
    residual_list = []
    actuals = []
    predictions = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model = RandomForestRegressor(n_estimators=n)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
        residual = mean_squared_error(y_pred, y_test)
        residual_list.append(residual)
        for (act, pred) in zip(y_test.to_numpy(), y_pred.to_numpy()):
            actuals.append(act)
            predictions.append(pred)
        
    '''return cv_err and stderr'''
    cv_score = np.mean(residual_list)
    stderr = stats.sem(np.array(residual_list))
    return cv_score, stderr, actuals, predictions


def cross_val_plot(X,y,details_plot,min_n=1, max_n=201, step=10):
    cv_list = []
    stderr_list = []
    nums = []
    for n in range(min_n, max_n+1, step):
        start = time()
        cv, stderr = mse_cross_val(X,y,n)
        nums.append(n)
        cv_list.append(cv) 
        stderr_list.append(stderr) 
        end = time()
        print('model ', n, ' trained in ', round(end-start), ' seconds.')
    
    '''Plotting trees vs cross_validation_error'''     
    #plt.scatter(x_plot, y_plot, c='y', label=None)
    plt.errorbar(x=nums, y=cv_list, yerr=stderr_list, \
                 ecolor='skyblue', color='orange', marker='o',fmt='o', \
                 capsize=2)

    title, ylab, filename_full = details_plot
    print(cv_list, stderr_list)

    plt.title(title)
    plt.grid(axis='both')
    plt.yticks(ticks=[2,4,6,8,10,12,14,16,18,20,22,24]) #manually put in here for oracle
    plt.xlabel('# of Trees')
    plt.ylabel(ylab)
    plt.savefig(filename_full)
    plt.show()
    plt.clf()


def plot_confusion_matrix(y_test, y_predicted):
    cm = confusion_matrix(y_test, y_predicted)
    plt.figure(figsize=(10,7))
    sn.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()