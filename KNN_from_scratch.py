"""
KNN Classifier
Yuliya Akchurina
"""

#!/usr/bin/env python3
# -- coding: utf-8 --


#region import libraries
import pandas as pd
import statistics as stat
import numpy as np
from datetime import datetime
from scipy import stats

#endregion import libraries

#== start run time clock
start_time = datetime.now()

#region Declarations

#== Data input files
fn_trainData = "spam_train.csv"
fn_testData = "spam_test.csv"

#== List of K values to test
K_list = [1, 5, 11, 21, 41, 61, 81, 101, 201, 401]

#== number of rows in prediction array to print
num_pred_rows = 50

#endregion Declarations

#region Functions

def pairwise_dists(x_test, x_train):
    """ Computing pairwise distances using memory-efficient
        vectorization.

        Parameters
        ----------
        x_train : numpy.ndarray, shape=(M, D)
        x_test : numpy.ndarray, shape=(N, D)

        Returns
        -------
        numpy.ndarray, shape=(M, N)
            The Euclidean distance between each pair of
            rows between `x_train` and `x_test`."""
    dists = -2 * np.matmul(x_train, x_test.T)
    dists +=  np.sum(x_train**2, axis=1)[:, np.newaxis]
    dists += np.sum(x_test**2, axis=1)
    return  np.sqrt(dists)

def most_common(series): 
    return(stat.mode(series)) 

def predict_scores(arr_sorted_spam_scores, klist):
    """ predict spam score based on majority for each K """

    k_preds = np.empty((0,len(klist)), int)

    for row in arr_sorted_spam_scores:

        row_prediction = []
        for k in klist:
            prediction = most_common(row[:k])
            row_prediction = np.append(row_prediction, int(prediction))

        k_preds = np.vstack((k_preds, np.array((row_prediction))))

    return k_preds

def calculate_accuracy (arr_preds, test_spam_scores):
    """ calculate accuracy for each K """
    K_accuracies = []

    for row in arr_preds.T:
        matches = np.count_nonzero(row == test_spam_scores)
        accuracy = (matches/len(row))*100
        K_accuracies = np.append(K_accuracies, accuracy)

    return K_accuracies

def KNN_Classify (x_train, y_train, x_test, y_test, klist):
    #== get Euclidian distance between train and test data points
    distance = pairwise_dists(x_train, x_test)

    #== sort disctance array ascending by row and return array of indexes
    sorted_distance_indexes = distance.argsort(axis=1)

    #== assign train spam scores to sorted distance array by index 
    sortedarray = y_train[sorted_distance_indexes]

    #== predict spam score based on majority for each K
    K_predictions = predict_scores(sortedarray, K_list)

    #== calucalate accuracy of predictions
    arr_K_Accuracy = calculate_accuracy(K_predictions, y_test)

    return arr_K_Accuracy, K_predictions

def print_accuracy_report(arr_k_accuracy, klist):

    Report = np.vstack((klist, arr_k_accuracy))
    Report = Report.T

    for row in Report:
        print(f"K-{int(row[0])}: {str(round(row[1], 2))}%")
    print(f"\n")

def print_prediction_rows(arr_pred, arr_test, num_rows, klist):
    
    df = pd.DataFrame(data=(arr_pred[:num_rows,:]), index=((arr_test.iloc[:, 0]).head(num_rows)), columns=klist)
    df.replace(to_replace = 1, value ="spam", inplace=True)
    df.replace(to_replace = 0, value ="no", inplace=True)
    print(f"K_predictions, z normalized, first {num_rows} rows:\n\n {df}")

#endregion Functions

#region import and format data

#== import data into dataframes from the csv files, no changes 
trainData = pd.read_csv(fn_trainData)
testData = pd.read_csv(fn_testData)

#== convert to numpy arrays and remove label rows and columns
x_trainArray = np.array(trainData.iloc[:,0:-1])     # train data
y_trainArray = np.array(trainData.iloc[:,-1])       # tain spam score
x_testArray = np.array(testData.iloc[:,1:-1])       # test data
y_testArray = np.array(testData.iloc[:,-1])         # test spam score

#endregion import and format data

#region MAIN

#===> Q:a - Process data without normalizing features

#== calculate KNN accuracy for given K List
arr_K_Accuracy, arr_predictions = KNN_Classify (x_trainArray, y_trainArray, x_testArray, y_testArray, K_list)

#== print accuracy report
print(f"\nKNN accuracy report without normalizing features:\n")
print_accuracy_report(arr_K_Accuracy, K_list)


#===> Q:b - Process data with z normalized features

#== Z-Normailze data sets
x_train_norm = stats.zscore(x_trainArray)
x_test_norm = stats.zscore(x_testArray)

#== calculate KNN accuracy for given K List
arr_K_Accuracy, arr_predictions = KNN_Classify (x_train_norm, y_trainArray, x_test_norm, y_testArray, K_list)

#== print accuracy report
print(f"KNN accuracy report with z normalized features:\n")
print_accuracy_report(arr_K_Accuracy, K_list)

#== Q:c - Print first 50 preditions of z normalized test data set
print_prediction_rows(arr_predictions, testData, num_pred_rows, K_list)


#== report run time
time_elapsed = datetime.now() - start_time 
print('\nTime elapsed (hh:mm:ss.ms) {}\n'.format(time_elapsed))

#endregion MAIN



