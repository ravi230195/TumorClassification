# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 12:03:13 2019

@author: ravik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

'''
@ Function to calculate the sigmod of a vector.
@ Returns a vector, its nothing but pridicted value in logistic regression.
'''
def sigmod(y):
    return 1/(1 + np.exp(-y))

''' 
@ Function to  Matrix multiplication of to matrix of dimensions
@ (m,n) and (n,k) the output is a matix of dimensions (m,k)
'''
def matrixMultiply(A,B):
    #print("dimensions of A{} and B{}".format(A.shape,B.shape))
    return np.matmul(A, B)

'''
@ Function to plot multiple data in onr plot.
@ X is a list of values for each plot and similarly Y
'''
def plotMultiple(X, Y, color, labels):
    plt.figure(figsize=(10,10))
    plt.ylim(0,1)
    X_scale = []
    for i in range(1,21):
        X_scale.append(100*i)
    plt.xticks(X_scale)
    plt.ylabel(labels[1])
    plt.xlabel(labels[0])
    plt.text(600, 0.5, 'Blue: Training Loss', horizontalalignment='center',verticalalignment='center', fontsize=18)
    plt.text(600, 0.58, 'Red: Validation Loss', horizontalalignment='center',verticalalignment='center', fontsize=18)
    for i in range(0, len(Y)):
        plt.plot(X, Y[i], color[i])
    plt.show()

def plot(X, Y, color, labels):
    plt.figure(figsize=(10,10))
    plt.ylim(0.4,1)
    plt.ylabel(labels[1])
    plt.xlabel(labels[0])
    plt.text(600, 0.5, 'Acuuracy per epoch', horizontalalignment='center',verticalalignment='center', fontsize=18)
    plt.plot(X, Y, color)
    plt.show()
    
'''
@ Function to Convert the Labels to integers for logistic regression
'''
def encodingTextToLabels(X, labels):
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    X = le.transform(X)
    return X

'''
   @Function to calculate FP, FN, TP, TN
   FP = False Positive
   FN = False Negative
   TP = True Positive
   TN = True Negative
'''
def calculateConfusionMatrix(Y_actual, Y_pridicted):
    FP = FN = TP = TN = 0 
    for i in range(0,Y_actual.shape[0]):
        if Y_actual[i] == 0:
            if Y_pridicted[i] == 0:
                TN +=1
            else:
                FN +=1
        else:
            if Y_pridicted[i] == 1:
                TP +=1
            else:
                FP +=1
    return FP, FN, TP, TN

'''
   @ Function to calculate the Accuarcy, recall, precision
'''
def calculateMetrix(Y_actual, Y_pridicted, isOnlyAccuracyRequired = False):
    FP, FN, TP, TN = calculateConfusionMatrix(Y_actual, Y_pridicted)
    accuracy = (TN + TP)/(TN + FN + TP + FP)
    if (isOnlyAccuracyRequired):
        return accuracy
    recall = (TP)/(TP + FN)
    precision = (TP)/(TP + FP)
    print("Accuracy is [{}]".format(accuracy))
    print("recall is [{}]".format(recall))
    print("precision is [{}]".format(precision))


'''
@ Function to calculate the costfunction
@ Both y_actual, y_pridicted are (569,1) array for tainingSet
@ summazion over i 1 to m (y log(h(x))) is nothing but y *transpose log(h(x))
@ simplarly over i 1 to m ((1-y)*log((1-h(x))) is nothing but (1- y)*transpose log(1-h(x))
'''
def costFunction(y_actual, y_pridicted):
    one_array = np.ones((y_actual.shape[0], 1))
    one_minus_actual = one_array - y_actual
    one_minus_pridicted = one_array - y_pridicted
    log_pridicted = np.log(y_pridicted)
    one_minus_log_pridicted = np.log(one_minus_pridicted)
    summation = matrixMultiply(y_actual.transpose(), log_pridicted) + matrixMultiply(one_minus_actual.transpose(),one_minus_log_pridicted)
    return np.asscalar((-summation/(y_actual.shape[0])))


'''
   @let inital weights be initialized to 0.
   @ QX is a (569, 1) matrix for trainig and accordingly for other
   @pass X*Q(transpose) to sigmod function to get hypothosis function   
'''
def gradientDesent(X, Y, X_validate, Y_validate, alpha=0.1):
    print("\nFor alpha {}".format(alpha))
    cost_function_training = []
    cost_function_validation = []
    accuracy_list = []
    interations = []
    ' @let inital weights be initialized to 0.'
    weights = np.zeros((1, X.shape[1]))
    for i in range(0,2001):
        QX = matrixMultiply(X, np.transpose(weights))
        '''@ QX is a (569, 1) matrix for trainig and accordingly for other
           @pass X*Q(transpose) to sigmod function to get hypothosis function
        '''
        y_pridicted = sigmod(QX)
        '''
           @ Calculating loss for the training data.
           @ Using the old weights.
        '''
        cost_function_training.append(costFunction(Y, y_pridicted))
        interations.append(i)
        '@ derivative of cost function is summation(H(x) -y)xi followed by mean (X.shape[0])'
        error = y_pridicted - Y
        derivative_weights = matrixMultiply(np.transpose(error.reshape(error.shape[0],1)), X) / (X.shape[0])
        '''# Multiply with learning rate (0.05),
           # derivative_weights is of dimension (1,31) and old weights also in (1,31)
        '''
        new_weights = weights - alpha*derivative_weights
        '''
           @ Calculating loss on validation  data Using the newly obtained weights.
           @ And also calculate the accuary to print the graph
        '''
        QX_NewWeights = matrixMultiply(X_validate, np.transpose(new_weights))
        y_Newpridicted = sigmod(QX_NewWeights)
        cost_function_validation.append(costFunction(Y_validate, y_Newpridicted))
        accuracy_list.append(calculateMetrix(Y_validate, mapTheOututToTwoModel(y_Newpridicted), True))
        weights = new_weights
    '''
       @ Plot Accuracy vs Epoch
       @ Plot loss-function vs Epoch (For Tarining and validation) 
    '''
    plotMultiple(interations, [cost_function_training, cost_function_validation], ["b","r"],["epoach", "costFunction"])
    plot(interations, accuracy_list, "r", ["epoach", "accuracy"])
    return weights


'''
   @ Function convets the continues values to wither 0, 1 class
   @ if y > 0.5 then class 1 
   @ if y=< 0.5 then class 0 
'''
def mapTheOututToTwoModel(y_pridicted):
    for i in range(0, y_pridicted.shape[0]):
        if y_pridicted[i] > 0.5:
            y_pridicted[i] = 1
        else:
            y_pridicted[i] = 0
    return y_pridicted



############ MAIN FUNCTION ####################
if __name__ == "__main__":
    df = pd.read_csv(r'wdbc.csv', header=None)
    
    '@ Converting text to labels'
    y = encodingTextToLabels(df[1], ['B','M'])
    y_actual = y.reshape((y.shape[0], 1))
    
    '@ Remove the patient ID and Label column'
    independent_data = df.drop(columns = [1,0])
    
    '@ Transforming X matrix by adding Q0 parameter bias parameter'
    rows = independent_data.shape[0]
    one_series = np.ones(rows)
    independent_data.insert(0, "Q0", 1)
    columns = independent_data.shape[1]
    df_np = independent_data.to_numpy()
    
    '@ Normalize the data the data before spliting'
    normalized_data = df_np / (df_np.max(axis=0))
    
    '''
    @ First plit the tratining data to 80% and test_total to 20%
    @ Then again split the test_total equally into validation data and test data
    @ Trainig is 80% validation is 10% and test is 10% 
    '''
    X_train, X_test_total, y_train, y_test_total = train_test_split(normalized_data, y_actual, test_size=0.2)
    X_validation, X_test, y_validation, y_test = train_test_split(X_test_total, y_test_total, test_size=0.5)
    print("length of training data {}, testing data {}, validation data{}".format(len(X_train), len(X_test),len(X_validation)))
    
    '''
    @ For different values of alpha (0.1,0.3,0.5,0.7,0.9)
    @ calculate ConfusionMatrix and Accuracy, Recall, precision
    '''
    for i in np.arange(0.1,1,0.2):
        weights = gradientDesent(X_train, y_train, X_validation, y_validation, alpha=i)
        QX_pridicted = matrixMultiply(X_test, np.transpose(weights))
        y_pridicted = sigmod(QX_pridicted)
        mapTheOututToTwoModel(y_pridicted)
        calculateMetrix(y_test, y_pridicted)
        print("\n")



######################################## END######################################

################### TEST CASES FOR SANITY TESTING #################
#print(y_pridicted.astype(int) - y_train_validation)
#plot(interations1,cost_function1, "interations","cost_function", 'r')
#plot.show()
#print(np.transpose(zero_series.reshape(zero_series.shape[0],1)).shape)
#Q traspose X

#print(np.transpose((y_pridicted - y_actual)).shape)
##### Matrix Mul Test###########
#weights = np.random.rand(31)
#print(weights)
#output = np.matmul(df_np_normalized, np.transpose(weights))
#print(output.shape)


#Sigmod TestCase
#test_sigmod = np.array([1,2,3,4,4,5,5,6])
#sigmodial_values = sigmod(test_sigmod)
#print((test_sigmod).size)
#print(sigmodial_values)
#print(sigmodial_values.size)
#print((test_sigmod).shape)
#print(sigmodial_values.shape)
