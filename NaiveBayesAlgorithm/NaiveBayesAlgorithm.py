#Used dataset https://archive.ics.uci.edu/dataset/863/maternal+health+risk
#pip install pandas numpy
################ Main Program ##########################
import time
import random
import math
import sys
import pandas as pd
import numpy
from sklearn.model_selection import train_test_split

############################################## Main Loop ######################################
running = True
while (running):
    #print("Hello world")
    
    #Load .csv file     columns will have catergories with the right-most column being the target that the algorithm will be trying to classify
    #the first row will have the catergory's name
    dataset = pd.read_csv('Maternal Health Risk Data Set.csv')
    print(dataset.head())

    #a = dataset['Age']
    #print(a)

    #Split dataset, 80% training data, 20% for testing
    train, test = train_test_split(dataset, test_size=0.2)

    # Separate features and labels
    features_train = train.iloc[:, :-1]     #Category data
    label_train = train.iloc[:, -1]         #Type: low, med, or high risk
    
    features_test = test.iloc[:, :-1]       #Category data
    label_test = test.iloc[:, -1]           #Type: low, med, or high risk


    


    #classify using the Naive Bayes algorithm (should work for any amount of catergories)
    #priors = label_train.value_counts(normalize=True)  # Normalized counts to get probability
    
    #foreach row
        #foreach feature
            #calculate the probability of the feature n with the current data of every row we've seen so far

            #then classify the instance based on the probabilty found
            #Cm(x) = argmax(...)
            #numpy.argmax()



    #Compare against the test sets to see how well the algorithm performs
    #accuracy = accuracy_score(y_test, predictions)
    #print(f'Test set accuracy: {accuracy * 100:.2f}%')










    running = False;