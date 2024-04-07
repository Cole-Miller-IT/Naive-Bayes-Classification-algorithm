#Used dataset https://archive.ics.uci.edu/dataset/863/maternal+health+risk
#pip install pandas numpy
################ Main Program ##########################
from asyncio.windows_events import NULL
from itertools import count
import random
import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
from collections import defaultdict

############################################## Main Loop ######################################
running = True
while (running):
    #print("Hello world")
    

    # Generate random data because finding one is too difficult
    generate = False
    if (generate):
        # Define the options for each feature
        outlook_options = ['sunny', 'rain', 'overcast']
        temperature_options = ['hot', 'cold', 'mild']
        humidity_options = ['high', 'low', 'medium']
        wind_options = ['strong', 'weak']
        occupied_options = ['yes', 'no']

    
        numInstances = 1000
        data = {
            'Outlook': [random.choice(outlook_options) for _ in range(numInstances)],
            'Temperature': [random.choice(temperature_options) for _ in range(numInstances)],
            'Humidity': [random.choice(humidity_options) for _ in range(numInstances)],
            'Wind': [random.choice(wind_options) for _ in range(numInstances)],
            'Occupied': [random.choice(occupied_options) for _ in range(numInstances)]
        }

        # Create a DataFrame
        df = pd.DataFrame(data)

        # Save to a CSV file
        df.to_csv('myDataset.csv', index=False)


    #Load .csv file     columns will have catergories with the right-most column being the target that the algorithm will be trying to classify
    #the first row will have the catergory's name
    dataset = pd.read_csv('myDataset.csv')
    #print(dataset.head())
    


    #a = dataset['Age']
    #print(a)

    #Split dataset, 80% training data, 20% for testing
    train, test = train_test_split(dataset, test_size=0.2)

    # Separate features and labels
    #features_train = train.iloc[:, :-1]     #Category data
    #label_train = train.iloc[:, -1]         #Target/class: Yes or No
    
    #features_test = test.iloc[:, :-1]       #Category data
    #label_test = test.iloc[:, -1]           #Target/class: Yes or No

    print("train")
    print(train)
    #print(features_train)

    print("test")
    print(test)
    
    #classify using the Naive Bayes algorithm (should work for any amount of features and labels)
    #Create an array to hold the priors
    #priors = [label_train[Y] for Y in range(label_train)]
    
    #Create an array to hold the conditional probabilities
    #conditionalProbabilities = [features_train[X] for X in range(features_train) for Y in range(label_train)]

    #print(priors)
    
    #print(conditionalProbabilities)

    priorsDict = defaultdict(int)
    conditionalProbabilitiesDict = defaultdict(int)

    #foreach row of the train data
    myCount = 0
    for row in train.iterrows():
        #print("Current Row: ")
        #print(row)

        #Get each feature's value. The second value in train/test contains all of the
        #features and there values
        #print("current occupied value: ", end='')
        classValue = row[1].iloc[-1];   #gets the last column value for this row (which is the label/class)
        #print(classValue) 
      
        # Increment the count for this classValue
        priorsDict[classValue] += 1
    
        #print(priorsDict)
        #print(priorsDict['no'])    
        #print(priorsDict['yes'])




        #print("Columns in current row " + str(len(row[1])))
        

        #Loop through all of the current rows column values, skipping the last value (the class/label)
        i = 0
        while(i < (len(row[1]) - 1)):
            featureValue = row[1].iloc[i]   
            #print(featureValue)
            
            key = str(featureValue) + " | " + str(classValue)
            #print(key)

            #increment the counter for the associated feature given that it's part of the above label
            #i.e. P(sunny | yes) += 1 is updated/added to the dictionary
            conditionalProbabilitiesDict[key] += 1
            
            i += 1
        
        #print(conditionalProbabilitiesDict)

        
        #myCount += 1
        #if (myCount == 10):
            #break
    print("-------------------------------------------")
    print("Priors: ")
    print(priorsDict)
    
    print("-------------------------------------------")
    print("Cond. Prob.: ")
    print(conditionalProbabilitiesDict)
    

    #If any of the probabilities are 0 set them equal to 1 or something, because there is a chance that they can occur     


    
    #Compare against the test sets to see how well the algorithm performs
    #foreach instance/row in the test set
        #determine the proability that it is one of the labels
        #Cm(x) = argmax(...)
        #numpy.argmax()

        #save the result for accuracy later

    #accuracy = accuracy_score(y_test, predictions)
    #print(f'Test set accuracy: {accuracy * 100:.2f}%')










    running = False;