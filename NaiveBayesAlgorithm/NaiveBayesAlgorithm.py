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

#returns a string to show conditional probability --> i.e. sunny | yes
def probString(valueOne, valueTwo):
    result = str(valueOne) + " | " + str(valueTwo)
    return result;


############################################## Main Loop ######################################
running = True
while (running):
    disclaimer = """
    Disclaimer: If using the randomly generated data then the naive assumption will actually be true, all values are independant of one another.
    That will probably result in a low accuracy. For example a high temperature rating won't mean anything. It would be by chance that 
    a higher temperature (or a specific feature) would actually skew towards a specific classification. That being said the algorithm should work 
    correctly for real datasets."""
    print(disclaimer)
    
    print("P.S. It was taking to long to find a data set the worked.")
    

    # Generate random data because finding one is too difficult
    generate = False
    if (generate):
        # Define the options for each feature
        outlook_options = ['sunny', 'rain', 'overcast']
        temperature_options = ['hot', 'cold', 'mild']
        humidity_options = ['high', 'low', 'medium']
        wind_options = ['strong', 'weak']
        occupied_options = ['yes', 'no']

    
        numInstances = 100      ###NOTE: How many rows to generate for the .csv###
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
        df.to_csv('myDataset.csv', index=False) #Will create a .csv in the same folder as this program


    #Load .csv file     columns will have catergories with the right-most column being the target that the algorithm will be trying to classify
    #the first row will have the catergory's name
    dataset = pd.read_csv('myDataset.csv')          ###NOTE: Change to your specific .csv###

    #Split dataset, 80% training data, 20% for testing
    train, test = train_test_split(dataset, test_size=0.2)
    totalTestCases = len(test)
    totalRows = len(train)
    
    #print("train")
    #print(train)

    #print("test")
    #print(test)
    
    
    #Create 2 generic dictionaries that will keep a count of the classes (priors) and the count of conditional probabilities 
    #Both the classifications and features could be any length
    #e.g. How many "yes"'s were counted in the training set = priorDict[yes]
    priorsDict = defaultdict(int)
    conditionalProbabilitiesCountDict = defaultdict(int)

    #foreach row of the training data
    for row in train.iterrows():
        #print("Current Row: ")
        #print(row)

        #Get each feature's value. The second value in train/test contains all of the
        #features and their values as a tuple
        classValue = row[1].iloc[-1];   #gets the last column value for this row (which is the label/class)
      
        # Increment the count for this classValue
        priorsDict[classValue] += 1
 

        #Loop through all of the current rows column values, skipping the last value (the class/label)
        i = 0
        while(i < (len(row[1]) - 1)):
            featureValue = row[1].iloc[i]   
            #print(featureValue)
            
            key = probString(featureValue, classValue)
            #print(key)

            #increment the counter for the associated feature given that it's part of the above label
            #i.e. count of (sunny | yes) += 1 is updated/added to the dictionary
            conditionalProbabilitiesCountDict[key] += 1
            
            i += 1

    #print("Observed training data")
    #print("-------------------------------------------")
    #print("Priors: ")
    #print(priorsDict)
    
    #print("-------------------------------------------")
    #print("Cond. Prob. Count: ")
    #print(conditionalProbabilitiesCountDict)
    
    
    #Compare against the test sets to see how well the algorithm performs
    #foreach instance/row in the test set
    correctPredictions = 0
    conditionalProbabilitiesDict = defaultdict(float)
    unnormalizedDict = defaultdict(float)
    normalizedDict = defaultdict(float)
    for row in test.iterrows(): #the row would be x in Cm(x) = argmax(...)
        #print("Current Test Row: ")
        #print(row)

        #Classify using the Naive Bayes algorithm (should work for any amount of features and labels)
        realClassification = classValue = row[1].iloc[-1];
    
        #Loop through all of the current rows column values, skipping the last value (the class/label)
        i = 0
        while(i < (len(row[1]) - 1)):
            featureValue = row[1].iloc[i]  
            
            #Calculate probability and store for later (for all of the possible classifications)
            #P(sunny | yes) = (# of (sunny | yes)'s counted during training   /   # of yes's counted during training)
            for classification in priorsDict:
                keyConditional = probString(featureValue, classification)
                keyClass = classification
                conditionalProbabilitiesDict[keyConditional] += conditionalProbabilitiesCountDict[keyConditional] / priorsDict[classification]
                
                #Calculate un-normalized final probabilities
                if (unnormalizedDict[classification] == 0):
                    #first value
                    unnormalizedDict[classification] = priorsDict[classification] / totalRows
                else:
                    #other values
                    unnormalizedDict[classification] = unnormalizedDict[classification] * conditionalProbabilitiesDict[keyConditional]
                
            i += 1
    
        #save the result for accuracy later
        #print("Computed condtional probabilities: ")
        #print(conditionalProbabilitiesDict)
        
        #print("Unnormailized final prob: ")
        #print(unnormalizedDict)
 

        #Normalize the values
        for classification in priorsDict:
            denominator = 0
            for prob in unnormalizedDict:
                denominator += unnormalizedDict[prob]

            normalizedDict[classification] = unnormalizedDict[classification] / denominator
        
        #print("normailized final prob: ")
        #print(normalizedDict)     
        
        #Choose the highest probability   Cm(x) = argmax(...)
        highest = 0
        highestString = ""
        for prob in normalizedDict:
            if normalizedDict[prob] >= highest:
                highest = normalizedDict[prob]
                highestString = prob
                


        #print(highest)
        #print(highestString)
        
        #verify if it was correct or not
        if highestString == realClassification:
            correctPredictions += 1
            #print("correct prediction")

        #break
        
    
    accuracy = correctPredictions / totalTestCases
    print("")
    print(f'Test set accuracy: {accuracy * 100:.2f}%')



    






    running = False;