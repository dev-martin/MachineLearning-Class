import matplotlib.pyplot as pyplot
import pylab as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.neural_network  import MLPClassifier

noises = [0.01, 0.1, 0.2, 0.5, 2.0]
uid = "116928472"
name = "Martin Iglesias"
seed = 13

#Creating results file
results = open("moonResults.txt" , "w")

#Writting name and UID
results.write("UID: "+ uid + "      Name: "+ name + "       Seed: "+ str(seed) +"\n")

#Writting column titles
results = open("moonResults.txt" , "a")
results.write("Train #  |   Test #  |   Noise   |   EBP_Train_Accuracy   |    EBP_Test_Accuracy\n")

for noise in noises:

    #Generating sample data
    samples, classes = make_moons(n_samples=500, noise=noise, random_state=seed)
    samples_train, samples_test, classes_train, classes_test = train_test_split(samples, classes, test_size=.25, train_size=.75)

    train_size = len(samples_train)
    test_size = len(samples_test)

    #Error BackProp Learning
    mlp = MLPClassifier(hidden_layer_sizes=(6,), activation='tanh', solver='lbfgs', random_state=13);
    
    #EBP Training
    mlp.fit(samples_train,classes_train);

    #EBP Accuracy
    test_accuracy = mlp.score(samples_test,classes_test);
    train_accuracy = mlp.score(samples,classes);

    #Write all data down to results.txt 
    results = open("moonResults.txt" , "a")
    results.write(str(train_size)     + "           "
            +     str(test_size)      + "    "
            +     str(noise)          + "                  "
            +     str(train_accuracy) + "                  "
            +     str(test_accuracy)  + "\n"
            ) 

