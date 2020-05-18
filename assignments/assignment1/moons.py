import matplotlib.pyplot as pyplot
import pylab as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn import linear_model
from sklearn.model_selection import train_test_split

noises = [0.01, 0.1, 0.2, 0.5, 2.0]
uid = "116928472"
name = "Martin Iglesias"
seed = 13

#Creating results file
results = open("results.txt" , "w")

#Writting name and UID
results.write("UID: "+ uid + "      Name: "+ name + "       Seed: "+ str(seed) +"\n")

#Writting column titles
results = open("results.txt" , "a")
results.write("Train #  |   Test #  |   Mean            |   StandardDev             |   Noise   |   First_Weight        |     Second_Weight         |   Intercept   |   Train_Accuracy   |    Test_Accuracy\n")

for noise in noises:

    #Generating sample data
    samples, classes = make_moons(n_samples=500, noise=noise, random_state=seed)
    samples_train, samples_test, classes_train, classes_test = train_test_split(samples, classes, test_size=.25, train_size=.75)

    train_size = len(samples_train)
    test_size = len(samples_test)

    samples_stdev = np.std(samples)
    samples_mean = np.mean(samples)

    #Creating perceptron 
    perceptron = linear_model.Perceptron(max_iter=100, eta0=0.1)

    #Training on train_data
    perceptron.fit(samples_train,classes_train)

    #Test data accuracy after training
    test_accuracy = perceptron.score(samples_test,classes_test )

    #Train data accuracy after training
    train_accuracy = perceptron.score(samples,classes)

    #Intercept, 1st weight, 2nd weight
    intercept = perceptron.intercept_.item(0)
    weight1 = perceptron.coef_.item(0)
    weight2 = perceptron.coef_.item(1)

    #Write all data down to results.txt 
    results = open("results.txt" , "a")
    results.write(str(train_size) + "           " +  str(test_size)  + "    " + str(round(samples_mean, 16)) + "       "+ str(round(samples_stdev, 15)) + "           "+ str(noise) + "         "+ str(weight1) + "         "+ str(weight2) + "         "+ str(round(intercept, 4)) + "             "+ str(train_accuracy) + "                  "+ str(test_accuracy) + "\n") 

    # Plot data set in 2D space
    pyplot.ion()
    pyplot.figure()
    class0 = samples[np.where(classes < 0.5)]
    class1 = samples[np.where(classes > 0.5)]
    pyplot.plot(class0[:,0],class0[:,1],'bo')
    pyplot.plot(class1[:,0],class1[:,1],'ro')
    pyplot.xlabel('samples[0] values')
    pyplot.ylabel('samples[1] values')
    pyplot.title('2D for Noise= '+ str(noise))
    pyplot.show(block=True)

