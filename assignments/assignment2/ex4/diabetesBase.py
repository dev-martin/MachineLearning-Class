import matplotlib.pyplot as pyplot
import pylab as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neural_network  import MLPRegressor


def rmse(predicted, actual):
    diff = predicted - actual;
    squared = np.square(diff);
    summation = np.sum(squared);
    division = summation/len(actual)
    return np.sqrt(division);
    
uid = "116928472"
name = "Martin Iglesias"
seed = 13

#Creating results file
results = open("diabetesBaseResults.txt" , "w")

#Writting name and UID
results.write("UID: "+ uid + "      Name: "+ name + "       Seed: "+ str(seed) +"\n")

#Generating sample data
data, target = load_diabetes(return_X_y=True)
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=.25, train_size=.75)

#Error BackProp w/Regression Learning
mlp = MLPRegressor(max_iter=200, random_state=13);

####Before Training(After 1 epoch)
mlp.partial_fit(data_train, target_train);
#RMSE Train Data
predict_train0 = mlp.predict(data_train);
rmse_train0 = rmse(target_train, predict_train0);

#RMSE Test Data
predict_test0 = mlp.predict(data_test);
rmse_test0 = rmse(target_test, predict_test0);

####After Training
mlp.fit(data_train, target_train);
#RMSE Train Data
predict_train1  = mlp.predict(data_train);
rmse_train1 = rmse(target_train, predict_train1);

#RMSE Test Data
predict_test1  = mlp.predict(data_test);
rmse_test1 = rmse(target_test, predict_test1);


#Weight values
weights = mlp.coefs_;
#Bias values
biases = mlp.intercepts_;

#Write Params down
results = open("diabetesBaseResults.txt" , "a")
results.write( "----PARAMS----\n"); 
results.write(str(mlp.get_params()) + "\n"+ "\n"); 

#Write RMSE down
results = open("diabetesBaseResults.txt" , "a")
results.write( "----RMSE----\n"); 
results.write("            Train Data            |    Test Data  "+ "\n"); 
results.write("Pre-Train    "+str(rmse_train0) +"      " +str(rmse_test0)+ "\n"); 
results.write("Post-Train   "+str(rmse_train1) +"      " +str(rmse_test1)+ "\n"+ "\n"); 

#Write down target values and actual values for TEST data
results = open("diabetesBaseResults.txt" , "a")
results.write( "----TARGETS----PREDICTED VALS----\n");
for elem  in range(len(target_test)):
    results.write(str(target_test[elem]) +"              "+ str(predict_test1[elem]) + "\n"); 
results.write("\n");

#Write learned weights ad bias values
results = open("diabetesBaseResults.txt" , "a")

results.write( "----WEIGHTS----\n");
for elem  in range(len(weights)):
    results.write(str(weights[elem]) + "\n"); 

results.write( "----BIASES----\n");
for elem  in range(len(biases)):
    results.write(str(biases[elem]) + "\n"); 
