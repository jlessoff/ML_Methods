import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


# Question 1: Load wine data set and prepare for ML.  Keep 30% of data set for final evaluation
#import data
wines=pd.read_csv('winequality-red.csv',delimiter=',',skiprows=1)
#explanatory variable
wines_X=wines[wines.columns[1:12]]
#target variable
wines_Y=wines['quality']



#split test, train
from sklearn.model_selection import train_test_split
X_tr, X_test, Y_tr, Y_test=train_test_split(wines_X,wines_Y,
                                                    train_size=0.7, random_state=42)
# Question 2: create a default decision tree regressor, fit it to the learning data, and show the prediction performances based on test set.  using MSE and graphical representation
#Fitting model using training data
regr_1=DecisionTreeRegressor()
regr_1.fit(X_tr,Y_tr)
# Predict
y_1 = regr_1.predict(X_test)
print(y_1)
# Plot the results
#print(y_1)
#accuracy check using MSE
mse = mean_squared_error(Y_test, y_1)
print(mse)


#is this overfitting?  a good way to check: compare error from training set and test set.

y_training_prediction=regr_1.predict(X_tr)
mse_train = mean_squared_error(Y_tr, y_training_prediction)
print(mse_train)
###Yes, it is overfitting because the training MSE is zero and the test MSE is high###
print("Yes, it is overfitting becaus the training MSE is zero and the test MSE is high")

#graphical representation
X_AXIS = range(len(Y_test))
plt.plot(X_AXIS, Y_test, label="original")
plt.plot(X_AXIS, y_1, label="predicted")
plt.title("Wine test and predicted data")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()


##Create a loop that learns a series of trees with larger values for min_samp and collect the MSE of trees on learning set and test set
mse_test=[]
mse_learn=[]
for i in range(2,100):
    regr_1 = DecisionTreeRegressor(min_samples_split=i)
    regr_1.fit(X_tr, Y_tr)
    # Predict
    y_1 = regr_1.predict(X_test)
    # accuracy check using MSE
    mse1 = mean_squared_error(Y_test, y_1)
    mse_test.append(mse1)
    y_training_prediction = regr_1.predict(X_tr)
    mse_train = mean_squared_error(Y_tr, y_training_prediction)
    mse_learn.append(mse_train)
#printing MSE for test and learning set

print(mse_test)
print(mse_learn)
plt.plot(mse_test)
plt.show()

plt.plot(mse_learn)
plt.show()

#
mse_test=[]
mse_learn=[]
for i in range(2,40):
    regr_1 = DecisionTreeRegressor(max_depth=i)
    regr_1.fit(X_tr, Y_tr)
    # Predict
    y_1 = regr_1.predict(X_test)
    # Plot the results
    # accuracy check using MSE
    mse1 = mean_squared_error(Y_test, y_1)
    mse_test.append(mse1)
    y_training_prediction = regr_1.predict(X_tr)
    mse_train = mean_squared_error(Y_tr, y_training_prediction)
    mse_learn.append(mse_train)
print(mse_test)
plt.plot(mse_test)
plt.show()

print(mse_learn)
plt.plot(mse_learn)
plt.show()




path=regr_1.cost_complexity_pruning_path(X_tr,Y_tr)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
acc_test=[]
acc_train=[]
for alpha in range(0,len(ccp_alphas)):
    clf= DecisionTreeRegressor(random_state=42, ccp_alpha= ccp_alphas[alpha])
    clf.fit(X_tr, Y_tr)
    tree_prediction_test = clf.predict(X_test)
    accuracy_test=mean_squared_error(Y_test,tree_prediction_test)
    #print(ccp_alphas[alpha])
    acc_test.append(accuracy_test)

    tree_prediction_train = clf.predict(X_tr)
    accuracy_train=mean_squared_error(Y_tr,tree_prediction_train)
    acc_train.append(accuracy_train)
plt.plot(acc_test)
plt.show()

plt.plot(acc_train)
plt.show()
##PRUNING AND DIRECTLY STOPPING THE TREE WILL HAVE SIMILAR RESULTS.  HOWEVER, PRUNING SHOWS SLIGHTLY BETTER ACCURACY (BEING ABLE TO SEE THE ENTIRE ERROR BEFORE CUTTING BACK INSTEAD OF PREVENTING GROWTH).
print("PRUNING AND DIRECTLY STOPPING THE TREE WILL HAVE SIMILAR RESULTS.  HOWEVER, PRUNING SHOWS SLIGHTLY BETTER ACCURACY (BEING ABLE TO SEE THE ENTIRE ERROR BEFORE CUTTING BACK INSTEAD OF PREVENTING GROWTH).")