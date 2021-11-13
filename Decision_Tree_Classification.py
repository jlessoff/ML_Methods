import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


#question 1: load the rice data set and prepare it for machine learning application using sample evaluation (with train-test-split).  keep 30%
rice=pd.read_csv("rice.csv")
rice_X=rice.drop(columns="CLASS")
rice_Y=rice['CLASS']
#get class names
rice_class = rice_Y.unique()
rice_features = rice_X.columns
#split the dataset

from sklearn.model_selection import train_test_split
r_X_tr, r_X_test, r_Y_tr, r_Y_test=train_test_split(rice_X,rice_Y,
                                                    train_size=0.7, random_state=42)
cl_model=DecisionTreeClassifier( random_state=42)
cl_model.fit(r_X_tr,r_Y_tr)
tree_prediction_test=cl_model.predict(r_X_test)

print(tree_prediction_test)
# question 2: compute the pruning summary on the training set for the rice data set and plot the evolution of the impurity as a function of the alpha param
path=cl_model.cost_complexity_pruning_path(r_X_tr,r_Y_tr)
print(path)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
print(ccp_alphas)
print(impurities)
#plot the evolution of the impurity on the training as a function of the alpha parameter
plt.plot(ccp_alphas,impurities)
plt.show()

#Question 3 Using a loop, compute all the pruned trees as well as their accuracy on both the training set and the test set.
acc_train=[]
acc_test=[]
for i in range(0,len(ccp_alphas)):
    clf= DecisionTreeClassifier(random_state=42, ccp_alpha= ccp_alphas[i])
    clf.fit(r_X_tr, r_Y_tr)
    tree_prediction_test = clf.predict(r_X_test)
    accuracy_test=accuracy_score(r_Y_test,tree_prediction_test)
    acc_test.append(accuracy_test)

    tree_prediction_train = clf.predict(r_X_tr)
    accuracy_train=accuracy_score(r_Y_tr,tree_prediction_train)
    acc_train.append(accuracy_train)

#Plot the results and comment them.
print(acc_test)
plt.plot(acc_test)
plt.show()

plt.plot(acc_train)
plt.show()
##ACCURACY INCREASES IN TEST AS THERE IS MORE PRUNING, AND DECREASES IN TRAINING WHEN THERE IS MORE PRUNING.  THIS MAKES SENSE BECAUSE PRUNING WILL HELP PREVENT OVERFITTING, LEADING TO BETTER RESULTS FOR THE TEST SET.