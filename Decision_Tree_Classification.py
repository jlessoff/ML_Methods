import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


#Question 1 Load the rice data set and prepare it for a machine learning application using split sample evaluation (with the train_test_split function of Scikit-learn). Keep 30 % of the date set for the final evaluation.
rice=pd.read_csv("rice.csv")
rice_X=rice.drop(columns="CLASS")
rice_Y=rice['CLASS']
#Variable names
rice_class = rice_Y.unique()
rice_features = rice_X.columns
#Split the dataset

from sklearn.model_selection import train_test_split
r_X_tr, r_X_test, r_Y_tr, r_Y_test=train_test_split(rice_X,rice_Y,
                                                    train_size=0.7, random_state=42)
#Question 2 Compute the pruning summary on the training set for the rice data set and plot the evolution of the impurity on the training as a function of the alpha parameter.
cl_model=DecisionTreeClassifier( random_state=42)
cl_model.fit(r_X_tr,r_Y_tr)
tree_prediction_test=cl_model.predict(r_X_test)
print(tree_prediction_test)
#Plot the evolution of the impurity on the training as a function of the alpha parameter.
path=cl_model.cost_complexity_pruning_path(r_X_tr,r_Y_tr)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
plt.plot(ccp_alphas,impurities)
plt.title("Accuracy and Impurity Tradeoff")
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
plt.title("Accuracy Plot for Test Data")
plt.show()

plt.plot(acc_train)
plt.title("Accuracy Plot for Training Data")
plt.show()
#Pruning can prevent overfitting.  This will cause accuracy to increase in test predictions as there is more pruning, and  accuracy to decrease in training predictions when there is more pruning.
print('Pruning can prevent overfitting.  This will cause accuracy to increase in test predictions as there is more pruning, and  accuracy to decrease in training predictions when there is more pruning.')