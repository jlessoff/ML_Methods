import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix



#Question 1 Load the rice data set and prepare it for a machine learning application using split sample evaluation (with the train_test_split function of Scikit-learn). Keep 30 % of the date set for the final evaluation.
rice=pd.read_csv("rice.csv")
rice_X=rice.drop(columns="CLASS")
rice_Y=rice['CLASS']
rice_class = rice_Y.unique()
rice_features = rice_X.columns
gs = GridSearchCV(
    DecisionTreeClassifier(random_state=40),
    param_grid={"min_samples_split": range(2, 100)},
    scoring='accuracy',
)
gs.fit(rice_X, rice_Y)
results = gs.cv_results_

#Show best parameter, and best model score
print('bestparam',gs.best_params_)
print('bestscore',gs.best_score_)
#PRint all of the results from the resulting dictionary
for key,value in results.items():
    print(key)
##Question 2 Using the results of the grid search from question 1, plot the estimates obtained during the search for the best model.
scores=list(results)[11:12]
scores = {k: results[k] for k in scores}
for key,value in (scores.items()):
    plt.plot(value)
    plt.title(key)
    plt.show()


#3.1 use train_test_split to build a training set and a testing set with 30 % of the original data for testing;
Xtrain, Xval, ytrain, yval = train_test_split(rice_X, rice_Y,
                                              train_size=0.7, random_state=42, shuffle=True)

#3.2 Use GridSearchCV to build an optimal decision tree by 5-fold cross-validation on the training set, optimizing the min_samples_split parameter;
gs = GridSearchCV(
    DecisionTreeClassifier(random_state=40),
    param_grid={"min_samples_split": range(2, 100)},
    scoring='accuracy',
    cv=5
)
gs.fit(Xtrain, ytrain)
results = gs.cv_results_
scores=list(results)[11:13]
scores = {k: results[k] for k in scores}
for key,value in (scores.items()):
    plt.plot(value)
    plt.title(key)
    plt.show()
scores=list(results)[11:14]
scores = {k: results[k] for k in scores}
print('bestparam',gs.best_params_)
print('bestscore',gs.best_score_)

#3.3 Compare the performances predicted on the test set with the one obtained by 5-fold CV for the best parameters (available in the best_score_ attribute of the result of GridSearchCV).


#Question 4: Estimate the future performances of the tree by computing and printing its confusion matrix and its accuracy on the test set.
y_pred = gs.best_estimator_.predict(Xval)
print(confusion_matrix(yval, y_pred))

for i in range(1,20):
    Xtrain, Xval, ytrain, yval = train_test_split(rice_X, rice_Y,
                                                  train_size=0.7, random_state=i, shuffle=True)
    gs = GridSearchCV(
        DecisionTreeClassifier(random_state=40),
        param_grid={"min_samples_split": range(2, 100)},
        scoring='accuracy',
        cv=5
    )
    gs.fit(Xtrain, ytrain)
    print('bestparam for ',i,'random state:', gs.best_params_)
    print('bestscore for ',i,'random state:', gs.best_score_)
print('By changing the random state and comparing the best parameter and score for each random state, I conclude that there is a small amount of variability resulting from the split sample approach. There is a small negative effect on the stability of this procedure')
