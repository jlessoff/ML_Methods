import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
import statistics
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, SCORERS, confusion_matrix



#Question 1 Load the rice data set and prepare it for a machine learning application using split sample evaluation (with the train_test_split function of Scikit-learn). Keep 30 % of the date set for the final evaluation.
rice=pd.read_csv("rice.csv")
rice_X=rice.drop(columns="CLASS")
rice_Y=rice['CLASS']
#Variable names
rice_class = rice_Y.unique()
rice_features = rice_X.columns
gs = GridSearchCV(
    DecisionTreeClassifier(random_state=40),
    param_grid={"min_samples_split": range(2, 1000)},
    scoring='accuracy',
)
gs.fit(rice_X, rice_Y)
results = gs.cv_results_
# print(results)
print('bestparam',gs.best_params_)
print('bestscore',gs.best_score_)
#
for key,value in results.items():
    print(key)
# #
scores=list(results)[11:12]
scores = {k: results[k] for k in scores}
for key,value in (scores.items()):
    plt.plot(value)
    plt.title(key)
    plt.show()



Xtrain, Xval, ytrain, yval = train_test_split(rice_X, rice_Y,
                                              train_size=0.7, random_state=42, shuffle=True)
gs = GridSearchCV(
    DecisionTreeClassifier(random_state=40),
    param_grid={"min_samples_split": range(2, 1000)},
    scoring='accuracy',
    cv=5
)
gs.fit(Xtrain, ytrain)
results = gs.cv_results_
scores=list(results)[11:13]
scores = {k: results[k] for k in scores}
# for key,value in (scores.items()):
#     plt.plot(value)
#     plt.title(key)
#     plt.show()
scores=list(results)[11:14]
scores = {k: results[k] for k in scores}
print('bestparam',gs.best_params_)
print('bestscore',gs.best_score_)

y_pred = gs.best_estimator_.predict(Xval)
print(confusion_matrix(yval, y_pred))

for i in range(1,20):
    Xtrain, Xval, ytrain, yval = train_test_split(rice_X, rice_Y,
                                                  train_size=0.7, random_state=i, shuffle=True)
    gs = GridSearchCV(
        DecisionTreeClassifier(random_state=40),
        param_grid={"min_samples_split": range(2, 200)},
        scoring='accuracy',
        cv=5
    )
    gs.fit(Xtrain, ytrain)
    print('bestparam for ',i,'random state:', gs.best_params_)
    print('bestscore for ',i,'random state:', gs.best_score_)