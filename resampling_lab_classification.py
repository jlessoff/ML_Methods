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
    param_grid={"min_samples_split": range(2, 50, 1)},
    scoring=make_scorer(accuracy_score),
    # refit=True
)
gs.fit(rice_X, rice_Y)
results = gs.cv_results_
# print(results)
print(gs.best_params_)

for key,value in results.items():
    print(key)
# #
scores=list(results)[11:14]
scores = {k: results[k] for k in scores}
# for key,value in (scores.items()):
#     ah=list(range(2, 50))
#     plt.plot(ah,value)
#     plt.title(key)
#     print(key,value)
#     plt.show()
print('ah',gs.best_score_)

Xtrain, Xval, ytrain, yval = train_test_split(rice_X, rice_Y,
                                              train_size=0.7, random_state=42)
gs = GridSearchCV(
    DecisionTreeClassifier(random_state=40),
    param_grid={"min_samples_split": range(2, 50, 1)},
    scoring=make_scorer(accuracy_score),
    cv=5
    # refit=True
)
gs.fit(Xtrain, ytrain)
results = gs.cv_results_
print(gs.best_params_)
scores=list(results)[11:14]
scores = {k: results[k] for k in scores}
print('ah',gs.best_score_)
y_pred = gs.best_estimator_.predict(Xval)
print(confusion_matrix(yval, y_pred))