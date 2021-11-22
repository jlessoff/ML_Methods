import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer



#Question 1 Load the rice data set and prepare it for a machine learning application using split sample evaluation (with the train_test_split function of Scikit-learn). Keep 30 % of the date set for the final evaluation.
rice=pd.read_csv("rice.csv")
rice_X=rice.drop(columns="CLASS")
rice_Y=rice['CLASS']
#Variable names
rice_class = rice_Y.unique()
rice_features = rice_X.columns
gs = GridSearchCV(
    DecisionTreeClassifier(random_state=40),
    param_grid={"min_samples_split": range(2, 10, 1)},
    # scoring=make_scorer(accuracy_score),
    # refit=True
)
gs.fit(rice_X, rice_Y)
results = gs.cv_results_
# print(results)
print(gs.best_params_)

for key,value in results.items():
    print(key)
#
scores=list(results)[11:14]
scores = {k: results[k] for k in scores}
for key,value in (scores.items()):
    ah=list(range(2, 10))
    plt.plot(ah,value)
    plt.title(key)
    print(key,value)
    plt.show()
