import pandas as pd
import statistics
from matplotlib import pyplot
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.model_selection import cross_val_score
import sklearn.metrics
import seaborn as sns
#sns relplot data . melt value name x y kind=line

#import data
wines=pd.read_csv('winequality-red.csv',delimiter=',',skiprows=1)
# explanatory variable
X=wines[wines.columns[1:12]]
#target variable
y=wines['quality']
average_score=[]
i_list=[]
name= []
for i in range(0,20):
    kf = KFold(n_splits=5, shuffle=True, random_state=i)
    score = cross_val_score(DecisionTreeRegressor(min_samples_split=5, random_state=42), X, y, cv= kf, scoring='neg_mean_squared_error')
    i_list.append((statistics.mean(-score)))
    name.append('fold')

for i in range(0, 100):
    from sklearn.model_selection import train_test_split

    Xtrain, Xval, ytrain, yval = train_test_split(X, y,
                                                  train_size=0.8, random_state=i)
    regr_1 = DecisionTreeRegressor(min_samples_split=5, random_state=42)
    regr_1.fit(Xtrain, ytrain)
    # Predict
    yvalid = regr_1.predict(Xval)
    # accuracy check using MSE
    mse1 = mean_squared_error(yval, yvalid)
    i_list.append(mse1)
    name.append("split")
print(len(i_list))
print(name)

dataframe=pd.DataFrame()
dataframe['name']=name
dataframe['mse']=i_list
sns.kdeplot(x=dataframe['name'], data=dataframe)
plt.show()
