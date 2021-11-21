import pandas as pd
import statistics
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.model_selection import cross_val_score
import seaborn as sns

#import data
wines=pd.read_csv('winequality-red.csv',delimiter=',',skiprows=1)
# explanatory variable
X=wines[wines.columns[1:12]]
#target variable
y=wines['quality']
i_list=[]

average_score=[]
#question 4
for i in range(2,100):
    kf = KFold(n_splits=5, shuffle=True, random_state=4)
    score = cross_val_score(DecisionTreeRegressor(min_samples_split=i, random_state=42), X, y, cv= kf, scoring='neg_mean_squared_error')
    list=[]
    for l in score:
        df=(i,l)
        average_score.append(-l)
        i_list.append(i)
sns.relplot(x=i_list, y=average_score, kind='line')
import matplotlib.pyplot as plt
# plt.show()
# QUESTION 5
name_1=[]
i_list_1=[]
for i in range(0,20):
    kf = KFold(n_splits=5, shuffle=True, random_state=i)
    score = cross_val_score(DecisionTreeRegressor(min_samples_split=5, random_state=42), X, y, cv= kf, scoring='neg_mean_squared_error')
    i_list_1.append((statistics.mean(-score)))
    name_1.append('fold')
average=statistics.mean(i_list_1)
ste=statistics.stdev(i_list_1)
print(i_list_1)
X=wines[wines.columns[1:12]]
#target variable
y=wines['quality']





#FOR TRAIN SPLIT

X=wines[wines.columns[1:12]]
#target variable
y=wines['quality']
from sklearn.model_selection import train_test_split
Xtrain, Xval, ytrain, yval=train_test_split(X,y,
                                                    train_size=0.8, random_state=4)

##QUESTION 4
mse_val=[]
for i in range(2,100):
    regr_1 = DecisionTreeRegressor(min_samples_split=i,random_state=42)
    regr_1.fit(Xtrain, ytrain)
    # Predict
    yvalid = regr_1.predict(Xval)
    # accuracy check using MSE
    mse1 = mean_squared_error(yval, yvalid)
    mse_val.append(mse1)
sns.relplot(x=range(2,100), y=(mse_val), kind='line')
plt.show()

###QUESTION 5
i_list=[]
name=[]
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
average=statistics.mean(mse_val)
ste=statistics.stdev(mse_val)
print(i_list)
print(name)
print(len(i_list))
print(len(name))
# # question 5
dataframe=pd.DataFrame()
print(dataframe)
dataframe['name']=name
dataframe['mse']=i_list
print(dataframe)
sns.kdeplot(data=dataframe)


dataframe1=pd.DataFrame()
print(dataframe1)
dataframe1['name']=name_1
dataframe1['mse']=i_list_1
print(dataframe1)
sns.kdeplot(data=dataframe1)
plt.show()


##TEXT ANSWERS:
print('Question 3: the results for the cross validation approach shows a more stable descent in MSE than for the test/train approach.  This would be recommended')
print('Question 4: using cross validation results in a much more stable result than using test/train across many random states.  The MSE results obtained from test/train are very variable and depend a lot on the data that is split.  This is mitigated by breaking into 5 different folds')
