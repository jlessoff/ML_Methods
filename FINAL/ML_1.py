import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import  StandardScaler
import numpy as np
from sklearn import metrics
from sklearn.impute import KNNImputer

from sklearn.model_selection import RandomizedSearchCV
from numpy import ravel
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#load data from provided from class on test and learn data sets
learn=pd.read_csv("learn_dataset.csv")
learn_jobs=pd.read_csv("learn_dataset_job.csv")
learn_sport=pd.read_csv("learn_dataset_sport.csv")
learn_emp=pd.read_csv("learn_dataset_Emp.csv")
test=pd.read_csv("test_dataset.csv")
test_jobs=pd.read_csv("test_dataset_job.csv")
test_sport=pd.read_csv("test_dataset_sport.csv")
test_emp=pd.read_csv("test_dataset_Emp.csv")

#load city data
city_admin=pd.read_csv("city_adm.csv")
departments=pd.read_csv("departments.csv")
latlong=pd.read_csv("city_loc.csv")
citypop=pd.read_csv("city_pop.csv")
regions=pd.read_csv("regions.csv")

#load clubinformaton
club=pd.read_csv("code_CLUB.csv")

#load job information
pcsesemap=pd.read_csv("pcsese2017-map.csv")
code_job=pd.read_csv('code_Job_42.csv')
code_job_desc=pd.read_csv('code_job_desc.csv')

#load data on activity rate: EXTERNAL
activity=pd.read_excel('insee_data.xlsx',sheet_name='Data',skiprows=[0,1,2],usecols='A,I',index_col=0)
activity.index.rename('INSEE_CODE',inplace=True)


#merge data to get all fields relating to jobs,sports, city information
learn=pd.merge(learn,learn_jobs,on='UID',how='left')
learn=pd.merge(learn,learn_sport,on='UID',how='left')
learn=pd.merge(learn,city_admin,on='INSEE_CODE',how='left')
learn=pd.merge(learn,activity,on='INSEE_CODE',how='left')
learn=pd.merge(learn,departments,on='dep',how='left')
learn=pd.merge(learn,regions,on='REG',how='left')
learn=pd.merge(learn,latlong,on='INSEE_CODE',how='left')
learn=pd.merge(learn,citypop,on='INSEE_CODE',how='left')
learn=pd.merge(learn,club,left_on='CLUB',right_on='Code',how='left')
learn= learn.drop(columns='Code')
learn=pd.merge(learn,code_job,left_on='Job_42',right_on='Code',how='left')
learn= learn.drop(columns='Code')
learn=pd.merge(learn,code_job_desc,left_on='job_desc',right_on='Code',how='left')
learn=pd.merge(learn,pcsesemap,left_on='Code',right_on='N3',how='left')
learn.rename(columns={"Taux d'activité par tranche d'âge 2018_x000D_\nEnsemble":"Taux"},inplace=True)


#import test data and perform same steps to join
test=pd.merge(test,test_jobs,on='UID',how='left')
test=pd.merge(test,test_sport,on='UID',how='left')
test=pd.merge(test,activity,on='INSEE_CODE',how='left')
test=pd.merge(test,city_admin,on='INSEE_CODE',how='left')
test=pd.merge(test,departments,on='dep',how='left')
test=pd.merge(test,regions,on='REG',how='left')
test=pd.merge(test,latlong,on='INSEE_CODE',how='left')
test=pd.merge(test,citypop,on='INSEE_CODE',how='left')
test=pd.merge(test,club,left_on='CLUB',right_on='Code',how='left')
test= test.drop(columns='Code')
test=pd.merge(test,code_job,left_on='Job_42',right_on='Code',how='left')
test= test.drop(columns='Code')
test=pd.merge(test,code_job_desc,left_on='job_desc',right_on='Code',how='left')
test=pd.merge(test,pcsesemap,left_on='Code',right_on='N3',how='left')
test.rename(columns={"Taux d'activité par tranche d'âge 2018_x000D_\nEnsemble":"Taux"},inplace=True)

#create variable for 'club' : to indicate whether someone is in a club or not
learn['club_indicator']= np.where(learn['Categorie'].isna(),0,1)
test['club_indicator']= np.where(test['Categorie'].isna(),0,1)


UID=test['UID'].to_numpy()
#set UID as index
learn=learn.set_index('UID')
test=test.set_index('UID')


#look at na
learn.isna().sum().reset_index(name="n").plot.bar(x='index', y='n', rot=45)
plt.ylim([0, 50000])
plt.savefig('na_plot.png')


#fill nas with zero: this will give continuous variables a value of 'zero' which makes sense for pay, etc.
print(learn['Taux'].mean())
learn['Taux']=learn['Taux'].fillna(74.21852281764991)
test['Taux']=test['Taux'].fillna(74.21852281764991)
test= test.fillna(0)
learn= learn.fillna(0)

#determine and separate cat variables and continuous variables
cat_vars=['Nom fédération','club_indicator','Is_student',"Nom catégorie",'Code','N3','N2','N1','INSEE_CODE','dep','FAMILTY_TYPE','Sex','Employee_count','Job_42','DEGREE','ACTIVITY_TYPE','job_condition','Job_category','Terms_of_emp','economic_sector','JOB_DEP','job_desc','Nom de la commune','Nom du département','city_type','REG','Nom de la région','employer_category','Categorie']
#convert categories to string
learn[cat_vars] = learn[cat_vars].astype(str)
test[cat_vars] = test[cat_vars].astype(str)



#specify continuous variables
cont_vars=['Taux','ACTIVITY_TYPE','AGE_2018','Working_hours','Pay','inhabitants','Lat','Long','X','Y']
#convert new activity rate variable to float (necessary due to csv)
learn['Taux']=learn['Taux'].astype(float)
test['Taux']=test['Taux'].astype(float)


#make 'na' category for categorical variables.  this step is not necessary as it will be encoded, but mainly included to clarify what the '0' represent nas
learn[cat_vars]= np.where(learn[cat_vars]==0,'na',learn[cat_vars])
test[cat_vars]= np.where(test[cat_vars]==0,'na',test[cat_vars])



#get quick encoding to make correlation heatmap for the learn data to show relationships
label = preprocessing.LabelEncoder()
label_encode_cat = learn[cat_vars].apply(label.fit_transform)
plt.figure(figsize=(16, 6))

#
heatmap = sns.heatmap(label_encode_cat.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap or Categorical Variables', fontdict={'fontsize':12}, pad=4);
plt.show()
plt.savefig('categorical_heatmap.png')

#
#
onehotlabels_cont = learn[cont_vars].apply(label.fit_transform)
heatmap = sns.heatmap(onehotlabels_cont.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap for Continuous Variables', fontdict={'fontsize':12}, pad=4);
plt.show()
plt.savefig('continuous_heatmap.png')

#
# as we are going to one hot encode the variables and end up with many features, we will drop perfectly correlated variables and variables that are highly correlate to eachother (above .9).  as we are one hot encoding, this will reduce computational complexity a lot, while minimally affectingn results
cat_vars=['Nom fédération','Is_student',"Nom catégorie",'N1','dep','FAMILTY_TYPE','Sex','Employee_count','club_indicator','Job_42','DEGREE','job_condition','Terms_of_emp','economic_sector','JOB_DEP','city_type','REG','Nom de la région','Categorie']
cont_vars=['AGE_2018','Working_hours','Pay','inhabitants','Lat','Long','Taux']
target=['target']
feature_names=['Group','AGE_2018','Working_hours','Pay','inhabitants','Lat','Long','Taux','Nom fédération','Is_student',"Nom catégorie",'N1','dep','FAMILTY_TYPE','Sex','Employee_count','club_indicator','Job_42','DEGREE','job_condition','Terms_of_emp','economic_sector','JOB_DEP','city_type','REG','Nom de la région','Categorie']

#create cont variables for test, as they do not contain target var

#convert cat to string
learn[cat_vars] = learn[cat_vars].astype(str)
test[cat_vars] = test[cat_vars].astype(str)
#
test_cat=test[cat_vars]
learn_cat=learn[cat_vars]

test_con=test[cont_vars]
learn_con=learn[cont_vars]

target_learn=learn[target].to_numpy()

#convert taux to float
learn['Taux']=learn['Taux'].astype(float)
test['Taux']=test['Taux'].astype(float)

# # # create encoding

OH_encoder = OneHotEncoder(sparse=False ,handle_unknown='ignore')
encoded_columns_learn =    OH_encoder.fit_transform(learn_cat)
encoded_columns_test =    OH_encoder.transform(test_cat)
cont_learn=learn_con.to_numpy()
cont_test=test_con.to_numpy()


#scale continuous data for kmeans, knn, ridge
mms = StandardScaler()
cont_learn_scaled= mms.fit_transform(cont_learn)
cont_test_scaled= mms.transform(cont_test)

processed_data_scaled = np.concatenate([cont_learn_scaled, encoded_columns_learn], axis=1)
processed_data_test_scaled=np.concatenate([cont_test_scaled, encoded_columns_test], axis=1)
#concat non scaled data
processed_data = np.concatenate([cont_learn, encoded_columns_learn], axis=1)
processed_data_test = np.concatenate([cont_test, encoded_columns_test], axis=1)


# find k value (ending with k=4)
K = range(2,10)
CS_scores = []
Sum_of_squared_distances = []
for k in K:
    km = KMeans(n_clusters=k, init='k-means++', random_state=1)
    cluster_found = km.fit_predict(processed_data_scaled)
    Sum_of_squared_distances.append(km.inertia_)
    CS_scores.append(metrics.calinski_harabasz_score(processed_data_scaled, km.labels_))
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.title('Find Optimal k using elbow')
plt.savefig('elbow_score.png')
plt.show()




# Scale data and do k-means classification
kmeans = KMeans(n_clusters=6,random_state=1)
clusters = kmeans.fit_predict(processed_data_scaled)
clusters_test=kmeans.predict(processed_data_test_scaled)
# clusters=np.transpose(clusters)
processed_data = np.concatenate(([clusters[:,None], processed_data]),axis=1)
processed_data_test = np.concatenate(([clusters[:,None], processed_data_test]),axis=1)
#
#
#
# #extract x,y,group for learn
Y=target_learn
X=processed_data[:,0:]
X_scaled=processed_data_scaled[:,0:]
Group=processed_data[:,0]

# #extract x,y,group for test
X_test=processed_data_test[:,0:]
X_test_scaled=processed_data_test_scaled[:,0:]

##Do Gridsearchcv to narrow down estmators; we found best estimator:  we found the best to be:
#(random_state = 1,bootstrap= True, max_depth= 500, max_features= 400, min_samples_leaf= 1, min_samples_split= 2, n_estimators= 625)

Xtrain, Xval, ytrain, yval, grouptrain, grouptest = train_test_split(X, Y, Group,
                                              train_size=0.7, random_state=42, shuffle=True)

#use group k fold based on the k-means clusters found previously
gkf = GroupKFold(n_splits=6)
rfr = RandomForestRegressor(random_state = 1)
param_grid = {
    'bootstrap': [True],
    'max_depth': [400, 500, 300],
    'max_features': [400, 450],
    'min_samples_leaf': [1, 5],
    'min_samples_split': [2, 4],
    'n_estimators': [550, 600, 625]}

tuning_model = RandomizedSearchCV(estimator=rfr, n_iter=1,param_distributions=param_grid, scoring='neg_mean_absolute_error', cv = gkf, verbose=2, random_state=42, n_jobs=-1, return_train_score=True)
tuning_model.fit(Xtrain,ytrain.ravel(),groups=grouptrain)
print(tuning_model.best_params_)
print(tuning_model.best_score_)
results = tuning_model.cv_results_
for key,value in results.items():
    print(key, value)

rfr = RandomForestRegressor(random_state = 1,bootstrap= True, max_depth= 500, max_features= 400, min_samples_leaf= 1, min_samples_split= 2, n_estimators= 625)
rfr.fit(Xtrain,ytrain)
y_pred = rfr.predict(Xval)
mse = mean_squared_error(yval, y_pred)
r2=r2_score(y_true=yval,y_pred=y_pred)
print('RF test r2',r2)
print('RF test mse', mse)



rfr.fit(X,Y)
y_pred = rfr.predict(X_test)
y_pred=y_pred.tolist()
prediction=pd.DataFrame(y_pred, UID,columns=['Y_PRED'])
prediction.to_csv('rfrprediction.csv')


#
#
Xtrain_scaled, Xval_scaled, ytrain, yval, grouptrain, grouptest = train_test_split(X, Y, Group,
                                              train_size=0.7, random_state=43, shuffle=True)


gkf = GroupKFold(n_splits=6)
from sklearn.model_selection import GridSearchCV
rlr=Ridge()
parameters = {'alpha':list(range(1, 10))}
grid_search = GridSearchCV(estimator = rlr, param_grid = parameters,
                          cv = gkf, n_jobs = -1, verbose = 2, scoring='neg_mean_squared_error')
grid_search.fit(Xtrain_scaled,ytrain,groups=grouptrain)

print(grid_search.best_params_)
print(grid_search.best_score_)

rlr=Ridge(alpha=6)
rlr.fit(Xtrain_scaled,ytrain)
y_pred = rlr.predict(Xval_scaled)
mse = mean_squared_error(yval, y_pred)
r2=r2_score(y_true=yval,y_pred=y_pred)
print('mse test cv',mse)
print('mse test r2',r2)

rlr.fit(X_scaled,Y)
test_id=test.index.tolist()
y_pred = rlr.predict(X_test_scaled)
y_pred=y_pred.tolist()
prediction=pd.DataFrame(y_pred, UID,columns=['Y_PRED'])
prediction.to_csv('ridgeprediction.csv')



####
knn = KNeighborsRegressor()
k_range = list(range(1, 10))
param_grid = dict(n_neighbors=k_range)
rscv = GridSearchCV(knn, param_grid,  refit=True, cv=gkf, verbose=0)

# fitting the model for grid search
grid_search = rscv.fit(Xtrain_scaled, ytrain, groups=grouptrain)
print(rscv.best_params_)
print(rscv.best_score_)
results = rscv.cv_results_
for key, value in results.items():
    print(key, value)

knn = KNeighborsRegressor(n_neighbors=9)
knn.fit(Xtrain_scaled, ytrain)
y_pred = knn.predict(Xval_scaled)
mse = mean_squared_error(yval, y_pred)
r2=r2_score(y_true=yval,y_pred=y_pred)
print('mse for knn test',mse)
print('r2 for knn test',r2)

y_pred = knn.predict(Xtrain_scaled)
mse = mean_squared_error(ytrain, y_pred)
print(mse)

knn.fit(X_scaled, Y)
y_pred = knn.predict(X_test_scaled)
y_pred=y_pred.tolist()
prediction=pd.DataFrame(y_pred, UID,columns=['Y_PRED'])
prediction.to_csv('knnprediction.csv')
