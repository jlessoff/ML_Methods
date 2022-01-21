import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import  StandardScaler
from sklearn import metrics
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import category_encoders
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder


#load data
learn=pd.read_csv("learn_dataset.csv")
learn_jobs=pd.read_csv("learn_dataset_job.csv")
learn_sport=pd.read_csv("learn_dataset_sport.csv")
learn_emp=pd.read_csv("learn_dataset_Emp.csv")
city_admin=pd.read_csv("city_adm.csv")
departments=pd.read_csv("departments.csv")
latlong=pd.read_csv("city_loc.csv")
citypop=pd.read_csv("city_pop.csv")
regions=pd.read_csv("regions.csv")
club=pd.read_csv("code_CLUB.csv")
pcsesemap=pd.read_csv("pcsese2017-map.csv")
code_job=pd.read_csv('code_Job_42.csv')
code_job_desc=pd.read_csv('code_job_desc.csv')
# activity=pd.read_csv('Data-Table 1.csv')
# activity=activity[['Code',"Taux"]]
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
#
learn['club_indicator']= np.where(learn['Categorie'].isna(),0,1)
# learn['emp_status_indicator']= np.where(learn['ACTIVITY_TYPE']=='type1-1',0,1)
learn['Is_student']= np.where(learn['Is_student']==False,0,1)

learn=learn.set_index('UID')
learn= learn.fillna(0)
learn.rename(columns={"Taux d'activité par tranche d'âge 2018_x000D_\nEnsemble":"Taux"},inplace=True)
#
#

#import test data
test=pd.read_csv("test_dataset.csv")
test_jobs=pd.read_csv("test_dataset_job.csv")
test_sport=pd.read_csv("test_dataset_sport.csv")
test_emp=pd.read_csv("test_dataset_Emp.csv")
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

test['club_indicator']= np.where(test['Categorie'].isna(),0,1)
test['Is_student']= np.where(test['Is_student']==False,0,1)


test=test.set_index('UID')
test= test.fillna(0)
test.rename(columns={"Taux d'activité par tranche d'âge 2018_x000D_\nEnsemble":"Taux"},inplace=True)


cat_vars=['Nom fédération','Is_student',"Nom catégorie",'Code','N3','N2','N1','INSEE_CODE','dep','FAMILTY_TYPE','Sex','Employee_count','Job_42','DEGREE','ACTIVITY_TYPE','job_condition','Job_category','Terms_of_emp','economic_sector','JOB_DEP','job_desc','Nom de la commune','Nom du département','city_type','REG','Nom de la région','employer_category','Categorie']

learn[cat_vars] = learn[cat_vars].astype(str)
cont_vars=['Taux','target','ACTIVITY_TYPE','AGE_2018','Working_hours','Pay','inhabitants','Lat','Long','X','Y','club_indicator']
#
learn['Taux']=learn['Taux'].astype(float)

learn[cat_vars]= np.where(learn[cat_vars]==0,'na',learn[cat_vars])
test[cat_vars]= np.where(test[cat_vars]==0,'na',test[cat_vars])

cat_vars=['Nom fédération','Is_student',"Nom catégorie",'N1','dep','FAMILTY_TYPE','Sex','Employee_count','club_indicator','Job_42','DEGREE','job_condition','Terms_of_emp','economic_sector','JOB_DEP','city_type','REG','Nom de la région','Categorie']
cont_vars=['target','AGE_2018','Working_hours','Pay','inhabitants','Lat','Long','Taux']
learn[cat_vars] = learn[cat_vars].astype(str)
learn['Taux']=learn['Taux'].astype(float)

# # Scale data and do k-means classification

mms = StandardScaler()
OH_encoder = OneHotEncoder(sparse=False ,handle_unknown='ignore')
encoded_columns_learn =    OH_encoder.fit_transform(learn[cat_vars])
encoded_columns_learn_test =    OH_encoder.transform(learn_test[cat_vars])

cont_learn=learn[cont_vars].to_numpy()
cont_learn_scaled= mms.fit_transform(cont_learn)


processed_data_scaled = np.concatenate([cont_learn_scaled, encoded_columns_learn], axis=1)


processed_data = np.concatenate([cont_learn, encoded_columns_learn], axis=1)
processed_data_test = np.concatenate([cont_learn_test, encoded_columns_learn_test], axis=1)


kmeans = KMeans(n_clusters=4,random_state=1)
clusters = kmeans.fit_predict(processed_data_scaled)
clusters_test = kmeans.predict(processed_data_scaled_test)

processed_data = np.concatenate(([clusters[:,None], processed_data]),axis=1)
processed_data_test = np.concatenate(([clusters_test[:,None], processed_data_test]),axis=1)


Group=processed_data[:,0]
Group_test=processed_data_test[:,0]


X_scaled=processed_data_scaled[:,2:]
X_scaled_test=processed_data_scaled_test[:,2:]

X_scaled = np.concatenate(([clusters[:,None], X_scaled]),axis=1)
X_scaled_test = np.concatenate(([clusters_test[:,None], X_scaled_test]),axis=1)

Y=processed_data[:,1]
Y_test=processed_data_test[:,1]


#
# # Xtrain, Xval, ytrain, yval, grouptrain, grouptest = train_test_split(X, Y, Group,
# #                                               train_size=0.7, random_state=3, shuffle=True)
# #
# #
Xtrain_scaled, Xval_scaled, ytrain, yval, grouptrain, grouptest = train_test_split(X_scaled, Y, Group,
                                              train_size=0.7, random_state=6, shuffle=True)




gkf = GroupKFold(n_splits=4)
from sklearn.model_selection import GridSearchCV
rlr=Ridge()
parameters = {'alpha':[1,2,3,4,5,6,7,8,9,10]}
grid_search = GridSearchCV(estimator = rlr, param_grid = parameters,
                          cv = gkf, n_jobs = -1, verbose = 2, scoring='neg_mean_squared_error')
grid_search.fit(Xtrain_scaled,ytrain,groups=grouptrain)
print(grid_search.best_params_)
print(grid_search.best_score_)
#
y_pred = grid_search.best_estimator_.predict(Xval_scaled)
mse = mean_squared_error(yval, y_pred)
r2=r2_score(y_true=yval,y_pred=y_pred)

print('mse test cv',mse)
print('mse test r2',r2)


y_pred = grid_search.best_estimator_.predict(Xtrain_scaled)
mse = mean_squared_error(ytrain, y_pred)
r2=r2_score(y_true=ytrain,y_pred=y_pred)
print('mse train',mse)
print('r2 train',r2)



y_pred = grid_search.best_estimator_.predict(X_scaled_test)
mse = mean_squared_error(Y_test, y_pred)
r2=r2_score(y_true=Y_test,y_pred=y_pred)
print('mse TEST',mse)
print('r2 TEST',r2)


best_random = grid_search.best_estimator_
n_estimators = [int(x) for x in np.linspace(start = 800 , stop = 1000, num = 50)]


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

y_pred = rscv.best_estimator_.predict(Xval_scaled)
mse = mean_squared_error(yval, y_pred)
print(mse)
print(y_pred)

y_pred = rscv.best_estimator_.predict(Xtrain_scaled)
mse = mean_squared_error(ytrain, y_pred)
print(mse)
print(y_pred)

y_pred = rscv.best_estimator_.predict(X_scaled_test)
mse = mean_squared_error(Y_test, y_pred)
print(mse)
print(y_pred)