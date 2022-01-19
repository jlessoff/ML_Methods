import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import  StandardScaler
from sklearn import metrics
import numpy as np
from sklearn.metrics import mean_squared_error

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



pd.set_option('display.max_columns', None)
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
activity=pd.read_csv('Data-Table 1.csv')
activity=activity[['Code',"Taux"]]

#merge data to get all fields relating to jobs,sports, city information
learn=pd.merge(learn,learn_jobs,on='UID',how='left')
learn=pd.merge(learn,learn_sport,on='UID',how='left')
learn=pd.merge(learn,city_admin,on='INSEE_CODE',how='left')
learn=pd.merge(learn,activity,left_on='INSEE_CODE',right_on='Code',how='left')
learn= learn.drop(columns='Code')
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
#
#

#import test data
test=pd.read_csv("test_dataset.csv")
test_jobs=pd.read_csv("test_dataset_job.csv")
test_sport=pd.read_csv("test_dataset_sport.csv")
test_emp=pd.read_csv("test_dataset_Emp.csv")
test=pd.merge(test,test_jobs,on='UID',how='left')
test=pd.merge(test,test_sport,on='UID',how='left')
test=pd.merge(test,activity,left_on='INSEE_CODE',right_on='Code',how='left')
test= test.drop(columns='Code')
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
# test['emp_status_indicator']= np.where(test['ACTIVITY_TYPE']=='type1-1',0,1)
test['Is_student']= np.where(test['Is_student']==False,0,1)


test=test.set_index('UID')
test= test.fillna(0)


cat_vars=['Nom fédération','Is_student',"Nom catégorie",'Code','N3','N2','N1','INSEE_CODE','dep','FAMILTY_TYPE','Sex','Employee_count','Job_42','DEGREE','ACTIVITY_TYPE','job_condition','Job_category','Terms_of_emp','economic_sector','JOB_DEP','job_desc','Nom de la commune','Nom du département','city_type','REG','Nom de la région','employer_category','Categorie']

learn[cat_vars] = learn[cat_vars].astype(str)
cont_vars=['Taux','target','ACTIVITY_TYPE','AGE_2018','Working_hours','Pay','inhabitants','Lat','Long','X','Y','club_indicator']
#
learn['Taux']=learn['Taux'].astype(float)

#make 'na' category
learn[cat_vars]= np.where(learn[cat_vars]==0,'na',learn[cat_vars])
test[cat_vars]= np.where(test[cat_vars]==0,'na',test[cat_vars])

#get quick encoding to make correlation heatmap
label = preprocessing.LabelEncoder()
onehotlabels_cat = learn[cat_vars].apply(label.fit_transform)
plt.figure(figsize=(16, 6))

#
heatmap = sns.heatmap(onehotlabels_cat.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':5}, pad=4);
plt.show()
#
#
onehotlabels_cont = learn[cont_vars].apply(label.fit_transform)
heatmap = sns.heatmap(onehotlabels_cont.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':5}, pad=4);
plt.show()
#
# # #drop perfectly correlated variables
# test= test.drop(columns=['Code','INSEE_CODE','N3','N2','X','Y'])#
# learn= learn.drop(columns=['Code','INSEE_CODE','N3','N2','X','Y'])#
#
# cat_vars=['Nom fédération','Is_student',"Nom catégorie",'N1','dep','FAMILTY_TYPE','Sex','Employee_count','Job_42','DEGREE','ACTIVITY_TYPE','job_condition','Job_category','Terms_of_emp','economic_sector','JOB_DEP','job_desc','Nom de la commune','Nom du département','city_type','REG','Nom de la région','employer_category','Categorie']

cat_vars=['Nom fédération','Is_student',"Nom catégorie",'N1','dep','FAMILTY_TYPE','Sex','Employee_count','club_indicator','Job_42','DEGREE','job_condition','Terms_of_emp','economic_sector','JOB_DEP','city_type','REG','Nom de la région','Categorie']
cont_vars=['target','AGE_2018','Working_hours','Pay','inhabitants','Lat','Long','Taux']
learn[cat_vars] = learn[cat_vars].astype(str)
learn['Taux']=learn['Taux'].astype(float)
# onehotlabels_cat = learn[cat_vars].apply(label.fit_transform)
# onehotlabels_cont = learn[cont_vars].apply(label.fit_transform)
# #
# # heatmap = sns.heatmap(onehotlabels_cat.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
# # heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':5}, pad=4);
# # plt.show()
# #
# #
# # onehotlabels_cont = learn[cont_vars].apply(label.fit_transform)
# # heatmap = sns.heatmap(onehotlabels_cont.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
# # heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':5}, pad=4);
# # plt.show()
#
#
# #
# #
#
# #
#
# # #save df
# # learn.to_csv('learn.csv')
# # # create encoding
#


OH_encoder = OneHotEncoder(sparse=False ,handle_unknown='ignore')
encoded_columns_learn =    OH_encoder.fit_transform(learn[cat_vars])
# encoded_columns_test =    OH_encoder.transform(test[cat_vars])
cont_learn=learn[cont_vars].to_numpy()
processed_data = np.concatenate([cont_learn, encoded_columns_learn], axis=1)
print(processed_data)


# Scale data and do k-means classification
mms = StandardScaler()
transformed= mms.fit(processed_data)
data_transformed = (transformed)
Sum_of_squared_distances = []
kmeans = KMeans(n_clusters=4,random_state=1)
clusters = kmeans.fit_predict(processed_data)
# clusters=np.transpose(clusters)
print(clusters)
processed_data = np.concatenate(([clusters[:,None], processed_data]),axis=1)
print(processed_data)




Y=processed_data[:,1]
X=processed_data[:,2:]
Group=processed_data[:,0]
group_kfold = StratifiedGroupKFold(n_splits=5,random_state=True,shuffle=True)




# group_kfold.get_n_splits(X, Y, Group)
print(group_kfold)
print(len(encoded_columns_learn[0]))
reg_decision_model=RandomForestRegressor()
parameters={'min_samples_split': range(70, 80)}
# #
# n_estimators = [int(x) for x in np.linspace(start = 800 , stop = 1000, num = 50)]
n_estimators = [800, 900, 700, 400, 800]
max_depth = [100, 200, 300, 400, 500]
# max_depth = [int(x) for x in np.linspace(1500, 2000, num = 50)]
Xtrain, Xval, ytrain, yval, grouptrain, grouptest = train_test_split(X, Y, Group,
                                              train_size=0.7, random_state=42, shuffle=True)
gkf = GroupKFold(n_splits=4)
rfr = RandomForestRegressor(random_state = 1)
r_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth}
tuning_model = RandomizedSearchCV(estimator=rfr, param_distributions=r_grid, n_iter = 4, scoring='neg_mean_absolute_error', cv = gkf, verbose=2, random_state=42, n_jobs=-1, return_train_score=True)
tuning_model.fit(Xtrain,ytrain,groups=grouptrain)
print(tuning_model.best_params_)
print(tuning_model.best_score_)
results = tuning_model.cv_results_
for key,value in results.items():
    print(key, value)



rfr.fit(Xtrain,ytrain)
y_pred = tuning_model.best_estimator_.predict(Xval)
mse = mean_squared_error(yval, y_pred)
print(mse)
print(y_pred)

y_pred = tuning_model.best_estimator_.predict(Xtrain)
mse = mean_squared_error(ytrain, y_pred)
print(mse)
print(y_pred)





# regr_1=DecisionTreeRegressor(min_samples_split=10,max_depth=30)
# regr_1.fit(X,Y)
# y_1 = regr_1.predict(X)
# print(Y)
# print(y_1)

#
# ##CREATE ONE HOT ENCODING
# # print(test)
# #
# #
# # print(learn.index)
# #
#
# #
# #
#
# #
# #
# #
#
#
# #
# # # missing values
# # learn.isna().sum().reset_index(name="n").plot.bar(x='index', y='n', rot=45)
# # plt.ylim([0, 50000])
# #
# # plt.show()
# # # print(learn['UID'].isna().sum())
# # na_group=(learn[[
# #        'job_condition', 'Job_category', 'Pay', 'Employee_count',
# #        'Terms_of_emp', 'economic_sector', 'JOB_DEP', 'employer_category',
# #        'job_desc']].isna())
# #
# # na= learn.groupby('job_condition', dropna=False).count()
# #
# # club= learn.groupby('job_condition', dropna=False)['CLUB'].count()
# # group= learn.groupby('job_condition', dropna=False)['UID'].count()
# # # print(club,group)
# # # print((club/group))
# #
# #
# #
# #
# #

# # #
# #
