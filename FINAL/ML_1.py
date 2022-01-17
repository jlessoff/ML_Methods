import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import  StandardScaler
from sklearn import metrics
import numpy as np
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import category_encoders
from sklearn.model_selection import GroupKFold



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


#merge data to get all fields relating to jobs,sports, city information
learn=pd.merge(learn,learn_jobs,on='UID',how='left')
learn=pd.merge(learn,learn_sport,on='UID',how='left')
learn=pd.merge(learn,city_admin,on='INSEE_CODE',how='left')
learn=pd.merge(learn,departments,on='dep',how='left')
learn=pd.merge(learn,latlong,on='INSEE_CODE',how='left')
learn=pd.merge(learn,citypop,on='INSEE_CODE',how='left')
learn=pd.merge(learn,regions,on='REG',how='left')
learn=pd.merge(learn,club,left_on='CLUB',right_on='Code',how='left')
learn['club_indicator']= np.where(learn['Categorie'].isna(),0,1)
learn['emp_status_indicator']= np.where(learn['ACTIVITY_TYPE']=='type1-1',0,1)
learn['Is_student']= np.where(learn['Is_student']=='FALSE',0,1)

learn=learn.set_index('UID')
learn= learn.fillna(0)

#import test data
test=pd.read_csv("test_dataset.csv")
test_jobs=pd.read_csv("test_dataset_job.csv")
test_sport=pd.read_csv("test_dataset_sport.csv")
test_emp=pd.read_csv("test_dataset_Emp.csv")
test=pd.merge(test,learn_jobs,on='UID',how='left')
test=pd.merge(test,learn_sport,on='UID',how='left')
test=pd.merge(test,city_admin,on='INSEE_CODE',how='left')
test=pd.merge(test,departments,on='dep',how='left')
test=pd.merge(test,latlong,on='INSEE_CODE',how='left')
test=pd.merge(test,citypop,on='INSEE_CODE',how='left')
test=pd.merge(test,regions,on='REG',how='left')
test=pd.merge(test,club,left_on='CLUB',right_on='Code',how='left')
test['Is_student']= np.where(test['Is_student']=='FALSE',0,1)
test['club_indicator']= np.where(test['Categorie'].isna(),0,1)
test=test.set_index('UID')
test['emp_status_indicator']= np.where(test['ACTIVITY_TYPE']=='type1-1',0,1)
test=test.fillna(0)


#get quick encoding to make correlation heatmap

# label = preprocessing.LabelEncoder()
# onehotlabels = learn.apply(label.fit_transform)
# plt.figure(figsize=(16, 6))
# heatmap = sns.heatmap(onehotlabels.corr())
# heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);
# plt.show()
#
# heatmap = sns.heatmap(onehotlabels.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
# heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':5}, pad=12);
# plt.show()


#drop perfectly correlated variables
test= test.drop(columns=['Nom fédération',"Nom catégorie",'Code','X','Y','INSEE_CODE'])#
learn= learn.drop(columns=['Nom fédération',"Nom catégorie",'Code','X','Y','INSEE_CODE'])#


#save df
learn.to_csv('learn.csv')
# create encoding
cat_vars=['FAMILTY_TYPE','Is_student','Sex','Employee_count','Job_42','DEGREE','ACTIVITY_TYPE','job_condition','Job_category','Terms_of_emp','economic_sector','JOB_DEP','job_desc','Nom de la commune','Nom du département','city_type','REG','Nom de la région','employer_category']

learn[cat_vars] = learn[cat_vars].astype(str)
cont_vars=['target','emp_status_indicator','AGE_2018','Working_hours','Pay','inhabitants','Lat','Long','X','Y','club_indicator']
OH_encoder = OneHotEncoder(sparse=False ,handle_unknown='ignore')
encoded_columns_learn =    OH_encoder.fit_transform(learn[cat_vars])
# encoded_columns_test =    OH_encoder.transform(test[cat_vars])
cont_learn=learn[cont_vars].to_numpy()
processed_data = np.concatenate([cont_learn, encoded_columns_learn], axis=1)
print(processed_data)
print(len(cat_vars))
print(len(cont_vars))

Y=processed_data[:,0]
X=processed_data[:,2:]
Group=processed_data[:,1]
group_kfold = GroupKFold(n_splits=5)
group_kfold.get_n_splits(X, Y, Group)
print(group_kfold)


# X=[row[1,:] for row in processed_data]


##CREATE ONE HOT ENCODING
# print(test)
#
#
# print(learn.index)
#

#
#

#
#
#


#
# # missing values
# learn.isna().sum().reset_index(name="n").plot.bar(x='index', y='n', rot=45)
# plt.ylim([0, 50000])
#
# plt.show()
# # print(learn['UID'].isna().sum())
# na_group=(learn[[
#        'job_condition', 'Job_category', 'Pay', 'Employee_count',
#        'Terms_of_emp', 'economic_sector', 'JOB_DEP', 'employer_category',
#        'job_desc']].isna())
#
# na= learn.groupby('job_condition', dropna=False).count()
#
# club= learn.groupby('job_condition', dropna=False)['CLUB'].count()
# group= learn.groupby('job_condition', dropna=False)['UID'].count()
# # print(club,group)
# # print((club/group))
#
#
#
#
#
# #Scale data and do k-means classification
# # mms = StandardScaler()
# # transformed= mms.fit(onehotlabels)
# # data_transformed = (transformed)
# # Sum_of_squared_distances = []
# # kmeans = KMeans(n_clusters=4)
# # clusters = kmeans.fit_predict(onehotlabels)
# # labels1 = pd.DataFrame(clusters)
# # labeledCustomers = pd.concat((onehotlabels,labels1),axis=1)
# # labeledCustomers = pd.concat((onehotlabels,labels1),axis=1)
# # labeledCustomers = labeledCustomers.rename({0:'labels'},axis=1)
# # labeledCustomers.to_csv('labeled_customers.csv')
# # cluster_1=labeledCustomers[clusters==0]
# # cluster_1=labeledCustomers[clusters==0]
# #
# #
