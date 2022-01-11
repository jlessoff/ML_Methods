import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import  StandardScaler
from sklearn import metrics
from sklearn import preprocessing

import numpy as np
learn=pd.read_csv("learn_dataset.csv")
learn_jobs=pd.read_csv("learn_dataset_job.csv")
learn_sport=pd.read_csv("learn_dataset_sport.csv")
learn_emp=pd.read_csv("learn_dataset_Emp.csv")

print(len(learn))
learn=pd.merge(learn,learn_jobs,on='UID',how='left')
learn=pd.merge(learn,learn_sport,on='UID',how='left')
print(learn.columns)

# missing values
learn.isna().sum().reset_index(name="n").plot.bar(x='index', y='n', rot=45)
plt.ylim([0, 50000])
#
# plt.show()
print(learn['UID'].isna().sum())
na_group=(learn[[
       'job_condition', 'Job_category', 'Pay', 'Employee_count',
       'Terms_of_emp', 'economic_sector', 'JOB_DEP', 'employer_category',
       'job_desc']].isna())

na= learn.groupby('job_condition', dropna=False).count()

club= learn.groupby('job_condition', dropna=False)['CLUB'].count()
group= learn.groupby('job_condition', dropna=False)['UID'].count()
print(club,group)
print((club/group))

# TODO: create a OneHotEncoder object, and fit it to all of X
le = preprocessing.LabelEncoder()
onehotlabels = learn.apply(le.fit_transform)
# X_2.head()
# # 1. INSTANTIATE
# enc = preprocessing.OneHotEncoder()
#
# # 2. FIT
# enc.fit(X_2)
#
# # 3. Transform
# onehotlabels = enc.transform(X_2).toarray()
# onehotlabels.shape
# print(onehotlabels)





# as you can see, you've the same number of rows 891
# but now you've so many more columns due to how we changed all the categorical data into numerical data
kmeans = KMeans(3)
clusters = kmeans.fit_predict(onehotlabels)
labels = pd.DataFrame(clusters)
labeledCustomers = pd.concat((onehotlabels,labels),axis=1)
labeledCustomers = labeledCustomers.rename({0:'labels'},axis=1)
kmeans = KMeans(3)
clusters = kmeans.fit_predict(onehotlabels)
labels1 = pd.DataFrame(clusters)
labeledCustomers = pd.concat((onehotlabels,labels1),axis=1)
labeledCustomers = labeledCustomers.rename({0:'labels'},axis=1)
print(labeledCustomers)

#

# find ideal number of clusters using silhouette scores,elbow method
mms = StandardScaler()
# mms.fit(learn)
data_transformed = (onehotlabels)
Sum_of_squared_distances = []

K = range(2,15)
CS_scores = []
for k in K:
    km = KMeans(n_clusters=k, init='k-means++', random_state=1)
    cluster_found = km.fit_predict(data_transformed)
    Sum_of_squared_distances.append(km.inertia_)
    CS_scores.append(metrics.calinski_harabasz_score(data_transformed, km.labels_))



plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.title('Find Optimal k using elbow')
plt.savefig('elbow_score.png')
plt.show()
#
# #cluster data with 5 clusters
# kmeans = KMeans(n_clusters=5, random_state=1)
# y = kmeans.fit_predict(data_transformed)


# You must normalize the data before applying the fit method




# ['JOB_DEP'].unique())