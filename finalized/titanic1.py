import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression,Perceptron,SGDClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import itertools
sns.set_theme()
traindata=pd.read_csv('train.csv')
testdata=pd.read_csv('test.csv')
#checking and understanding the data in the training data set and the test data set
traindata.head()
traindata.shape
traindata.describe()
#(traindata.describe(include=['0']))
traindata.info
traindata.isnull().sum()
testdata.shape
testdata.head()
testdata.info
testdata.isnull().sum()
#comparing the features
#comparing the individual attributes to check the survival rate
survived=traindata[traindata['Survived']==1]
not_survived=traindata[traindata['Survived']==0]
"Survived: %i (%.1f%%)"%(len(survived),float(len(survived))/len(traindata)*100.0)
"not survived: %i (%.1f%%)"%(len(not_survived),float(len(not_survived))/len(traindata)*100.0)
"Total: %i"%(len(traindata))
#checking the class of passengers survival data
traindata.Pclass.value_counts()
#grouping them categorically
traindata.groupby('Pclass').Survived.value_counts()
traindata[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean()
#plotting the graphs
sns.barplot(x='Pclass',y='Survived',data=traindata)
plt.show()
#comparing if gender has any differnces in survival
traindata.Sex.value_counts()
traindata.groupby('Sex').Survived.value_counts()
traindata[['Sex','Survived']].groupby(['Sex'],as_index=False).mean()
#plotting the results
sns.barplot(x='Sex',y='Survived',data=traindata)
plt.show()
#now combining the Pclass(passanger class)with their gender and checking their survival
tab=pd.crosstab(traindata['Pclass'],traindata['Sex'])
tab
tab.div(tab.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True)
plt.xlabel('Pclass')
plt.ylabel('Percentage')
plt.show()
#comparing the passengers that embarked on the journey and survived
traindata.Embarked.value_counts()
traindata.groupby("Embarked").Survived.value_counts()
traindata[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean()
#plotting the data
sns.barplot(x='Embarked',y='Survived',data=traindata)
plt.show()
#plotting a combined data graph
total_survived=traindata[traindata['Survived']==1]
total_not_survived=traindata[traindata['Survived']==0]
male_survived=traindata[(traindata['Survived']==1)&(traindata['Sex']=="male")]
female_survived=traindata[(traindata['Survived']==1)&(traindata['Sex']=="female")]
male_not_survived=traindata[(traindata['Survived']==0)&(traindata['Sex']=="male")]
female_not_survived=traindata[(traindata['Survived']==0)&(traindata['Sex']=="female")]
plt.figure(figsize=[15,5])
plt.subplot(111)
sns.distplot(total_survived['Age'].dropna().values,bins=range(0,81,1),kde=False,color='blue')
sns.distplot(total_not_survived['Age'].dropna().values,bins=range(0,81,1),kde=False,color='red',axlabel='Age')
plt.figure(figsize=[15,5])
plt.subplot(121)
sns.distplot(female_survived['Age'].dropna().values,bins=range(0,81,1),kde=False,color='blue')
sns.distplot(female_not_survived['Age'].dropna().values,bins=range(0,81,1),kde=False,color='red',axlabel='Female Age')
plt.subplot(122)
sns.distplot(male_survived['Age'].dropna().values,bins=range(0,81,1), kde=False,color='blue')
sns.distplot(male_not_survived['Age'].dropna().values,bins=range(0,81,1),kde=False,color='red',axlabel='Male Age')
plt.show()
#corellating the features
plt.figure(figsize=(15,6))
sns.heatmap(traindata.drop('PassengerId',axis=1).corr(),vmax=0.6,square=True,annot=True)
plt.show()
#feature extraction to select the appropriate feature to train the model
test_train=[traindata,testdata]
for dataset in test_train:
    dataset['Title']=dataset.Name.str.extract('([A-Za-z]+)\.')
traindata.head()
pd.crosstab(traindata['Title'],traindata['Sex'])
for dataset in test_train:
    dataset['Title']=dataset['Title'].replace(['Lady','Countess','Capt','Col'\
        'Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Other')
    dataset['Title']=dataset['Title'].replace('Mlle','Miss')
    dataset['Title']=dataset['Title'].replace('Ms','Miss')
    dataset['Title']=dataset['Title'].replace('Mr','Mrs')
(traindata[['Title','Survived']].groupby(['Title'],as_index=False).mean())
title_map={"Mr":1,
            "Miss":2,
            "Mrs":3,
            "Master":4,
            "Other":5}
for dataset in test_train:
    dataset['Title']=dataset['Title'].map(title_map)
    dataset['Title']=dataset['Title'].fillna(0)
for dataset in test_train:
    dataset['Sex']=dataset['Sex'].map({'female':1,'male':0}).astype(int)
#finding the unique values in the embarked feature
traindata.Embarked.unique()
traindata.Embarked.value_counts()
for dataset in test_train:
    dataset['Embarked']=dataset['Embarked'].fillna('S')
for dataset in test_train:
    dataset['Embarked']=dataset['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)
#mapping age and also creating a range of ages for easy classification
for dataset in test_train:
    age_avg=dataset['Age'].mean()
    age_std=dataset['Age'].std()
    age_null_count=dataset['Age'].isnull().sum()
    age_null_random=np.random.randint(age_avg-age_std,age_avg+age_std,size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])]=age_null_random
    dataset['Age']=dataset['Age'].astype(int)
traindata['AgeBand']=pd.cut(traindata['Age'],5)
print(traindata[['AgeBand','Survived']].groupby(['AgeBand'],as_index=False).mean())
#mapping age to ageband feature and commiting it to the dataframe
for dataset in test_train:
    dataset.loc[dataset['Age']<=16,'Age']=0
    dataset.loc[(dataset['Age']>16)&(dataset['Age']<=32),'Age']=1
    dataset.loc[(dataset['Age']>32)&(dataset['Age']<=48),'Age']=2
    dataset.loc[(dataset['Age']<=16)&(dataset['Age']<=64),'Age']=3
    dataset.loc[dataset['Age']>64,'Age']=4
#mapping fare 
for dataset in test_train:
    dataset['Fare']=dataset['Fare'].fillna(traindata['Fare'].median())
traindata['FareBand']=pd.qcut(traindata['Fare'],4)
print(traindata[['FareBand','Survived']].groupby(['FareBand'],as_index=False).mean())
for dataset in test_train:
    dataset.loc[ dataset['Fare']<=7.91,'Fare'] = 0
    dataset.loc[(dataset['Fare']>7.91)&(dataset['Fare']<=14.454), 'Fare']=1
    dataset.loc[(dataset['Fare']>14.454)&(dataset['Fare']<=31), 'Fare']=2
    dataset.loc[ dataset['Fare']>31, 'Fare']=3
    dataset['Fare']=dataset['Fare'].astype(int)
#Sibsp
for dataset in test_train:
    dataset['FamilySize']=dataset['SibSp']+ dataset['Parch'] + 1
print (traindata[['FamilySize','Survived']].groupby(['FamilySize'],as_index=False).mean())
for dataset in test_train:
    dataset['IsAlone']=0
    dataset.loc[dataset['FamilySize']==1,'IsAlone'] = 1
print (traindata[['IsAlone','Survived']].groupby(['IsAlone'],as_index=False).mean())
#feature selection
features_drop = ['Name','SibSp','Parch','Ticket', 'Cabin','FamilySize']
traindata=traindata.drop(features_drop,axis=1)
testdata=testdata.drop(features_drop,axis=1)
traindata=traindata.drop(['PassengerId','AgeBand','FareBand'],axis=1)
traindata.head()
testdata.head()
#defining training and testing sets
x_traindata=traindata.drop('Survived',axis=1)
y_traindata=traindata['Survived']
x_testdata=testdata.drop('PassengerId',axis=1).copy()
x_traindata.shape
y_traindata.shape
x_testdata.shape
#checking the accuracy of various classification algorithims
#logistic regression
c=LogisticRegression()
c.fit(x_traindata,y_traindata)
y_pred_lg=c.predict(x_testdata)
a_lg=round(c.score(x_traindata,y_traindata)*100,2)
print(a_lg)
#support vector machine
c=SVC()
c.fit(x_traindata,y_traindata)
y_pred_svc=c.predict(x_testdata)
a_svm=round(c.score(x_traindata,y_traindata)*100,2)
print(a_svm)
#linear svm
c=LinearSVC()
c.fit(x_traindata,y_traindata)
y_pred_lsvm=c.predict(x_testdata)
a_lsvm=round(c.score(x_traindata,y_traindata)*100,2)
print(a_lsvm)
#k nearest neighbours
c=KNeighborsClassifier(n_neighbors=3)
c.fit(x_traindata,y_traindata)
y_pred_knn=c.predict(x_testdata)
a_knn=round(c.score(x_traindata,y_traindata)*100,2)
print(a_knn)
#decision treee
c=DecisionTreeClassifier()
c.fit(x_traindata,y_traindata)
y_pred_dtc=c.predict(x_testdata)
a_dtc=round(c.score(x_traindata,y_traindata)*100,2)
print(a_dtc)
#random forest
c=RandomForestClassifier(n_estimators=100)
c.fit(x_traindata,y_traindata)
y_pred_rfc=c.predict(x_testdata)
a_rfc=round(c.score(x_traindata,y_traindata)*100,2)
print(a_rfc)
#gaussian naive bayes
c=GaussianNB()
c.fit(x_traindata,y_traindata)
y_pred_nb=c.predict(x_testdata)
a_nb=round(c.score(x_traindata,y_traindata)*100,2)
print(a_nb)
#perceptron
c=Perceptron(max_iter=5,tol=None)
c.fit(x_traindata,y_traindata)
y_pred_p=c.predict(x_testdata)
a_p=round(c.score(x_traindata,y_traindata)*100,2)
print(a_p)
#sgd(stochastic gradient descent)
c=SGDClassifier(max_iter=5,tol=None)
c.fit(x_traindata,y_traindata)
y_pred_sgd=c.predict(x_testdata)
a_sgd=round(c.score(x_traindata,y_traindata)*100,2)
print(a_sgd)
#confusion matrix
print("accuracy: %i %% \n"%a_rfc)
cl_names=['Survived','Not Survived']
#comuting the confusion matrix
c_matrix=confusion_matrix(y_traindata,y_pred_rfc)
np.set_printoptions(precison=2)
print("the confusion matrix \n")
print(c_matrix)
print('')
c_matrix_per=c_matrix.astype('float')/c_matrix.sum(axis=1)[:,np.newaxis]
print("confusion matrix percentages of each \n")
print(c_matrix_per)
print('')
rue_class_names = ['True Survived', 'True Not Survived']
predicted_class_names = ['Predicted Survived', 'Predicted Not Survived']

df_c_matrix=pd.DataFram(c_matrix, index = true_class_names,columns = predicted_class_names)
df_c_matrix_percent=pd.DataFrame(c_matrix_percent,index = true_class_names,columns = predicted_class_names)
plt.figure(figsize=(15,5))
plt.subplot(121)
sns.heatmap(df_c_matrix,annot=True,fmt='d')
plt.subplot(122)
sns.heatmap(df_c_matrix_percent,annot=True)
plt.show()
#comparing models
