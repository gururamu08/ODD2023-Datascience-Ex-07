# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

## Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

## ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


## CODE
```
Developed By:GURUMOORTHI R
Register no:212222230042
```
```
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest

df=pd.read_csv("titanic_dataset.csv")
df1=df.drop({"Name","Sex","Ticket","Cabin","Embarked"},axis=1)
df1.columns
df1['Age'].isnull().sum()
df1['Age'].fillna(method='ffill')
df1['Age']=df1['Age'].fillna(method='ffill')
df1['Age'].isnull().sum()

feature =SelectKBest(mutual_info_classif,k=3)
df1.columns
data=pd.read_csv("titanic_dataset.csv")
data

data=data.dropna()
x=data.drop(['Survived','Name','Ticket'],axis=1)
y=data['Survived']
x
y

data["Sex"]=data["Sex"].astype("category")
data["Cabin"]=data["Cabin"].astype("category")
data["Embarked"]=data["Embarked"].astype("category")

data["Sex"]=data["Sex"].cat.codes
data["Cabin"]=data["Cabin"].cat.codes
data["Embarked"]=data["Embarked"].cat.codes

data

from sklearn.feature_selection import chi2
selector=SelectKBest(score_func=chi2,k=5)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features")
print(selected_features)

x.info()

x=x.drop(["Sex","Cabin","Embarked"],axis=1)
from sklearn.feature_selection import SelectKBest,f_regression

selector=SelectKBest(score_func=f_regression,k=5)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features")
print(selected_features)

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
sfm=SelectFromModel(model,threshold='mean')
sfm.fit(x,y)
selected_features=x.columns[sfm.get_support()]
print("Selected Features:")
print(selected_features)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model=LogisticRegression()
num_features_to_remove=2
rfe=RFE(model,n_features_to_select=(len(x.columns)-num_features_to_remove))

rfe.fit(x,y)
selected_features=x.columns[rfe.support_]

print("Selected Features:")
print(selected_features)

```
## OUTPUT

![image](https://github.com/kavinesh8476/ODD2023-Datascience-Ex-07/assets/118466561/088d8d1c-bb5f-4db7-898f-713adecf5fae)

![image](https://github.com/kavinesh8476/ODD2023-Datascience-Ex-07/assets/118466561/d09ac470-306f-4f93-a847-04c7a26329bb)

![image](https://github.com/kavinesh8476/ODD2023-Datascience-Ex-07/assets/118466561/1639c43c-0a52-494f-95e2-f224e58fb538)

![image](https://github.com/kavinesh8476/ODD2023-Datascience-Ex-07/assets/118466561/99db142b-cbd0-4a3b-a742-3a3d87aa4e7e)

![image](https://github.com/kavinesh8476/ODD2023-Datascience-Ex-07/assets/118466561/2e8ae3da-4b5e-43d2-ba5d-d8c180f7a28d)

![image](https://github.com/kavinesh8476/ODD2023-Datascience-Ex-07/assets/118466561/d70da928-2f5d-4316-8f43-cbf653fa2bdc)

![image](https://github.com/kavinesh8476/ODD2023-Datascience-Ex-07/assets/118466561/fca6a195-79d5-4507-8ec7-6ec9ed24640f)

![image](https://github.com/kavinesh8476/ODD2023-Datascience-Ex-07/assets/118466561/f945802f-3d39-4829-8dd6-26d0135c167a)

![image](https://github.com/kavinesh8476/ODD2023-Datascience-Ex-07/assets/118466561/aafb74aa-8c04-4c5a-a288-df4638d15b5b)

![image](https://github.com/kavinesh8476/ODD2023-Datascience-Ex-07/assets/118466561/a5809c85-083b-400e-ab5d-f54c0b9fb35f)

![image](https://github.com/kavinesh8476/ODD2023-Datascience-Ex-07/assets/118466561/02b95ffd-e1ef-471c-a921-c82ab11e4213)

## RESULT
Thus, Sucessfully performed the various feature selection techniques on a given dataset.
