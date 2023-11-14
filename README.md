# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

## Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


## CODE and OUTPUT
```
DEVELOPED BY: GURUMOORTHI R
REG NO: 212222230042
```
```
import pandas as pd
```
```
df=pd.read_csv("titanic_dataset.csv")
df
```

![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-07/assets/118904526/b4566409-e2dd-499e-82a5-39d8daff155e)

```
df.columns
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-07/assets/118904526/be73f7e1-a04d-4eb6-895f-bf64baeb568a)

```
df.shape
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-07/assets/118904526/6aa71ac3-e5b7-48c3-96ea-21ab36b116ed)

```
X=df.drop("Survived",1)
Y=df['Survived']
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-07/assets/118904526/d35e1f77-471e-4b22-82a4-ffed00ac3ea0)

```
df1=df.drop(["Name","Sex","Ticket","Cabin","Embarked"],axis=1)
```
```
df1.columns
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-07/assets/118904526/5fdd95f9-3c6d-486d-9bd9-cd987f33f2e2)

```
df1['Age'].isnull().sum()
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-07/assets/118904526/ed42a4da-be8f-4bd8-9f22-d2e8ec549d3a)
```
df1['Age'].fillna(method='ffill')
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-07/assets/118904526/9d969fc5-f727-4e35-b627-495e4c772655)
```
df1['Age']=df1['Age'].fillna(method='ffill')
```
```
df1['Age'].isnull().sum()
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-07/assets/118904526/90193309-c005-4549-9823-b7598c5cb3c1)

```
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
```
```
feature=SelectKBest(mutual_info_classif,k=3)
feature
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-07/assets/118904526/e04c5904-4a7d-4acd-b0d7-4d9dc299f757)
```
cols=df1.columns.tolist()
cols[-1],cols[1]=cols[1],cols[-1]
df1=df1[cols]
df1.columns
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-07/assets/118904526/5bf3641f-ac40-409b-b3e0-b0941e6cd3ba)

```
X=df1.iloc[:,0:6]
Y=df1.iloc[:,6]
```
```
X.columns
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-07/assets/118904526/e0b27216-7013-42e9-b0c6-155c43c2c2b5)
```
Y=Y.to_frame()
Y.columns
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-07/assets/118904526/6c6bbcb7-116f-4862-b7b4-c7b9960c38b7)

**Chi2 method**
```
from sklearn.feature_selection import chi2
data=df.copy()
```
```
data=data.dropna()
```
```
X=data.drop(['Survived','Name','Ticket'],axis=1)
Y=data['Survived']
X
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-07/assets/118904526/70d62334-d2dd-4c91-ba35-2b3c820260dd)
```
data['Sex']=data['Sex'].astype('category')
data['Cabin']=data['Cabin'].astype('category')
data['Embarked']=data['Embarked'].astype('category')
```
```
data['Sex']=data['Sex'].cat.codes
data['Cabin']=data['Cabin'].cat.codes
data['Embarked']=data['Embarked'].cat.codes
```
```
data
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-07/assets/118904526/3c4ec200-8cc0-45bc-894f-4a6e778bd7a4)
```
k=5
selector=SelectKBest(score_func=chi2,k=k)
x_new=selector.fit_transform(X,Y)
```
```
selected_feature_indices=selector.get_support(indices=True)
selected_features=X.columns[selected_feature_indices]
print("Selected_features")
print(selected_features)
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-07/assets/118904526/1395f06d-9c01-4a53-9b6f-893217f6631a)
```
X.info()
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-07/assets/118904526/54a2638a-cb17-4ec8-b2b1-88b3aeb2c296)

**Correlation coefficient**
```
from sklearn.feature_selection import f_regression
selector=SelectKBest(score_func=f_regression,k=5)
x_new=selector.fit_transform(X,Y)
```
```
selected_feature_indices=selector.get_support(indices=True)
selected_features=X.columns[selected_feature_indices]
print("selected_features")
print(selected_features)
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-07/assets/118904526/1cf9853e-d725-4320-9e6b-c31a4638a77c)

**Mutual information**
```
from sklearn.feature_selection import mutual_info_classif
selector=SelectKBest(score_func=mutual_info_classif,k=5)
x_new=selector.fit_transform(X,Y)
```
```
selected_feature_indices=selector.get_support(indices=True)
selected_features=X.columns[selected_feature_indices]
print("Selected features:")
print(selected_features)
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-07/assets/118904526/4d525acf-457a-4535-88ba-4da10a93c42b)
```
from sklearn.feature_selection import SelectPercentile,chi2
selector=SelectPercentile(score_func=chi2,percentile=10)
X_new=selector.fit_transform(X,Y)
```
**Forward selection**
```
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
```
```
model=RandomForestClassifier()
sfm=SelectFromModel(model,threshold='mean')
sfm.fit(X,Y)
selected_features=X.columns[sfm.get_support()]
print("selected features")
print(selected_features)
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-07/assets/118904526/c313f8c4-53c8-45ca-ab6b-03c2b5bb76c7)

**Backward elimination**
```
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
```
```
model=LogisticRegression()
num_features_to_remove=2
rfe=RFE(model,n_features_to_select=(len(X.columns))-num_features_to_remove)
rfe.fit(X,Y)
selected_features=X.columns[rfe.support_]
print("Selected features")
print(selected_features)
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-07/assets/118904526/53bc4c0f-d3f4-4de9-a5dd-a5da60ad7df0)

**EMBEDDED METHODS**
```
from sklearn.linear_model import Lasso
```
```
model=Lasso(alpha=0.01)
model.fit(X,Y)
feature_coefficients=model.coef_
selected_features=X.columns[feature_coefficients!=0]
print("Selected features")
print(selected_features)
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-07/assets/118904526/1c64d303-0bc2-414d-90fd-fba7018eba26)
```
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X,Y)
feature_importances=model.feature_importances_
threshold=0.15
selected_features=X.columns[feature_importances>threshold]
print("selected features")
print(selected_features)
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-07/assets/118904526/6e378e75-7a45-47ab-b4a0-151b2ff40d8c)

## RESULT
Hence various feature selection techniques were performed on the dataset.
