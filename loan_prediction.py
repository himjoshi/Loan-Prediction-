# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 18:20:30 2017

@author: HIMANSHU
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import mode

#importing dataset
df = pd.read_csv('loan_train.csv')

#dropping Loan_id column
df.drop(df.columns[[0]], axis=1, inplace=True)

#check for null values
def num_missing(x):
  return sum(x.isnull())
df.apply(num_missing, axis = 0)

#Counting persons depending on Self_Employed criteria
df['Self_Employed'].value_counts()

#imputing Self_Employed column
df['Self_Employed'].fillna('No', inplace = True)

#check for remaining null values in Self_Employed
df['Self_Employed'].isnull().sum()

#Viusalise and fill Loan_Amount
df.boxplot(column = 'LoanAmount' , by = ['Self_Employed', 'Education'])
table = df.pivot_table(values = 'LoanAmount', index = 'Self_Employed', columns = 'Education', aggfunc = np.median)
print (table)
def fage(x):
    return table.loc[x['Self_Employed'], x['Education']]
df['LoanAmount'].fillna(df.apply(fage, axis = 1), inplace = True)

#check for remaining null valued in LoanAmount
df['LoanAmount'].isnull().sum()

#Visualising Loan Amount distribution
df.hist(column = 'LoanAmount')

#Making the normal distribution of loan Amount
df['LoanAmount_log'] = np.log(df['LoanAmount'])
df.hist(column = 'LoanAmount_log')

#Changing applicant and coapplicant income, reducing one perameter
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df.hist(column = 'TotalIncome')
df['TotalIncome_log']= np.log(df['TotalIncome'])

#Imputing Gender column
df['Gender'].fillna('Male', inplace = True)

#Visualising and Imputing married column
df.hist(column = 'Married', by= ['Gender'])
df.loc[(pd.isnull(df['Married'])) & (df['Gender']=='Male'), 'Married'] = 'Yes'
df.loc[(pd.isnull(df['Married'])) & (df['Gender']=='Female'), 'Married'] = 'No'


#Imputing dependents column
df['Dependents'] = df['Dependents'].map({"0":0, "1":1, "2":2, "3+":3})
df.hist(column = 'Dependents', by = ['Gender'])
df['Dependents'].fillna(0, inplace = True)

#Imputing Credit_History values
df.hist(column = 'Loan_Status', by = ['Credit_History'])
df.loc[(pd.isnull(df['Credit_History'])) & (df['Loan_Status']=='Y'), 'Credit_History'] = 1
df.loc[(pd.isnull(df['Credit_History'])) & (df['Loan_Status']=='N'), 'Credit_History'] = 0

#including budget column and distributing it normally
df['budget'] = df['TotalIncome']/df['LoanAmount']
df.hist(column = 'budget_log')
df['budget_log']= np.log(df['budget'])


#Imputing loan amount term
df['Loan_Amount_Term'].fillna(360, inplace = True)
df['Loan_Amount_Term'].value_counts()

#drop unnecessary columns
df.drop(df.columns[[5,6,13,15]], axis=1, inplace=True)


#perform label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
col = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
for i in col:
    df[i]= le.fit_transform(df[i])

#one hot encoding
from sklearn.preprocessing import OneHotEncoder
oe = OneHotEncoder(categorical_features = [2])


 #Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold                 #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

#splitting the outcome and training variables
y = df.iloc[:,9].values
df.drop(df.columns[[9]], axis=1, inplace=True)
X = df.iloc[:,:].values

#classifiers
log_class = LogisticRegression()
rf_class = RandomForestClassifier(n_estimators = 10)
dt_class = DecisionTreeClassifier()

#splitting the data set
from sklearn.model_selection import train_test_split
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state =0 )

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#fitting logistic regression to the training set
log_class.fit(X_train, y_train)  
y_pred = log_class.predict(X_test)
metrics.accuracy_score(y_test,y_pred)

#fitting random forest to the training set
rf_class.fit(X_train, y_train)  
y_pred = rf_class.predict(X_test)
metrics.accuracy_score(y_test,y_pred)

#fitting decision tree to the training set
dt_class.fit(X_train, y_train)  
y_pred = dt_class.predict(X_test)
metrics.accuracy_score(y_test,y_pred)

#performing crossvalidation on logistic regression classifer
from sklearn.model_selection import cross_val_score
scores = cross_val_score(log_class, X_train,y_train, cv = 10, scoring = 'accuracy')
print (scores.mean())

#performing crossvalidation on random forest classifier
from sklearn.model_selection import cross_val_score
scores = cross_val_score(rf_class, X_train,y_train, cv = 10, scoring = 'accuracy')
print (scores.mean())

#performing crossvalidation on decision tree classifier
from sklearn.model_selection import cross_val_score
scores = cross_val_score(dt_class, X_train,y_train, cv = 10, scoring = 'accuracy')
print (scores.mean())