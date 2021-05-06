#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Adam


"""
def cls(): return print("\033[2J\033[;H", end='')

cls()

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import pandas as pd

# Make column names for dataset
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'labelvalue']


# load dataset from CSV file 
pima = pd.read_csv("pima-indians-diabetes-database.csv", header=None, names=col_names)
print(pima.head(), '\n')


#split dataset in features and target variable 
# select 5 if possible else 4 featured columns
feature_cols = ['pregnant','insulin','bmi','age','glucose']
X = pima[feature_cols]
#X2 = pima[['pregnant','insulin']] # Features
#X3 = pima[0:3]
#X4 =pima.loc[0:3,['pregnant','insulin']]  # 0:3 -> 4 rows
#X5 = pima.iloc[0:3,0:1]  # 0:3 -> 3 rows, one column
y = pima.labelvalue # Target variable


# splitting the data so there is 40% - test data and 60% train data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state=0)


logreg = LogisticRegression()

# fit the model with the training data  data
logreg.fit(X_train,y_train)


y_pred=logreg.predict(X_test)

#print(y_pred)
#print("accuracy score using sklearn model score",logreg.score(X_test,y_test))
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

print('This is the Cnf Matrix:\n' , cnf_matrix, '\n')


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)


# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlOrRd" ,fmt='g')
#sns.heatmap(cnf_matrix, annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred), '\n')
print(metrics.classification_report(y_test,y_pred))
plt.figure()



y_pred_proba = logreg.predict_proba(X_test)[:,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.plot([0,1],[0,1], 'k--')
plt.plot([0,1])


plt.legend(loc=4)
plt.show()













