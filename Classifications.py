# -*- coding: utf-8 -*-
"""
@author: Adam 

"""
def cls(): return print("\033[2J\033[;H", end='')

cls()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn import tree
import sklearn.svm as svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.ensemble import ExtraTreesClassifier

cmap = plt.cm.bone
plot_idx = 1
C = 1.0
Gamma = 0.7

iris = load_iris()
models = [DecisionTreeClassifier(criterion="gini"),
          RandomForestClassifier(), 
          ExtraTreesClassifier(),
          AdaBoostClassifier(DecisionTreeClassifier())]

labelz = ['setosa', 'versicolor', 'virginica']

class_report = ['Logistic Regression Classification Report:',
           'KNN Classification Report:',
           'Linear SVM Classification Report:']

report_classifiers = [LogisticRegression(), 
                      KNeighborsClassifier(), 
                      svm.SVC(kernel='poly', C=C,gamma=Gamma) ]

iris_columns = ['(sepal length , sepal width)', 
                '(sepal length , petal length)', 
                '(petal length , petal width)']

i = 0

for pair in ([0, 1], [0, 2], [2, 3]):
    
    for model in models:
        #for title in ([titles]):
        X = iris.data[:, pair]
        y = iris.target
         # Shuffle
        idx = np.arange(X.shape[0])
        np.random.seed(20)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]
        
        # Standardize
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / std
        
        # Train
        model.fit(X, y)
        
        
        scores = model.score(X, y)
        
        model_title = str(type(model)).split(
            ".")[-1][:-2][:-len("Classifier")]

        model_details = model_title
        if hasattr(model, "estimators_"):
                model_details += " with {} estimators".format(
                    len(model.estimators_))
        #print(model_details + " with features", pair,
                 # "has a score of", scores)
        print(model_title, 'model comparing', iris_columns[i])
        
        plt.subplot(3, 4, 1)
        if 1 <= len(models):
                # Add a title at the top of each column
                plt.title(model_title, fontsize=15)
        
            # Now plot the decision boundary using a fine mesh as input to a
            # filled contour plot
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                 np.arange(y_min, y_max, 0.1))
        
        if isinstance(model, DecisionTreeClassifier):
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=cmap)
        else:
            # Choose alpha blend level with respect to the number
            # of estimators
            # that are in use (noting that AdaBoost can use fewer estimators
            # than its maximum if it achieves a good enough fit early on)
            estimator_alpha = 1.0 / len(model.estimators_)
            for tree in model.estimators_:
                Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                cs = plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)

        # Build a coarser grid to plot a set of ensemble classifications
        # to show how these are different to what we see in the decision
        # surfaces. These points are regularly space and do not have a
        # black outline
        xx_coarser, yy_coarser = np.meshgrid(
            np.arange(x_min, x_max, 0.5),
            np.arange(y_min, y_max, 0.5))
        Z_points_coarser = model.predict(np.c_[xx_coarser.ravel(),
                                         yy_coarser.ravel()]
                                         ).reshape(xx_coarser.shape)
        cs_points = plt.scatter(xx_coarser, yy_coarser, s=15,
                                c=Z_points_coarser, cmap=cmap,
                                edgecolors="none")        
            
        plt.scatter(X[:, 0], X[:, 1], c=y,
                                cmap=ListedColormap(['r', 'y', 'g']),
                                edgecolor='k', s=35)
        plot_idx += 1 

        fig_size = plt.rcParams["figure.figsize"]
        fig_size[0] = 12
        fig_size[1] = 10
        plt.rcParams["figure.figsize"] = fig_size
        
        setosa = mpatches.Patch(color='red', label='Setosa')
        versicolor = mpatches.Patch(color='yellow', label='versicolor')
        virginica = mpatches.Patch(color='green', label='verginica')
        plt.legend(handles=[setosa, versicolor, virginica], 
                   bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        
       
        plt.show()
    print('\n')    
    le = LabelEncoder()
    labels = le.fit_transform(labelz)
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, random_state=20)
    report_classifiers[i].fit(X_train, Y_train)
    predictions = report_classifiers[i].predict(X_validation)
    print( class_report[i], '\n\n', 
          classification_report(Y_validation, predictions, 
                                target_names=le.classes_ ))
    i = i + 1
    
    
    
    

