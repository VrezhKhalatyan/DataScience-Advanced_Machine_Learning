
# coding: utf-8

# # Predicting With Decision_Tree & Cross_Validation

# The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

# Cross Validation is used to assess the predictive performance of the models and and to judge how they perform outside the sample to a new data set also known as test data.
# 
# The motivation to use cross validation techniques is that when we fit a model, we are fitting it to a training dataset. Without cross validation we only have information on how does our model perform to our in-sample data. Ideally we would like to see how does the model perform when we have a new data in terms of accuracy of its predictions.

# In[1]:



# Importing libraries and packages:
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


bSharing_df = pd.read_csv('/Users/anitribunyan/Downloads/train.csv')
bSharing_df.head()


# In[3]:


# create a python list of feature names that would like to pick from the dataset:
feature_cols = ['season','holiday',
                'workingday','weather','temp',
                'atemp','humidity','windspeed', 'casual', 'registered']

# use the above list to select the features from the original DataFrame
X = bSharing_df[feature_cols] 

# select a Series of labels (the last column) from the DataFrame
y = bSharing_df['count']

# print the first 5 rows
print(X.head())
print(y.head())


# In[4]:


my_decisionTree = DecisionTreeClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=2)
my_decisionTree.fit(X_train, y_train)
y_predict_dt = my_decisionTree.predict(X_test)
score_dt = accuracy_score(y_test, y_predict_dt)


# function cross_val_score performs Cross Validation:
accuracy_list = cross_val_score(my_decisionTree, X, y, cv=10, scoring='accuracy')

print(score_dt)
print(accuracy_list)


# In[5]:


# use average of accuracy values as final result
accuracy_cv = accuracy_list.mean()

print("Prediction Accuracy Using DecisionTree & Cross_Validation: " + str(accuracy_cv))


# # Predicting With RandomForest Classifier

# In[10]:


my_RandomForest = RandomForestClassifier(n_estimators = 100, bootstrap = True, random_state=3)
my_RandomForest.fit(X_train, y_train)
y_predict_RF = my_RandomForest.predict(X_test)
accuracy_RF =accuracy_score(y_test, y_predict_RF)

# function cross_val_score performs Cross Validation:
rf_accuracy_list = cross_val_score(my_decisionTree, X, y, cv=10, scoring='accuracy')
accuracy_rf = rf_accuracy_list.mean()

print("Prediction Accuracy with RandomForest Using Cross_Validation: " + str(accuracy_rf))
print("Prediction Accuracy using only RandomForest Classifier: " + str(accuracy_RF))

