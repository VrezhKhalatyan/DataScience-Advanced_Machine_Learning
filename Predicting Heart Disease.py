
# coding: utf-8

# In[1]:


#Vrezh Khalatyan HW3 Question 1
# The following line will import LogisticRegression and DecisionTreeClassifier Classes

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier


# In[2]:


# Importing the required packages and libraries
# we will need numpy and pandas later
import numpy as np
import pandas as pd


# In[3]:


# reading a CSV file directly from Web, and store it in a pandas DataFrame:
# "read_csv" is a pandas function to read csv files from web or local device:

hearts_df = pd.read_csv('/Users/anitribunyan/Desktop/HW3/Heart_s.csv')


# In[4]:


# checking the dataset by printing every 10 lines:
hearts_df[0::10]


# In[5]:


# Creating the Feature Matrix for iris dataset:

# create a python list of feature names that would like to pick from the dataset:
feature_cols = ['Age','RestBP','Chol','RestECG', 'MaxHR', 'Oldpeak']

# use the above list to select the features from the original DataFrame
X = hearts_df[feature_cols]  

# print the first 5 rows
X.head()


# In[6]:


# select a Series of labels (the last column) from the DataFrame
y = hearts_df['AHD']

# checking the label vector by printing every 10 values
y[::10]


# In[7]:


# "my_logreg" is instantiated as an "object" of LogisticRegression "class". 
# "my_decisiontree" is instantiated as an "object" of DecisionTreeClassifier "class". 


my_logreg = LogisticRegression()

my_decisiontree = DecisionTreeClassifier(random_state=5)


# In[8]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                    random_state=4)


# In[9]:


# Training ONLY on the training set:
from sklearn.neighbors import KNeighborsClassifier
k = 3
knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(X, y)
my_logreg.fit(X_train, y_train)
my_decisiontree.fit(X_train, y_train)


# In[10]:


y_predict_knn = knn.predict(X_test)

y_predict_lr = my_logreg.predict(X_test)

y_predict_dt = my_decisiontree.predict(X_test)


# In[11]:


# We can now compare the "predicted labels" for the Testing Set with its "actual labels" to evaluate the accuracy 
# Function "accuracy_score" from "sklearn.metrics" will perform the element-to-element comparision and returns the 
# portion of correct predictions:

from sklearn.metrics import accuracy_score

score_knn = accuracy_score(y_test, y_predict_knn)
score_lr = accuracy_score(y_test, y_predict_lr)
score_dt = accuracy_score(y_test, y_predict_dt)

print('KNN = ' + str(score_knn))
print('Logistic Regression = ' + str(score_lr))
print('Decision Tree = ' + str(score_dt))
print('\n')
print('The best accuracy is with KNN and the worst is with the Decision Tree')


# In[12]:


#perform a feature engineering process called OneHotEncoding for the categorical features
X = pd.get_dummies(hearts_df)
X.head()


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                    random_state=4)

k = 3
knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(X, y)
my_logreg.fit(X_train, y_train)
my_decisiontree.fit(X_train, y_train)

y_predict_knn = knn.predict(X_test)

y_predict_lr = my_logreg.predict(X_test)

y_predict_dt = my_decisiontree.predict(X_test)

score_knn = accuracy_score(y_test, y_predict_knn)
score_lr = accuracy_score(y_test, y_predict_lr)
score_dt = accuracy_score(y_test, y_predict_dt)

print('KNN = ' + str(score_knn))
print('Logistic Regression = ' + str(score_lr))
print('Decision Tree = ' + str(score_dt))
print('\n')
print('The prediction accuracy for KNN is the same; however, for both Logistic Regression and Decision Tree the accuracy is 1.0, which can be the cause of overfitting')


# In[14]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn import metrics

my_logreg = LogisticRegression()

# function cross_val_score performs Cross Validation:
accuracy_list = cross_val_score(my_logreg, X, y, cv=10, scoring='accuracy')

print(accuracy_list)


accuracy_cv = accuracy_list.mean()

print(accuracy_cv)

