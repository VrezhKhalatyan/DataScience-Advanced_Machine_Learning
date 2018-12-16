
# coding: utf-8

# In[43]:


# Vrezh Khalatyan HW4 Question 1
# Importing libraries and packages:

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[44]:


# reading a CSV file directly from Web (or local drive), and store it in a pandas DataFrame:
# "read_csv" is a pandas function to read csv files from web or local drive:

cancer = pd.read_csv('/Users/anitribunyan/Downloads/HW4/Cancer.csv')

cancer.head()


# In[45]:


# create a python list of feature names that would like to pick from the dataset:
feature_cols = ['Clump_Thickness','Uniformity_of_Cell_Size','Uniformity_of_Cell_Shape',
                'Marginal_Adhesion','Single_Epithelial_Cell_Size','Bare_Nuclei',
                'Bland_Chromatin','Normal_Nucleoli','Mitoses']

# use the above list to select the features from the original DataFrame
X = cancer[feature_cols] 

# select a Series of labels (the last column) from the DataFrame
y = cancer['Malignant_Cancer']

# Randomly splitting the original dataset into training set and testing set:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=3)

# print the size of the traning set:
print(X_train.shape)
print(y_train.shape)

# print the size of the testing set:
print(X_test.shape)
print(y_test.shape)


# In[46]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

my_DecisionTree = DecisionTreeClassifier(random_state=3)

my_DecisionTree.fit(X_train, y_train)

y_predict_dt = my_DecisionTree.predict(X_test)

score_dt = accuracy_score(y_test, y_predict_dt)

print("Decision Tree Accuracy: " + str(score_dt))


# In[47]:


from sklearn.utils import resample
from sklearn.metrics import accuracy_score
#from sklearn.ensemble import VotingClassifier
#perform a new Ensemble Learning method called “Bagging”
predData = pd.DataFrame()
for i in range(19):
    bootstrap_size = int(0.8*len(X_train))
    resample_X_train = resample(X_train, n_samples = bootstrap_size , random_state=i , replace = True)
    resample_y_train = resample(y_train, n_samples = bootstrap_size , random_state=i , replace = True) 
    Base_DecisionTree = DecisionTreeClassifier(random_state=3)
    Base_DecisionTree.fit(resample_X_train, resample_y_train)
    predData[str(i)] = Base_DecisionTree.predict(X_test)  


# In[48]:


#Performing voting on prediction results
sampleSum = pd.Series(predData.sum(axis=1))
predData = predData.assign(count = sampleSum)
predAccuracy = []

for i in range(len(predData)):
    if(sampleSum[i]>= 10):
        predAccuracy.append(1)
    else:
        predAccuracy.append(0)
        
predData = predData.assign(final = pd.Series(predAccuracy))

scorePredData = accuracy_score(y_test, predData['final'])
print("Final Prediction Score: " + str(scorePredData))


# In[52]:


print(predData)


# In[49]:


from sklearn.ensemble import RandomForestClassifier
my_RandomForest = RandomForestClassifier(n_estimators = 19, bootstrap = True, random_state=3)
my_RandomForest.fit(X_train, y_train)
y_predict_rf = my_RandomForest.predict(X_test)
score_rf = accuracy_score(y_test, y_predict_rf)
print('Random Forest Accuracy:',score_rf)

