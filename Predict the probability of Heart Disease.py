
# coding: utf-8

# In[2]:


#Vrezh Khalatyan HW4_Question2
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[3]:


# reading a CSV file directly from Web (or local drive), and store it in a pandas DataFrame:
# "read_csv" is a pandas function to read csv files from web or local drive:

heart_short_df = pd.read_csv('/Users/anitribunyan/Downloads/HW4/Heart_short.csv')

heart_short_df.head()


# In[4]:


# Creating the Feature Matrix for iris dataset:

# create a python list of feature names that would like to pick from the dataset:
feature_cols = ['Age','RestBP','Chol','RestECG', 'MaxHR', 'Oldpeak']

# use the above list to select the features from the original DataFrame
X = heart_short_df[feature_cols]  

# print the first 5 rows
X.head()


# In[5]:


# select a Series of labels (the last column) from the DataFrame
y = heart_short_df['AHD']
normalized = preprocessing.scale(X)
print(normalized)


# In[6]:




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                    random_state=3)

# print the size of the traning set:
print(X_train.shape)
print(y_train.shape)

# print the size of the testing set:
print(X_test.shape)
print(y_test.shape)

# "my_logreg" is instantiated as an "object" of LogisticRegression "class". 
my_logreg = LogisticRegression()

# Training ONLY on the training set:
my_logreg.fit(X, y)

# Testing on the testing set:
predict_lr = my_logreg.predict(X_test)


# In[7]:


from sklearn.metrics import accuracy_score
score_lr = accuracy_score(y_test, predict_lr)

print("Prediction Accuracy : " + str(score_lr))


# In[8]:


#Use Logistic Regression Classifier to predict the probability of Heart Disease based on the training/testing datasets that you built in part (c) (you have to use “my_logreg.predict_proba” method rather than “my_logreg.predict”). 
predict_lr = my_logreg.predict_proba(X_test)

fpr, tpr, thresholds = metrics.roc_curve(y_test, predict_lr[:,1], pos_label='Yes')

print(fpr)
print(tpr)
print("\n")
# AUC:
AUC = metrics.auc(fpr, tpr)
print("AUC : " + str(AUC))


# In[9]:


# Importing the "pyplot" package of "matplotlib" library of python to generate 
# graphs and plot curves:
import matplotlib.pyplot as plt

# The following line will tell Jupyter Notebook to keep the figures inside the explorer page 
# rather than openng a new figure window:
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure()

# Roc Curve:
plt.plot(fpr, tpr, color='red', lw=2, 
         label='ROC Curve (area = %0.2f)' % AUC)

# Random Guess line:
plt.plot([0, 1], [0, 1], color='blue', lw=1, linestyle='--')

# Defining The Range of X-Axis and Y-Axis:
plt.xlim([-0.005, 1.005])
plt.ylim([0.0, 1.01])

# Labels, Title, Legend:
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

plt.show()

