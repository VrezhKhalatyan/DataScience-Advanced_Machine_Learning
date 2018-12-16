
# coding: utf-8

# In[4]:


#Question2a
#a- Read the iris dataset from the following URL:https://raw.githubusercontent.com/mpourhoma/CS4661/master/iris.csvand assign it to a Pandas DataFrame as you learned in tutorial Lab2-3.
# Import the pandas library
import pandas as pd
# creates a empty dataframe
df = pd.DataFrame()
# reading a CSV file directly from Web, and store it in a pandas DataFrame:
# "read_csv" is a pandas function to read csv files from web or local device:
df = pd.read_csv('https://raw.githubusercontent.com/mpourhoma/CS4661/master/iris.csv')
df


# In[5]:


#Question2b
# Split the dataset into testing and training sets with the following parameters:test_size=0.4, random_state=6
# Importing the required packages and libraries
import numpy as np
# Randomly splitting the original dataset into training set and testing set
# The function"train_test_split" from "sklearn.cross_validation" library performs random splitting.
# "test_size=0.4" means that pick 40% of data samples for testing set, and the rest (60%) for training set.
from sklearn.model_selection import train_test_split

# create a python list of feature names that would like to pick from the dataset:
feature_cols = ['sepal_length','sepal_width','petal_length','petal_width']
# use the above list to select the features from the original DataFrame
X = df[feature_cols]
# select a Series of labels (the last column) from the DataFrame
# y = idf['label'] # this is the index that we gave to the labels
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=6)
# printing the size of the training set
print(X_train.shape)
print(y_train.shape)
# printing the size of the testing set
print(X_test.shape)
print(y_test.shape)
# printing the actual testing set
print(X_test)
print('\n')
print(y_test)


# In[6]:


#Question2c
# Instantiate a KNN object with K=3, train it on the training set and test it on the testing set.Then, calculate the accuracy of your prediction as you learned in Lab3.
# The following line will import KNeighborsClassifier "Class"
# KNeighborsClassifier is name of a "sklearn class" to perform "KNN Classification"
# Importing the required packages and libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# Instantiating another "object" of KNeighborsClassifier "class" with k=3:
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
# We use the method "fit" of the object along with training dataset and labels to train the model.
knn.fit(X_train, y_train)
# We use the method "predict" of the *trained* object knn on one or more testing data sample to perform prediction:
y_predict = knn.predict(X_test)
print(y_predict)
# Checking for the results
results = pd.DataFrame()

results['actual'] = y_test 
results['prediction'] = y_predict 
# Printing the results
print(results)


# In[7]:


#Question2d
#Repeat part (c) for K=1, K=5, K=7, K=11, K=15, K=27, K=59 (you can simply use a “for loop,”and save the final accuracy results in a list). Does the accuracy always get better byincreasing the value K? 
k = 1
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

y_predict = knn.predict(X_test)
print(y_predict)

results = pd.DataFrame()

results['actual'] = y_test 
results['prediction'] = y_predict 

print(results)
print('End of prediction with k = 1')
print('------------------------------------------------------------------')

k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

y_predict = knn.predict(X_test)
print(y_predict)

results = pd.DataFrame()

results['actual'] = y_test 
results['prediction'] = y_predict 

print(results)
print('End of prediction with k = 5')
print('------------------------------------------------------------------')

k = 7
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

y_predict = knn.predict(X_test)
print(y_predict)

results = pd.DataFrame()

results['actual'] = y_test 
results['prediction'] = y_predict 

print(results)
print('End of prediction with k = 7')
print('------------------------------------------------------------------')

k = 11
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

y_predict = knn.predict(X_test)
print(y_predict)

results = pd.DataFrame()

results['actual'] = y_test 
results['prediction'] = y_predict 

print(results)
print('End of prediction with k = 11')
print('------------------------------------------------------------------')

k = 15
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

y_predict = knn.predict(X_test)
print(y_predict)

results = pd.DataFrame()

results['actual'] = y_test 
results['prediction'] = y_predict 

print(results)
print('End of prediction with k = 15')
print('------------------------------------------------------------------')

k = 27
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

y_predict = knn.predict(X_test)
print(y_predict)

results = pd.DataFrame()

results['actual'] = y_test 
results['prediction'] = y_predict 

print(results)
print('End of prediction with k = 27')
print('------------------------------------------------------------------')

k = 59
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

y_predict = knn.predict(X_test)
print(y_predict)

results = pd.DataFrame()

results['actual'] = y_test 
results['prediction'] = y_predict 

print(results)
print('End of prediction with k = 59')
print('------------------------------------------------------------------')

print('The accuracy does not get better as k increases and also not good when k is too small. ' + '\n' + 'At k=1, there where 3 mistakes; at k=5, there was 1 mistake; ' + '\n' + 'at k=7, there was 2 mistakes; at k=11 there where 2 mistakes; ' + '\n' + 'at k=15, there where 4 mistakes; at k=27, there where 5 mistakes;' + '\n' + ' and at k=59, there where 11 mistakes.')




# In[8]:


#Question2e
#Prediction on Sepal Length
# train, test, and evaluate your model 4 times,each time on a dataset including only one of the features, and save the final accuracyresults in a list).
feature_cols = ['sepal_length']
X = df[feature_cols]
#print(X)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=6)
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

y_predict = knn.predict(X_test)
print(y_predict)

results = pd.DataFrame()

results['actual'] = y_test 
results['prediction'] = y_predict 

accuracySepalLength = accuracy_score(y_test, y_predict)
#print(accuracySepalLength)

print(results)
print('------------------------------------------------------------------')

#Prediction on Sepal Width
feature_cols = ['sepal_width']
X = df[feature_cols]
#print(X)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=6)
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

y_predict = knn.predict(X_test)
print(y_predict)

results = pd.DataFrame()

results['actual'] = y_test 
results['prediction'] = y_predict 

accuracySepalWidth = accuracy_score(y_test, y_predict)
print(results)
print('------------------------------------------------------------------')

#Prediction on Petal Length
feature_cols = ['petal_length']
X = df[feature_cols]
#print(X)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=6)
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

y_predict = knn.predict(X_test)
print(y_predict)

results = pd.DataFrame()

results['actual'] = y_test 
results['prediction'] = y_predict 

accuracyPetalLength = accuracy_score(y_test, y_predict)
print(results)
print('------------------------------------------------------------------')

#Prediction on Petal Width
feature_cols = ['petal_width']
X = df[feature_cols]
#print(X)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=6)
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

y_predict = knn.predict(X_test)
print(y_predict)

results = pd.DataFrame()

results['actual'] = y_test 
results['prediction'] = y_predict 

accuracyPetalWidth = accuracy_score(y_test, y_predict)
print(results)
print('------------------------------------------------------------------')

print('Final Results')
print('--------------')
listOfResults = {'Sepal Length':accuracySepalLength,'Sepal Width':accuracySepalWidth, 'Petal Length': accuracyPetalLength, 'Petal Width':accuracyPetalWidth}
print(listOfResults)

print('The best feature is Petal Width and the second best feature is Petal Length')


# In[9]:


#Question2f
#Now, we want to repeat part (e), this time using two features. you have to train, test, andevaluate your model for 6 different cases: using (1st and 2nd features), (1st and 3rdfeatures), (1st and 4th features), (2nd and 3rd features), (2nd and 4th features), (3rd and 4thfeatures)! 
feature_cols = ['sepal_length', 'sepal_width']
X = df[feature_cols]
#print(X)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=6)
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

y_predict = knn.predict(X_test)
#print(y_predict)

results = pd.DataFrame()

results['actual'] = y_test 
results['prediction'] = y_predict 

accuracy1 = accuracy_score(y_test, y_predict)
#print(accuracy1)
#-------------------------------------------------------------------------------------------
feature_cols = ['sepal_length', 'petal_length']
X = df[feature_cols]
#print(X)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=6)
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

y_predict = knn.predict(X_test)
#print(y_predict)

results = pd.DataFrame()

results['actual'] = y_test 
results['prediction'] = y_predict 

accuracy2 = accuracy_score(y_test, y_predict)
#print(accuracy2)
#-------------------------------------------------------------------------------------------

feature_cols = ['sepal_length', 'petal_width']
X = df[feature_cols]
#print(X)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=6)
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

y_predict = knn.predict(X_test)
#print(y_predict)

results = pd.DataFrame()

results['actual'] = y_test 
results['prediction'] = y_predict 

accuracy3 = accuracy_score(y_test, y_predict)
#print(accuracy3)
#-------------------------------------------------------------------------------------------

feature_cols = ['sepal_width', 'petal_length']
X = df[feature_cols]
#print(X)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=6)
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

y_predict = knn.predict(X_test)
#print(y_predict)

results = pd.DataFrame()

results['actual'] = y_test 
results['prediction'] = y_predict 

accuracy4 = accuracy_score(y_test, y_predict)
#print(accuracy4)
#-------------------------------------------------------------------------------------------

feature_cols = ['sepal_width', 'petal_width']
X = df[feature_cols]
#print(X)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=6)
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

y_predict = knn.predict(X_test)
#print(y_predict)

results = pd.DataFrame()

results['actual'] = y_test 
results['prediction'] = y_predict 

accuracy5 = accuracy_score(y_test, y_predict)
#print(accuracy5)
#-------------------------------------------------------------------------------------------

feature_cols = ['petal_length', 'petal_width']
X = df[feature_cols]
#print(X)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=6)
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

y_predict = knn.predict(X_test)
#print(y_predict)

results = pd.DataFrame()

results['actual'] = y_test 
results['prediction'] = y_predict 

accuracy6 = accuracy_score(y_test, y_predict)
#print(accuracy6)
#-------------------------------------------------------------------------------------------

listOfResults2 = {'Sepal Length/ Sepal Width':accuracy1,'Sepal Length/ Petal Length':accuracy2, 'Sepal Length/ Petal Width': accuracy3, 'Sepal Width/ Petal Length':accuracy4, 'Sepal Width/ Petal Width':accuracy5, 'Petal Length/ Petal Width':accuracy6}
print(listOfResults2)

print('The best feature pair is Sepal Length & Petal Length with accuracy 0.983')


# In[10]:


#Question2g
#BigQuestion: Doesthe “best feature pair” from part (f) contain of both “first best feature”and “second best feature” from part (e)? In other word, can we conclude that the “besttwo features” for classification are the first best feature along with the second best featuretogether?
print('False Claim! The best feature pair was Sepal Length & Petal Length from question f; however,'+ '\n' + 'the best feature from question e was Petal Width and the second being Petal Length.'+ '\n' + 'In conclusion, we cannot say that the best two features for classification are the first best feature along with the second best feature together; eventough; Petal Length was in both best features.')


# In[43]:


#Question2h
#Optional Question: Justify your answer for part (g)! If yes, why? If no, why not?
print('For the pair accuracy test, this could be due to the length of both Petal and Sepal. Accuracy may work better due to the same components being length and length.' + '\n' + 'It seems that Petal Length was in both best accuracy results, which contradict this claim')

