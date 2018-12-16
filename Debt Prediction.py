
# coding: utf-8

# In[1]:


#Vrezh Khalatyan HW3 Question 2
# Importing the required packages and libraries
# we will need numpy and pandas later
import numpy as np
import pandas as pd


# In[2]:


#Read Credit csv and assign to Pandas DataFrame
credit_df = pd.read_csv('/Users/anitribunyan/Desktop/HW3/Credit.csv')


# In[3]:


# checking the dataset by printing every 10 lines:
credit_df[0::10]


# In[4]:


from sklearn import preprocessing
# Creating the Feature Matrix:

# create a python list of feature names that would like to pick from the dataset:
feature_cols = ['Income', 'Limit', 'Rating', 'Cards', 'Age', 'Education', 'Married']

# use the above list to select the features:
X = credit_df[feature_cols]

# Another way to do this (notice double bracket!):
X = credit_df[['Income', 'Limit', 'Rating', 'Cards', 'Age', 'Education', 'Married']]
X_Norm = preprocessing.scale(X)

# check the size:
print(X_Norm.shape)

# show the first 5 rows
X_Norm[0::10]


# In[5]:


from sklearn.model_selection import train_test_split

# select the target (last column) from the DataFrame
y = credit_df['Balance']

#Split the dataset into testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X_Norm, y, test_size=0.24, 
                                                    random_state=4)


# In[6]:


from sklearn.linear_model import LinearRegression

linreg = LinearRegression()

# fitting the model to the training data:
linreg.fit(X_train, y_train)


# In[15]:


# printing Theta0 using attribute "intercept_":
print(linreg.intercept_)

# printing [Theta1, Theta2, Theta3, Theta4, Theta5, Theta6, Theta7] using attribute "coef_":
print(linreg.coef_)

print('\n')
print('The most important coefficient would be the Rating feature with 478.53169403' + '\n' + 'The least important would be the first feature Income having -264.98372644 ')


# In[9]:


# make predictions on the testing set
y_prediction = linreg.predict(X_test)

print(y_prediction)


# In[10]:


from sklearn import metrics

# Calculating "Mean Square Error" (MSE):
mse = metrics.mean_squared_error(y_test, y_prediction)

# Using numpy sqrt function to take the square root and calculate "Root Mean Square Error" (RMSE)
rmse = np.sqrt(mse)

print(rmse)


# In[11]:


from sklearn.model_selection import cross_val_score


mse_list = cross_val_score(linreg, X, y, cv=10, scoring='neg_mean_squared_error')

print(mse_list)


# In[12]:


# in order to calculate root mean square error (rmse), we have to make them positive!
mse_list_positive = -mse_list

# using numpy sqrt function to calculate rmse:
rmse_list = np.sqrt(mse_list_positive)
print(rmse_list)


# In[13]:


# calculate the average RMSE as final result of cross validation:
print(rmse_list.mean())


# In[14]:


print('Without the cross_validation, the performance of the regression was 161.51385491175333 and with the' + '\n' + '10 fold cross validation the performance was 160.33198910744073')

