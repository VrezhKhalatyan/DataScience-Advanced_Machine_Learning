
# coding: utf-8

# In[22]:


#Vrezh Khalatyan Homework 5
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import imageio
from IPython.display import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[23]:


# reading a CSV file directly from Web (or local drive), and store it in a pandas DataFrame:
# "read_csv" is a pandas function to read csv files from web or local drive:

label_df = pd.read_csv('/Users/anitribunyan/Desktop/HW5/label.csv')
label_df.head()


# In[24]:


numberOfImages = 1797 # number of images
dataMatrix = np.zeros((numberOfImages, 64))

for index in range(numberOfImages):
    dataPath = "/Users/anitribunyan/Desktop/HW5/Digit/{}.jpg".format(index)
    image = mpimg.imread(dataPath)
    feature = image.reshape(64)
    dataMatrix[index] = np.copy(feature)


# In[25]:


y = label_df['digit label']
X = pd.DataFrame.from_records(dataMatrix)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.21, random_state = 2)
my_RandomForest = RandomForestClassifier(n_estimators = 19, bootstrap = True, random_state = 3)
my_RandomForest.fit(X_train, y_train)
y_predict = my_RandomForest.predict(X_test)


# In[26]:


accuracy_r = accuracy_score(y_test, y_predict)
print("Random_Forest Classifier Accuracy Score: " + str(accuracy_r))

