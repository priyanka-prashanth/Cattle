#!/usr/bin/env python
# coding: utf-8

# In[88]:


#Importing necessary libraries

import numpy as np
import matplotlib.pyplot as plt 

import pandas as pd  
import seaborn as sns 
import sklearn 


# In[89]:


df = pd.read_csv("C:\\Users\\priya\\Desktop\\Data (2).csv"


# In[90]:


#Displaying columns whose datatype is 'object'

obj_df = df.select_dtypes(include=['object']).copy()
obj_df.head()


# In[91]:


obj_df[obj_df.isnull().any(axis=1)]


# In[92]:


obj_df = obj_df.fillna({"Activity": "WALKING"})


# # Activity Prediction 

# In[93]:


#Label encoding the Activity column
#Fit Transform assigns an integer value to every unique non integer value in the activity column

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df['Activity'] = le.fit_transform(df['Activity'])
df['Activity'].unique()


# In[94]:


df.head()


# In[95]:


#Separating feature column(Input column) and label column(Output column)

feature_column = ['X','Y','Z']
x = df[feature_column]
y = obj_df.Activity


# In[96]:


#Dividing dataset into training (70%) and testing set (30%)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=5)


# In[98]:


#Importing KNN algorithm 
#Training the KNN model using knn.fit

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(x_train,y_train)


# In[99]:


#Model predicts the output using the testing set

y_predict = knn.predict(x_test)


# # Confusion Matrix for Predicting Activity

# In[100]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)
cm


# In[101]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_predict))


# In[102]:


#Taking user input for x,y,z coordinates to predict the Activity

input_x = float(input('enter x coordinate: '))
input_y = float(input('enter y coordinate: '))
input_z = float(input('enter z coordinate: '))
out = [[input_x,input_y,input_z]]
final_out = knn.predict(out)
print(final_out)


# # Health Prediction

# In[107]:


df.drop(columns = 'Health',inplace = True)


# In[108]:


#Inserting the value of Health based on Temperature and heartbeat

H = pd.Series([])

for i in range(len(df)):
    if df["Temperature"][i] >= 100 and df["Temperature"][i] <= 102 and df['Heartbeat'][i] >= 48 and df['Heartbeat'][i] <= 84:
        H[i] = "Healthy"
    else:
        H[i] = "Unhealthy"
        
df.insert(6,"Health",H)
df.head()


# In[109]:


df.head()


# In[110]:


#Separating feature column(Input column) and label column(Output column)
#Dividing dataset into training (70%) and testing set (30%)

feature_columns = ['Temperature','Heartbeat','Activity']
x = df[feature_columns]
y = df['Health']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=5)


# In[111]:


#Importing KNN algorithm 
#Training the KNN model using knn.fit

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(x_train, y_train)


# In[112]:


#Taking user input to predict the Health

input_x = float(input('enter Temperature: '))
input_y = float(input('enter Heartbeat: '))
input_z = float(input('enter Activity: '))
out = [[input_x,input_y,input_z]]
final_out = knn.predict(out)
print(final_out)


# In[113]:


#Model predicts the output using the testing set

y_predict = knn.predict(x_test)


# # Confusion Matrix for Predicting Health

# In[114]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)
cm


# In[115]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_predict))

