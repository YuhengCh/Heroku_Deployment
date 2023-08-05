#!/usr/bin/env python
# coding: utf-8

# In[18]:


import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle


# In[7]:


load_data = load_iris()
X,y = load_data['data'], load_data['target']


# In[9]:


## split data into train and test datset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[11]:


clf = RandomForestClassifier(random_state=68)
clf.fit(X_train, y_train)


# In[14]:


## make predictions
y_preds = clf.predict(X_test)
# Calculate the accuracy
accuracy_score(y_test, y_preds)


# In[17]:


print(classification_report(y_test, y_preds))


# In[19]:


# Save the model to local using pickle
with open('model_iris.pkl', 'wb') as f:
    pickle.dump(clf, f)


# In[22]:


y_preds


# In[25]:


model = pickle.load(open('model_iris.pkl', 'rb'))


# In[26]:


load_data


# In[ ]:




