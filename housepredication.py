#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv(r"C:\Users\LENOVO\Downloads\house price data.csv")


# In[3]:


df


# In[4]:


df.head()


# In[5]:


df.isnull()


# In[6]:


df.isnull().sum()


# In[7]:


df.dropna()


# In[8]:


print(df)


# In[9]:


df.drop_duplicates(inplace=True)


# In[10]:


print(df)


# In[11]:


df.drop_duplicates(inplace=True)


# In[12]:


print(df)


# In[13]:


import matplotlib.pyplot as plt
plt.scatter(df['price'], df['bedrooms'])
plt.xlabel('Price')
plt.ylabel('Number of Bedrooms')
plt.title('Scatter Plot of Price vs Bedrooms')

# Display the plot
plt.show()


# In[14]:


plt.scatter(df['price'], df['sqft_living'])


# In[15]:


plt.scatter(df['price'], df['sqft_lot'])


# In[16]:


plt.scatter(df['price'], df['floors'])


# In[17]:


plt.scatter(df['price'], df['condition'])


# In[18]:


plt.scatter(df['price'], df['sqft_above'])


# In[19]:


plt.scatter(df['price'], df['sqft_basement'])


# In[20]:


plt.scatter(df['price'], df['yr_built'])


# In[21]:


plt.scatter(df['price'], df['yr_renovated'])


# In[22]:


plt.scatter(df['price'], df['street'])


# In[23]:


plt.scatter(df['price'], df['city'])


# In[24]:


plt.scatter(df['price'], df['statezip'])


# In[25]:


plt.scatter(df['price'], df['country'])


# In[26]:


df.dropna(how='all', axis=1, inplace=True)


# In[27]:


df.isnull().sum()


# In[28]:


X = df[['bathrooms', 'sqft_living','sqft_lot','condition','sqft_above','yr_built','street','statezip']]
y = df[['price']]


# In[29]:


X


# In[30]:


y


# In[31]:


pip install scikit-learn


# In[32]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[33]:


len(X)


# In[34]:


len(X_train)


# In[35]:


len(X_test)


# In[36]:


X_train


# In[37]:


X_test


# In[38]:


y_test


# In[39]:


y_train


# In[40]:


X = X.apply(pd.to_numeric, errors='coerce')
y = y.apply(pd.to_numeric, errors='coerce')


# In[41]:


X.fillna(0, inplace=True)
y.fillna(0, inplace=True)


# In[42]:


from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = LinearRegression().fit(X_train, y_train)


# In[43]:


X_test


# In[44]:


clf.predict(X_test)


# In[45]:


y_test


# In[46]:


y_test


# In[47]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100)
X_test


# In[ ]:




