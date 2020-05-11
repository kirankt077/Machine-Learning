
# coding: utf-8

# In[15]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


dataset = pd.read_csv("housing.csv")


# In[17]:


modifieddataset =dataset.fillna(" ")
modifieddataset.isnull().sum()
dataset = modifieddataset


# In[18]:


dataset.columns
iv=dataset[['longitude','median_income','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','ocean_proximity','median_house_value']]
dv=dataset[['median_house_value']]


# In[19]:


dataset.columns


# In[20]:


iv=pd.get_dummies(iv,drop_first=True)


# In[21]:


from sklearn.model_selection import train_test_split
iv_train,iv_test,dv_train,dv_test = train_test_split(iv,dv,test_size =0.2,random_state =0)


# ## Linear Regression Model
# from sklearn.linear_model import LinearRegression
# lin_regressor = LinearRegression()
# lin_regressor.fit(iv_train.reshape(-1,1 ),dv_train.reshape(-1,1))

# In[22]:


## Linear Regression Codes Start from Here 
from sklearn.linear_model import LinearRegression
lin_regressor= LinearRegression()
lin_regressor.fit(iv_train,dv_train)
lin_regressor


# In[23]:


result_train = pd.DataFrame()
result_train['Actual Value']=result_train.append(dv_train[['median_house_value']],ignore_index=True)
result_train['Linear Predictions']= lin_regressor.predict(iv_train)

result_test = pd.DataFrame()
result_test['Actual value']=result_test.append(dv_test[['median_house_value']],ignore_index=True)
result_test['Linear Predictions']= lin_regressor.predict(iv_test)

