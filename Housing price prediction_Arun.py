
# coding: utf-8

# In[70]:


## Import Required lib's
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[71]:


## Importing dataset
mydata = pd.read_csv("housing.csv")


# In[72]:


mydata.head()


# In[73]:


mydata.info()


# In[74]:


#Describe data
mydata.describe()


# In[75]:


#Get number of Null Values
def get_null_count(mydata):
    for i in mydata.columns:
        print(i,': ',len(mydata[mydata[i].isnull()][i]))
get_null_count(mydata)


# In[79]:


#Hist plot
mydata.hist(column='total_bedrooms',bins=40)


# In[80]:


#Imputation
#From hist plot we could say data is right sweked, so it's better to apply Median to replace missing values instead of Mean
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
mydata[['total_bedrooms']]=imputer.fit_transform(mydata[['total_bedrooms']])


# In[81]:


#Now replaced all Null values with mean and we could see from function we don't have any Null values
def get_null_count(mydata):
    for i in mydata.columns:
        print(i,': ',len(mydata[mydata[i].isnull()][i]))
get_null_count(mydata)        


# In[82]:


#Diving data set to indipendent and depenent variables
iv = mydata[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity']]
dv = mydata['median_house_value']


# In[83]:


#Label encoding and One Hot encoding for categorical variables
iv = pd.get_dummies(iv)


# In[84]:


#Dividing the Dataset into Test and Train
from sklearn.model_selection import train_test_split
iv_train,iv_test,dv_train,dv_test=train_test_split(iv,dv,test_size=0.2,random_state=0)


# In[85]:


#Applying LinearRegression, In Linear Regression scalling is done by Algorithum 
from sklearn.linear_model import LinearRegression
lin_regressor= LinearRegression()
lin_regressor.fit(iv_train,dv_train)


# In[86]:


#Linear prediction
linear_pred = lin_regressor.predict(iv_test)
actual = dv_test
predicted = linear_pred


# In[87]:


## To find RMSE for Linear regression
linear_RMSE = round(np.sqrt(np.mean(actual - predicted)**2),2)


# In[88]:


## Applying scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
iv_train = sc.fit_transform(iv_train)
iv_test = sc.transform(iv_test)


# In[89]:


## Applying Decission Tree
from sklearn.tree import DecisionTreeRegressor
dt_regressor = DecisionTreeRegressor(max_depth=3)
dt_regressor.fit(iv_train,dv_train)
dt_predt = dt_regressor.predict(iv_test)


# In[90]:


actual_dt = dv_test
predicted_dt = dt_predt


# In[91]:


## To find RMSE for decision Tree 
dt_RMSE = round(np.sqrt(np.mean(actual_dt - predicted_dt)**2),2)


# In[92]:


## Applying RandomForest
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=1)
rf_classifier.fit(iv_train,dv_train)


# In[93]:


rf_predict = rf_classifier.predict(iv_test)
actual_rf = dv_test
predicted_rf = rf_predict


# In[94]:


## To find RMSE for Random Forest
rf_RMSE = round(np.sqrt(np.mean(actual_rf - predicted_rf)**2),2)


# In[102]:


## RMSE for Liner Regression, Decission Tree and Random Forrest as follows
print("RMSE for Linear Regression:{}".format(linear_RMSE))
print("RMSE for Decission Tree:{}".format(dt_RMSE))
print("RMSE for Random Forrest:{}".format(rf_RMSE))


# In[103]:


mydata.head()


# In[104]:


#Bonus exercise:
iv = mydata.iloc[:,7]
dv = mydata.iloc[:,-1]


# In[105]:


from sklearn.model_selection import train_test_split
iv_train_med,iv_test_med,dv_train_med,dv_test_med=train_test_split(iv,dv,test_size=0.2,random_state=0)
iv_train_med=np.array(iv_train_med).reshape(len(iv_train_med),1)
dv_train_med=np.array(dv_train_med).reshape(len(dv_train_med),1)
iv_test_med = np.array(iv_test_med).reshape(len(iv_test_med),1)
dv_test_med = np.array(dv_test_med).reshape(len(dv_test_med),1) 


# In[106]:


# predicting housing values using Linear regression from median_income
from sklearn.linear_model import LinearRegression
med_linear = LinearRegression()
med_linear.fit(iv_train_med,dv_train_med)


# In[107]:


y_pred_med = med_linear.predict(iv_test_med)


# In[108]:


#Plotting Test data and Predicted data
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(20,10))
plt.scatter(iv_test_med, y_pred_med, color='orange')
plt.scatter(iv_test_med, dv_test_med, color = 'blue' )
plt.xlabel('median_income')
plt.ylabel('median_house_value')
plt.title('Plot aginst median_income and house values')
plt.grid()
plt.show()

