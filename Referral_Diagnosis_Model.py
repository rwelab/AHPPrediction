#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[44]:


#filepath
train_df=pd.read_csv('file path')
pred_df=pd.read_csv('file path')


# In[47]:


# Finding Common columns
common = pred_df.columns.intersection(train_df.columns)
train_df_filter=train_df.filter(common, axis=1)
diff=pred_df.columns.difference(diag_df.columns)
diag_df_filter=pred_df.filter(diff, axis=1) 
diagnosis_data=pd.concat[train_df_filter,diag_df_filter], ignore_index=True)


# In[49]:


diagnosis_data=diagnosis_data[diagnosis_data['Label'].notnull()]
diagnosis_data=diagnosis_data.fillna(0.001)
del diagnosis_data['Label']


# In[1]:


#Prediction
import pandas as pd
import numpy as np
import joblib


# In[2]:


# Load from file

model1 = joblib.load('Model2_Predict_AB.sav')


# In[57]:


# index slicing to retrieve a 1d-array of probabilities
y_pred = model1.predict_proba(diagnosis_data)[:, 1]
y_pred


# In[58]:


# top 5000 prediction indexes ( argsort sorts in ascending order):
list=np.argsort(y_pred)[-100:]


# In[59]:


newlist=y_pred[list].tolist()
listdf = pd.DataFrame()
listdf['Probability']=newlist


# In[60]:


listdf


# In[61]:


prediction=pred_df.loc[list]
prediction['PatientDurableKey']


# In[65]:


prediction.to_csv('100_DF_AB_UCSF.csv')


# In[66]:


listdf.to_csv('100_probability_AB_UCSF.csv')


# In[67]:





