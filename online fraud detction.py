#!/usr/bin/env python
# coding: utf-8

# # LOADING THE DATA SET

# In[28]:


import pandas as pd
import numpy as np
data=pd.read_csv(r"C:\Users\raghu\OneDrive\Desktop\credit card.csv")
data


# In[6]:


print(data.isnull().sum())


# In[7]:


print(data.type.value_counts())


# In[ ]:





#   # DATA VISUALIZATION

# In[18]:


import plotly.express as px
t=data.type.value_counts()
names=t.index
values=t.values
fig=px.pie(data,names=names,values=values,hole=0.7,title="distrbution in the given data set")
fig.show()


# In[29]:


data['type']=data['type'].map({"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4,"DEBIT": 5})
data.isFraud=data.isFraud.map({0:'NO FRAUD',1:'FRAUD'})
data.head()


# # SPLITING THE DATA

# In[34]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
x=np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y=np.array(data.isFraud)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
print("accuracy:")
print(model.score(x_test,y_test))


# In[35]:


predict=np.array([[1,2000,30000,400000]])
print(model.predict(predict))


# In[ ]:





# In[ ]:




