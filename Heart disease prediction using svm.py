#!/usr/bin/env python
# coding: utf-8

# In[249]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# In[113]:


df = pd.read_csv("C:\\Users\\SUVARNA\\Downloads\\heart.csv")
df.head()


# In[114]:


df.shape


# In[115]:


df.info()


# In[116]:


df.describe()


# In[117]:


df['target'].value_counts()


# In[118]:


df.columns


# In[258]:


df.corr()


# In[257]:


plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),annot=True,cmap=plt.cm.Spectral)


# In[259]:


plt.figure(figsize=(10,5))
sns.countplot(x='target',data=df)
plt.grid()


# In[119]:


X = df.loc[:,df.columns!='target']
X.head()


# In[260]:


y = df.target


# In[121]:


X_en = pd.get_dummies(df,columns=['sex','cp','fbs','restecg','exang','slope','ca','thal'])
X_en.head()


# In[135]:


#splitting the dataset
X_train,X_test,y_train,y_test = train_test_split(X_en,y,test_size=0.5,random_state=0)


# In[136]:


# Standardizing

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


# In[250]:


#Principle component Analysis
from sklearn.decomposition import PCA

pca = PCA(n_components = 25)

X_tr = pca.fit_transform(X_train_std)
X_te = pca.transform(X_test_std)


# In[251]:


explained_varance = pca.explained_variance_ratio_
explained_varance


# In[252]:


svc = SVC(kernel="linear",random_state=0, C=1.0, gamma="auto")
svc.fit(X_tr,y_train)


# In[253]:


y_pred = svc.predict(X_te)


# In[263]:


df2 = pd.DataFrame({"Expected":y_test,"Predicted":y1})
df2


# In[264]:


accuracy_score(y_test,y_pred)


# In[265]:


confusion_matrix(y_test,y_pred)

