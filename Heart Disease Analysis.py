#!/usr/bin/env python
# coding: utf-8

# # INTRODUCTION
# <br>
# We have a dataset which classifies if patients have heart disease or not according to the attributes. We will try to use this data to create a model which tries to predict if a patient has this disease or not. We will use logistic regression and random forest machine learning algorithms for our analysis. We will then use a correlation matrix to plot the degree of correlation between various features with the target value in the dataset. 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings("ignore")
sns.set_style("darkgrid")


# ## Read Data

# In[2]:


df = pd.read_csv("heart.csv")


# In[3]:


df.head()


# Data contains; <br>
# 
# * age - age in years <br>
# * sex - (1 = male; 0 = female) <br>
# * cp - chest pain type (0-3) <br>
# * trestbps - resting blood pressure (in mm Hg on admission to the hospital) <br>
# * chol - serum cholestoral in mg/dl <br>
# * fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false) <br>
# * restecg - resting electrocardiographic results <br>
# * thalach - maximum heart rate achieved <br>
# * exang - exercise induced angina (1 = yes; 0 = no) <br>
# * oldpeak - ST depression induced by exercise relative to rest <br>
# * slope - the slope of the peak exercise electrocardiogram ST segment/heart rate slope (0-2) <br>
# * ca - number of major vessels (0-3) colored by flourosopy <br>
# * thal -  A blood disorder called thalassemia (0-3) <br>
# * target - have disease or not (1=yes, 0=no)

# ## Data Exploration

# In[4]:


df.target.value_counts()


# In[5]:


sns.countplot(x="target", data=df, palette="bwr")
plt.show()


# In[6]:


countNoDisease = len(df[df.target == 0])
countHaveDisease = len(df[df.target == 1])
print("Percentage of Patients not having Heart Disease: {:.2f}%".format((countNoDisease / (len(df.target))*100)))
print("Percentage of Patients  Heart Disease: {:.2f}%".format((countHaveDisease / (len(df.target))*100)))


# In[7]:


sns.countplot(x='sex', data=df, palette="mako_r")
plt.xlabel("Sex (0 = female, 1= male)")
plt.show()


# In[8]:


fig,ax=plt.subplots(figsize=(24,6))
plt.subplot(1, 3, 1)
age_bins = [20,30,40,50,60,70,80]
df['bin_age']=pd.cut(df['age'], bins=age_bins)
g1=sns.countplot(x='bin_age',data=df ,hue='target',palette='plasma',linewidth=3)
g1.set_title("Age vs Heart Disease")
#The number of people with heart disease are more from the age 40-60

plt.subplot(1, 3, 2)
cho_bins = [100,150,200,250,300,350,400,450]
df['bin_chol']=pd.cut(df['chol'], bins=cho_bins)
g2=sns.countplot(x='bin_chol',data=df,hue='target',palette='plasma',linewidth=3)
g2.set_title("Cholestoral vs Heart Disease")
#Most people get the heart disease with 200-250 cholestrol 
#The others with cholestrol of above 250 tend to think they have heart disease but the rate of heart disease falls

plt.subplot(1, 3, 3)
thal_bins = [60,80,100,120,140,160,180,200,220]
df['bin_thal']=pd.cut(df['thalach'], bins=thal_bins)
g3=sns.countplot(x='bin_thal',data=df,hue='target',palette='plasma',linewidth=3)
g3.set_title("Thal vs Heart Disease")
#People who have their maximum heart rate between 140-180 have a very high chance of getting the heart disease 


# ### Creating Dummy Variables

# ![](http://)Since 'cp', 'thal' and 'slope' are categorical variables we'll turn them into dummy variables.

# In[9]:


a = pd.get_dummies(df['cp'], prefix = "cp")
b = pd.get_dummies(df['thal'], prefix = "thal")
c = pd.get_dummies(df['slope'], prefix = "slope")


# In[10]:


frames = [df, a, b, c]
df = pd.concat(frames, axis = 1)
df.head()


# In[11]:


df = df.drop(columns = ['cp', 'thal', 'slope'])
df.head()


# In[12]:


df = df.replace('?', np.nan)
df = df.dropna(axis=0, how="any")


# In[13]:


df.head(10)


# In[15]:


targets = df1['target'].astype('category')


# In[16]:


df1 = df1.drop(['target'], axis=1)


# In[17]:


df1.head(5)


# ## Creating Model for Logistic Regression
# <br>
# We can use sklearn library or we can write functions ourselves. We will use sklearn library to calculate score.

# ### Sklearn Logistic Regression

# In[20]:


y = df.target.values
x_data = df.drop(['target'], axis = 1)


# In[21]:


y.shape


# In[22]:


x_data.shape


# We will split our data. 80% of our data will be train data and 20% of it will be test data.

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x1,y,test_size = 0.2,random_state=0)


# In[ ]:


x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T


# In[ ]:


lr = LogisticRegression()
lr.fit(x_train.T,y_train.T)
print("Test Accuracy {:.2f}%".format(lr.score(x_test.T,y_test.T)*100))


# ## <font color = "purple">Our model works with <font color="red">**86.89%**</font> accuracy.</font>

# ## Random Forest Classification

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)
rf.fit(x_train.T, y_train.T)
print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(rf.score(x_test.T,y_test.T)*100))


# ## <font color="#0FBBAE">Test Accuracy of Random Forest: <font color="red">88.52%</font></font>

# ## Comparing Models

# In[28]:


methods = ["Logistic Regression", "Random Forest"]
accuracy = [86.89, 88.52]
colors = ["purple", "#0FBBAE"]

sns.set_style("whitegrid")
plt.figure(figsize=(8,5))
plt.yticks(np.arange(0,100,10))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=methods, y=accuracy, palette=colors)
plt.show()


# Our models work fine but best of them is Random Forest with 88.52% of accuracy.

# ## Correlation Matrix

# In[27]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True,cmap='YlGnBu',fmt='.2f',linewidths=2)


# So from this correlation matrix, we can observe which input features contribute most during the training process of various classifiers. <i>Blood disorder</i> , <i>heart rate slope</i>, <i>maximum heart rate</i> and <i>chest pain types</i> can be used for prediction more accuractely than other features.
# <br>
# It does not come as a surprise that the more complex algorithms like Random Forests generated better results compared to the basic ones. It is worth to emphasize that in most cases hyperparameter tuning is essential to achieve robust results out of these techniques. By producing decent results, simpler methods proved to be useful as well.
# <br>
# Machine learning has absolutely bright future in medical field. With just basic information about a certain patient's medical history, we may quite accurately predict whether a disease will occur or not.

# In[ ]:




