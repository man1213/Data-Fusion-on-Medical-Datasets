#!/usr/bin/env python
# coding: utf-8

# ## INTRODUCTION
# <br>
# We have a dataset which classifies if patients have chronic kidney disease or not according to the attributes. We will try to use this data to create a model which tries to predict if a patient has this disease or not. We will use logistic regression for our analysis. In order to observe which features contribute the most, we will use the feature importance algorithm which uses an eXtreme Gradient Boost Classifier.

# In[1]:


import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from IPython.core.pylabtools import figsize

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read Data

# In[2]:


df = pd.read_csv('kidney_disease.csv')


# In[3]:


df.head()


# Columns explain:
# <br>
# age - age <br>
# bp - blood pressure <br>
# sg - specific gravity <br>
# al - albumin <br>
# su - sugar <br>
# rbc - red blood cells <br>
# pc - pus cell <br>
# pcc - pus cell clumps <br>
# ba - bacteria <br>
# bgr - blood glucose random <br>
# bu - blood urea <br>
# sc - serum creatinine <br>
# sod - sodium <br>
# pot - potassium <br>
# hemo - hemoglobin <br>
# pcv - packed cell volume <br>
# wc - white blood cell count <br>
# rc - red blood cell count <br>
# htn - hypertension <br>
# dm - diabetes mellitus <br>
# cad - coronary artery disease <br>
# appet - appetite <br>
# pe - pedal edema <br>
# ane - anemia <br>
# classification - classification

# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


for i in ['rc','wc','pcv']:
    df[i] = df[i].str.extract('(\d+)').astype(float)
for i in ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','rc','wc','pcv']:
    df[i].fillna(df[i].mean(),inplace=True)


# In[7]:


sns.countplot(data=df,x='rbc')
df['rbc'].fillna('normal',inplace=True)


# In[8]:


sns.countplot(data=df,x='pc')
df['pc'].fillna('normal',inplace=True)


# In[9]:


df['cad'] = df['cad'].replace(to_replace='ckd\t',value='ckd')


# In[10]:


df.info()


# ## Data Wrangling

# In[11]:


df = df.replace('?', np.nan)
df = df.dropna(axis=0, how="any")


# In[12]:


rbc_1 = []
rbc_2 = []

for i in df['rbc']:
    if i == 'normal':
        rbc_1.append(1)
        rbc_2.append(0)
    else:
        rbc_1.append(0)
        rbc_2.append(1)


# In[13]:


pc_1 = []
pc_2 = []

for i in df['pc']:
    if i == 'normal':
        pc_1.append(1)
        pc_2.append(0)
    else:
        pc_1.append(0)
        pc_2.append(1)
        


# In[14]:


pcc_1 = []
pcc_2 = []

for i in df['pcc']:
    if i == 'present':
        pcc_1.append(1)
        pcc_2.append(0)
    else:
        pcc_1.append(0)
        pcc_2.append(1)


# In[15]:


ba_1 = []
ba_2 = []

for i in df['ba']:
    if i == 'present':
        ba_1.append(1)
        ba_2.append(0)
    else:
        ba_1.append(0)
        ba_2.append(1)


# In[16]:


htn_1 = []
htn_2 = []

for i in df['htn']:
    if i == 'yes':
        htn_1.append(1)
        htn_2.append(0)
    else:
        htn_1.append(0)
        htn_2.append(1)


# In[17]:


dm_1 = []
dm_2 = []

for i in df['dm']:
    if i == 'yes':
        dm_1.append(1)
        dm_2.append(0)
    else:
        dm_1.append(0)
        dm_2.append(1)


# In[18]:


cad_1 = []
cad_2 = []

for i in df['cad']:
    if i == 'yes':
        cad_1.append(1)
        cad_2.append(0)
    else:
        cad_1.append(0)
        cad_2.append(1)


# In[19]:


appet_1 = []
appet_2 = []

for i in df['appet']:
    if i == 'good':
        appet_1.append(1)
        appet_2.append(0)
    else:
        appet_1.append(0)
        appet_2.append(1)


# In[20]:


pe_1 = []
pe_2 = []

for i in df['pe']:
    if i == 'yes':
        pe_1.append(1)
        pe_2.append(0)
    else:
        pe_1.append(0)
        pe_2.append(1)


# In[21]:


ane_1 = []
ane_2 = []

for i in df['ane']:
    if i == 'yes':
        ane_1.append(1)
        ane_2.append(0)
    else:
        ane_1.append(0)
        ane_2.append(1)
        


# In[22]:


df['rbc_1'] = rbc_1
df['rbc_2'] = rbc_2
df['pc_1'] = pc_1
df['pc_2'] = pc_2
df['pcc_1'] = pcc_1
df['pcc_2'] = pcc_2
df['ba_1'] = ba_1
df['ba_2'] = ba_2
df['htn_1'] = htn_1
df['htn_2'] = htn_2
df['dm_1'] = dm_1
df['dm_2'] = dm_2
df['cad_1'] = cad_1
df['cad_2'] = cad_2
df['appet_1'] = appet_1
df['appet_2'] = appet_2
df['pe_1'] = pe_1
df['pe_2'] = pe_2
df['ane_1'] = ane_1
df['ane_2'] = ane_2


# In[23]:


df1 = df


# ## Principal Component Analysis

# In[24]:


targets = df['classification'].astype('category')
df = df.drop(['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'classification'], axis=1)


# In[25]:


label_color = ['red' if i=='ckd' else 'green' for i in targets]

df = preprocessing.StandardScaler().fit_transform(df)
pca = PCA(n_components=2)
pca.fit(df)
T = pca.transform(df)
T = pd.DataFrame(T)
T.columns = ['PCA component 1', 'PCA component 2']
T.plot.scatter(x='PCA component 1', y='PCA component 2', marker='o',
        alpha=0.7, # opacity
        color=label_color,
        title="red: ckd, green: not-ckd" )
plt.show()


# ## Sequential Model for Chronic Kidney Disease

# In[26]:


from sklearn.preprocessing import LabelEncoder

for i in ['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane','classification']:
    df1[i] = LabelEncoder().fit_transform(df1[i])


# In[27]:


from sklearn.preprocessing import MinMaxScaler

for i in df1.columns:
    df1[i] = MinMaxScaler().fit_transform(df1[i].astype(float).values.reshape(-1, 1))


# In[28]:


X = df1.drop(['id','classification'],axis=1)
Y = df1['classification']


# In[29]:


X.shape


# In[30]:


Y.shape


# In[31]:


from keras.models import Sequential
from keras.layers import Dense


# In[32]:


model = Sequential()


# In[33]:


model.add(Dense(100,input_dim=X.shape[1],activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(25,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[34]:


history = model.fit(X,Y,epochs=50,batch_size=40,validation_split=.2,verbose=2)


# In[35]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[36]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[37]:


scores = model.evaluate(X,Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# ## Model accuracy of <b>99.49%%</b> is achieved.

# ## Feature Importance

# In[38]:


X1 = df1.drop(columns=['classification'])
y = df1['classification']


# In[39]:


X1 = pd.get_dummies(X1, prefix_sep='_', drop_first=True)
X1.head()


# In[40]:


xgb_cl = xgb.XGBClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=21, stratify=y)
xgb_cl.fit(X_train, y_train)


# In[41]:


figsize(10,8)
plt.style.use('fivethirtyeight')
xgb.plot_importance(xgb_cl)


# From the feature importance plot above, we can observe that hemoglobin is the most informative feature for the dataset along with specific gravity, blood urea, albumin etc.   

# In[ ]:




