#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


cosmetics= pd.read_csv(r'C:\Users\DELL\Downloads\archive\cosmetics.csv')


# In[3]:


cosmetics.head()


# In[4]:


print ("taille de donnees est: ", cosmetics.shape)


# In[5]:


cosmetics.info


# In[6]:


cosmetics.describe()


# In[7]:


cosmetics.isnull().values.any()


# In[8]:


pip install numpy


# In[9]:


import numpy as np


# In[10]:


pip install matplotlib


# In[11]:


import matplotlib.pyplot as plt


# In[12]:


pip install seaborn


# In[13]:


import seaborn as sns


# In[14]:


plt.hist(cosmetics['Label'], bins=6)
plt.show()


# In[15]:


plt.hist(cosmetics['Brand'], bins=30)
plt.show()


# In[16]:


plt.hist(cosmetics['Price'], bins=5)
plt.show()


# In[17]:


plt.hist(cosmetics['Rank'], bins=5)
plt.show()


# In[18]:


cosmetics['Dry'].value_counts().plot.pie(autopct='%1.1f%%') 


# In[19]:


cosmetics['Oily'].value_counts().plot.pie(autopct='%1.1f%%') 


# In[20]:


cosmetics['Normal'].value_counts().plot.pie(autopct='%1.1f%%') 


# In[21]:


cosmetics['Sensitive'].value_counts().plot.pie(autopct='%1.1f%%') 


# In[22]:


cosmetics['Combination'].value_counts().plot.pie(autopct='%1.1f%%') 


# In[23]:


sns.heatmap(cosmetics.corr(),annot=True)


# In[24]:


cosmetics['Label'].unique().sum()


# In[25]:


cosmetics['Brand'].unique().sum()


# In[26]:


cosmetics['Name'].unique().sum()


# In[27]:


sns.FacetGrid(cosmetics,hue='Rank', height=6).map(plt.scatter,'Label','Price').add_legend()


# In[28]:


cosmetics['Label'].value_counts().plot.pie(autopct='%1.1f%%') 


# In[29]:


pip install sklearn


# In[30]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report,ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


# In[31]:


target= cosmetics['Label']
inputs= cosmetics.drop(['Label'],axis=1)


# In[32]:


x_train, x_test, y_train, y_test= train_test_split(inputs, target, test_size=0.2,random_state=365, stratify=target)


# In[33]:


y_train.value_counts(normalize= True)


# In[34]:


EnI= OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
EnT= LabelEncoder()


# In[35]:


x_train_transf = EnI.fit_transform(x_train)
x_test_transf = EnI.transform(x_test)

y_train_transf = EnT.fit_transform(y_train)
y_test_transf = EnT.transform(y_test)


# In[36]:


scaling = MinMaxScaler(feature_range=(-1,1)).fit(x_train_transf)
x_train_rescaled = scaling.transform(x_train_transf)
x_test_rescaled = scaling.transform(x_test_transf)


# In[37]:


ANN= MLPClassifier(solver='adam', hidden_layer_sizes=(20,))


# In[38]:


ANN.fit(x_train_rescaled, y_train_transf)


# In[39]:


predictions=ANN.predict(x_test_rescaled)


# In[40]:


print(classification_report(predictions, y_test_transf))


# In[41]:


ANN= MLPClassifier(solver='adam', hidden_layer_sizes=(8,))


# In[42]:


ANN.fit(x_train_rescaled, y_train_transf)


# In[43]:


predictions=ANN.predict(x_test_rescaled)


# In[44]:


print(classification_report(predictions, y_test_transf))


# In[45]:


ANN= MLPClassifier(solver='adam', hidden_layer_sizes=(12,))


# In[46]:


ANN.fit(x_train_rescaled, y_train_transf)


# In[47]:


predictions=ANN.predict(x_test_rescaled)


# In[48]:


print(classification_report(predictions, y_test_transf))


# In[49]:


ANN= MLPClassifier(solver='sgd', hidden_layer_sizes=(6,))


# In[50]:


ANN.fit(x_train_rescaled, y_train_transf)


# In[51]:


predictions=ANN.predict(x_test_rescaled)


# In[52]:


print(classification_report(predictions, y_test_transf))


# In[53]:


ANN= MLPClassifier(solver='sgd', hidden_layer_sizes=(12,))


# In[54]:


ANN.fit(x_train_rescaled, y_train_transf)


# In[55]:


predictions=ANN.predict(x_test_rescaled)


# In[56]:


print(classification_report(predictions, y_test_transf))


# In[57]:


ANN= MLPClassifier(solver='sgd', hidden_layer_sizes=(18,))


# In[58]:


ANN.fit(x_train_rescaled, y_train_transf)


# In[59]:


predictions=ANN.predict(x_test_rescaled)


# In[60]:


print(classification_report(predictions, y_test_transf))


# In[61]:


ANN= MLPClassifier(solver='lbfgs', hidden_layer_sizes=(4,))


# In[62]:


ANN.fit(x_train_rescaled, y_train_transf)


# In[63]:


predictions=ANN.predict(x_test_rescaled)


# In[64]:


print(classification_report(predictions, y_test_transf))


# In[65]:


ANN= MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10,))


# In[66]:


ANN.fit(x_train_rescaled, y_train_transf)


# In[67]:


predictions=ANN.predict(x_test_rescaled)


# In[68]:


print(classification_report(predictions, y_test_transf))


# In[69]:


ANN= MLPClassifier(solver='lbfgs', hidden_layer_sizes=(2,))


# In[70]:


ANN.fit(x_train_rescaled, y_train_transf)


# In[71]:


predictions=ANN.predict(x_test_rescaled)


# In[72]:


print(classification_report(predictions, y_test_transf))


# In[73]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics


# In[74]:


df = pd.DataFrame(cosmetics)


# In[75]:


print (df)
print(df.dtypes)


# In[76]:


df[pd.to_numeric(df['Label'], errors='coerce').notnull()]


# In[79]:


sns.scatterplot(data=cosmetics, x="Price", y="Label", hue="Rank")


# In[80]:


import numpy as np
import pandas as pd


# In[81]:


df_clean = df[df.Label.astype('str').str.isnumeric()]


# In[82]:


df_clean = df[df.Brand.astype('str').str.isnumeric()]


# In[83]:


df_clean = df[df.Price.astype('str').str.isnumeric()]


# In[84]:


df_clean = df[df.Rank.astype('str').str.isnumeric()]


# In[85]:


df_clean = df[df.Combination.astype('str').str.isnumeric()]


# In[86]:


df_clean = df[df.Dry.astype('str').str.isnumeric()]


# In[87]:


df_clean = df[df.Oily.astype('str').str.isnumeric()]


# In[88]:


df_clean = df[df.Normal.astype('str').str.isnumeric()]


# In[89]:


df_clean = df[df.Sensitive.astype('str').str.isnumeric()]


# In[90]:


df_clean = df[df.Name.astype('str').str.isnumeric()]


# In[91]:


df_clean = df[df.Ingredients.astype('str').str.isnumeric()]


# In[92]:


# make dataframe of column data types
col_types = df.dtypes.to_frame()
col_types.columns = ['dtype']

#make list of zeros
drop_it = [0]*col_types.shape[0]
k = 0

#make it a one if the data isn't numeric
#if you have other numeric types you need to add them to if statement
for t in col_types.dtype:
    if t != 'int64' and t != 'float64':
        drop_it[k] = 1
    k = k + 1

#delete types from drop list that aren't numeric
col_types['drop_it'] = drop_it
col_types = col_types.loc[col_types["drop_it"] == 1]

#finally drop columns that are in drop list
for col_to_drop in col_types.index.values.tolist():
    df = df.drop([col_to_drop], axis = 1)


# In[93]:


from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


# In[112]:


col_names = ['Label', 'Oily', 'Dry', 'Combination', 'Normal', 'Sensitive']
# load dataset
pima = pd.read_csv(r'C:\Users\DELL\Downloads\archive\cosmetics.csv', header=None, names=col_names)


# In[113]:


pima.head()


# In[114]:


#split dataset in features and target variable
feature_cols = ['Label', 'Dry', 'Combination', 'Normal', 'Sensitive']
X = pima[feature_cols] # Features
y = pima.Label # Target variable


# In[115]:


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test


# In[116]:


# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[ ]:




