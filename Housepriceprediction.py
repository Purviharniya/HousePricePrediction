#!/usr/bin/env python
# coding: utf-8

# In[63]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import re


# In[64]:


df = pd.read_csv('Bengaluru_House_Data.csv')
print("data imported successfully")


# In[65]:


df.head(3)


# In[66]:


df.shape


# # Data cleaning

# In[67]:


df.groupby('area_type')['area_type'].agg('count')
# df['area_type'].value_counts() #does the same job


# In[68]:


#dropped the columns not requried
df1 = df.drop(['area_type','availability','society','balcony'],axis='columns')
df1.head()


# In[69]:


df1.isna().sum()


# In[70]:


df2 = df1.dropna()
df2.isna().sum()


# ### Improving the Size column

# In[71]:


df2['size'].unique()


# In[72]:


df2['bhk'] = df2['size'].apply(lambda x: int(x.split()[0]))


# In[73]:


df2.head()
df2['bhk'].unique()


# In[74]:


df2[df2['bhk']>20]


# In[75]:


df2['total_sqft'].unique()


# In[76]:


#getting all the types of values in the coulmn total_sqft that are not float
def checkFloat(x):
    try:
        x= float(x)
        return True
    except:
        return False
    return True
df3=df2[~df2['total_sqft'].apply(checkFloat)]
df3['total_sqft'].head(15)


# ### Converting non-float sqft_area to float values

# In[77]:


# def convert_totalsqft(x):
#     if('-' in x):
#         li=x.split('-')
#         if(len(li)==2): #if the list is a range form
#             return (float(li[0])+float(li[1]))/2 #return the average of range
#     else:
#         li = re.match(r"([0-9]+)([a-z]+)", x , re.I)
# #         print(li)
#         if li:
#             temp = list(li.groups())
#             if temp[1] == 'Sq. Yards':
#                 return float(temp[0])*9
#             if temp[1] == 'Sq. Meter':
#                 return float(temp[0])*10.7639
#             if temp[1] == 'Perch':    
#                 return float(temp[0])*272.25
#             if temp[1] == 'Acres':
#                 return float(temp[0])*43560
#             if temp[1] == 'Guntha':
#                 return float(temp[0])*1089
#         try: #return only float if a number
#             return float(x) 
#         except: #if none of those, then we return none
#             return None


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


# In[78]:


df4=df2.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)


# ## Feature Engineering

# In[79]:


#creating a new column => price per sq ft
df4['price_per_sqft'] = (df4['price']*100000)/df4['total_sqft']
df4.head(2)


# ### Converting all the locations that occur less than 10 times to "other"

# In[80]:


df4['location'].nunique()
#toooo many locations to deal with


# In[81]:


df4['location']=df4['location'].apply(lambda x: x.lower())

location_stats = df4.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats


# In[82]:


print(len(location_stats[location_stats<=10]))#locations that occur less than 10 times
stats_less_then_10 = location_stats[location_stats<=10]
#so there are 1053 locations that occur less than 10 times 


# In[83]:


#converting the location value to "other" for the ones with less than 10 occurences
df4['location'] = df4['location'].apply(lambda x: 'other' if x in stats_less_then_10 else x)
df4.head(10) 


# # Outlier Removal

# In[84]:


#in real estate, the threshold for sqft is 300sqft per bedroom, 
#so here we find all the rows or houses that do not follow this threshold or have total_Sqft/bhk<300

df4[df4['total_sqft']/df4['bhk']<300].head(5)


# In[85]:


# we will remove all these values by negating the condition and create a new dataframe for the same 
df5 = df4[~(df4['total_sqft']/df4['bhk']<300)]
df5.shape


# In[86]:


#exploring and handling price_per_sqft column
df5['price_per_sqft'].describe()


# In[87]:


def remove_pricepersqft_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m=np.mean(subdf['price_per_sqft'])
        st=np.std(subdf['price_per_sqft'])
        reduced_df = subdf[(subdf['price_per_sqft']>(m-st)) & (subdf['price_per_sqft']<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out

df6 = remove_pricepersqft_outliers(df5)
df6.shape


# In[88]:


#there are many 3 bhk flats with price lower than 2 bhk flats. This seems to be an anomaly. So this should be handled

def plt_scatter(df,location):
    bhk2 = df[(df['location']==location ) & (df['bhk']==2)]
    bhk3 = df[(df['location']==location ) & (df['bhk']==3)]
    plt.figure(figsize=(14,10))
    plt.scatter(bhk2['total_sqft'],bhk2['price'],color='pink',label='2 BHK', s=50)
    plt.scatter(bhk3['total_sqft'],bhk3['price'],color='red',marker='+',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel('Price')
    plt.title(location)
    plt.legend()
    
plt_scatter(df6,'rajaji nagar')


# In[89]:


def remove_bhk(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')

df7 = remove_bhk(df6)
df7.shape


# In[90]:


def plt_scatter(df,location):
    bhk2 = df[(df['location']==location ) & (df['bhk']==2)]
    bhk3 = df[(df['location']==location ) & (df['bhk']==3)]
    plt.figure(figsize=(14,10))
    plt.scatter(bhk2['total_sqft'],bhk2['price'],color='pink',label='2 BHK', s=50)
    plt.scatter(bhk3['total_sqft'],bhk3['price'],color='red',marker='+',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel('Price')
    plt.title(location)
    plt.legend()
    
plt_scatter(df7,'rajaji nagar')


# In[91]:


plt.figure(figsize=(14,10))
plt.hist(df7['price_per_sqft'],rwidth=0.8)
plt.xlabel('Price per sq feet')
plt.ylabel('count')
plt.show()


# In[92]:


plt.figure(figsize=(14,10))
plt.hist(df7['bath'],rwidth=0.8)
plt.xlabel('Bathrooms')
plt.ylabel('count')
plt.show()


# In[93]:


df7[df7['bath']>df7['bhk']+2]


# In[94]:


df8 = df7[df7['bath']<df7['bhk']+2]
df8.shape


# # MODEL BUILDING

# In[95]:


df9= df8.drop(['size','price_per_sqft'],axis='columns')
df9.head()


# In[96]:


dummies = pd.get_dummies(df9['location'])
dummies


# In[97]:


df10 = pd.concat([df9,dummies.drop('yeshwanthpur',axis='columns')],axis='columns')
df10.head(3)


# In[98]:


df10 = df10.drop('location',axis='columns')


# In[99]:


df10.head(3)


# In[100]:


df10.shape


# In[101]:


X = df10.drop('price',axis='columns')
X.head(2)


# In[102]:


y = df10['price']
y.head(2)


# In[103]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=10)


# In[104]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train,y_train)
lr.score(X_test,y_test)


# # Use K Fold cross validation to measure accuracy of our LinearRegression model

# In[105]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score 

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), X, y, cv=cv)


# # Find best model using GridSearchCV

# In[106]:


from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(X,y)


# In[109]:


def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr.predict([x])[0]


# In[110]:


predict_price('indira nagar',1000, 2, 2)


# In[112]:


import pickle
with open('house_price_predict_model.pickle','wb') as f:
    pickle.dump(lr,f)


# # Export location and column information to a file that will be useful later on in our prediction application

# In[113]:


import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))


# In[ ]:




