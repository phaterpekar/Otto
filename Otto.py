#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pandas import read_csv
from xgboost import XGBClassifier 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV, ParameterGrid
from sklearn.model_selection import cross_val_score
import time
import numpy as np
from matplotlib import pyplot
from sklearn import model_selection
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix, auc




# In[3]:


# load data
train_data = read_csv('train.csv')
#test_data = read_csv('test.csv')


# In[4]:


# Training data dimensions
print(train_data.shape)
#print(test_data.shape)


# In[5]:


# SAMPLE 10 Rows from training dataset
train_data.sample(n=10)


# Each *column* represents a feature measured by an integer. Each *row* is an **Otto** product.
# The **Class Labels (targets)** are provided as character string in the last column. The **Otto** challenge is a multi class classification challenge.

# In[6]:


# checking datatypes of Training columns
train_data.dtypes


# In[ ]:





# In[7]:


# split data into X(features) and y(target labels)
X = train_data.values[:, 1:94]
y = train_data.values[:,94]

print("Input Features \n", X, "\n\n Target Labels \n", y)


# The classes are provided as character string in the **target column**. **XGBoost** doesn't support anything else than numbers. So we will convert classes to integers. Moreover, according to the documentation, it should start at 0.
# 

# In[8]:


# encode string class values as integers, a
label_encoded_y = LabelEncoder().fit_transform(y)
print(label_encoded_y)

# Check no. of unique labels
np.unique(label_encoded_y)


# In[9]:


"""Splitting data into train and test sets"""
train_X, valid_X, train_labels, valid_labels = model_selection.train_test_split(X, label_encoded_y,
                                                                                shuffle=True, test_size=0.20,
                                                                                random_state=42, 
                                                                                stratify= label_encoded_y)


# Before the learning we will use the cross validation to evaluate the our error rate.
# 
# Basically **XGBoost** will divide the training data in `nfold` parts, then **XGBoost** will retain the first part and use it as the test data. Then it will reintegrate the first part to the training dataset and retain the second part, do a training and so on...

# ## Parallel Thread XGBoost, Parallel Thread Cross-Validation
# 
# 

# In[182]:


# Building GRID SEARCH 
GS_model = XGBClassifier(objective= "multi:softprob",booster='gbtree',  n_jobs=-1)

subsample = [0.80, 1]
#learning_rate = [0.1, 0.2]
#max_depth = [2,3]
#min_child_weight = [1, 1.2]
#gamma = [0, 0.15]

param_grid = dict(subsample = subsample)
pg = ParameterGrid(param_grid)
len(pg)


# In[183]:


start = time.time()
kfold = StratifiedKFold(n_splits = 3, shuffle=True, random_state=42)
grid_search = GridSearchCV(GS_model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
grid_result = grid_search.fit(train_X, train_labels)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
elapsed = time.time() - start
print("Parallel Thread XGBoost, Parallel Thread CV, Elapsed Time: %f" % (elapsed))


# In[184]:


# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[21]:


len(valid_X), len(valid_labels)


# ## Single Thread XGBoost, Single Thread Cross-Validation
# 

# In[26]:


# Building GRID SEARCH 
GS_model = XGBClassifier(objective= "multi:softprob",booster='gbtree',  n_jobs=1)

subsample = [0.80, 1]
#learning_rate = [0.1, 0.2]
#max_depth = [2,3]
#min_child_weight = [1, 1.2]
#gamma = [0, 0.15]

param_grid = dict(subsample = subsample)
pg = ParameterGrid(param_grid)
print(len(pg))

start = time.time()
kfold = StratifiedKFold(n_splits = 3, shuffle=True, random_state=42)
grid_search = GridSearchCV(GS_model, param_grid, scoring="neg_log_loss", n_jobs=1, cv=kfold, verbose=1)
grid_result = grid_search.fit(train_X, train_labels)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
elapsed = time.time() - start
print("Parallel Thread XGBoost, Parallel Thread CV, Elapsed Time: %f" % (elapsed))


# In[27]:


# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# #### We can see that the Running Time without multithreading  is 782.629 more than twice longer than using Parallel Threads (i.e. all 6 cores of the local CPU) where the Running Time was 304.44.

# ## Final Model Training

# In[14]:


clf = XGBClassifier(n_jobs = -1,subsample=0.8, n_estimators=1000, learning_rate=0.1)
clf.fit(train_X, train_labels, early_stopping_rounds=100, eval_set = [(train_X, train_labels),(valid_X, valid_labels)])


# In[16]:


predict = clf.predict(valid_X)

print("Accuracy:", accuracy_score(valid_labels, predict))
print("Precision:", precision_score(valid_labels, predict, average='macro'),
      precision_score(valid_labels, predict, average='micro'), precision_score(valid_labels, predict, average='weighted'))
print(classification_report(valid_labels, predict))
print(confusion_matrix(valid_labels, predict))


# In[ ]:





# ## Evaluate the effect of the number of threads(parallelism)
# 

# In[32]:


results = []
num_threads = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


# In[34]:


for n in num_threads:
    start = time.time()
    model = XGBClassifier(n_jobs=n, subsample=0.8)
    model.fit(train_X, train_labels)
    elapsed = time.time() - start
    print("No. of parallel threads:", n,"Learning Time:", elapsed)
    results.append(elapsed)
    model_predict = model.predict(valid_X)
    #print(classification_report(valid_labels, model_predict))
    print("Accuracy:", accuracy_score(valid_labels, predict))


# ## Plot Learning Time vs Number of Threads 

# In[35]:


# plot Evaluation results
pyplot.plot(num_threads, results)
pyplot.ylabel('Learning Time (seconds)')
pyplot.xlabel('Number of Threads')
pyplot.title('XGBoost Training Speed vs Number of Threads')
pyplot.show()


# #### We can see a nice trend in the decrease in execution time as the number of threads is increased. 
# 
# For this local execution we ran this on on a **6-core intel processor, which has a total of 12 cores: 6 physical and an additional 6 virtual cores.***
# 
# #### It is interesting to note that we do not see much improvement beyond the no. of physical cores i.e. 6 threads (at about 40 seconds), after which the gains are more or less absent. 
# 
# The results suggest that if you have a machine with hyperthreading, you may want to set num_threads to equal the number of physical CPU cores in your machine.

# ### For Parallel-Cross Validation we can also compare the following scenarios:

# In[89]:


# Prepare cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

# Single Thread XGBoost, Parallel Thread CV
start = time.time()
model = XGBClassifier(nthread=1)
results = cross_val_score(model, X, train_labels, cv=kfold, scoring='neg_log_loss', n_jobs=-1)
elapsed = time.time() - start


# In[90]:


# Parallel Thread XGBoost, Single Thread CV
start = time.time()
model = XGBClassifier(nthread=-1)
results = cross_val_score(model, X, label_encoded_y, cv=kfold, scoring='neg_log_loss', n_jobs=1)
elapsed = time.time() - start
print("Parallel Thread XGBoost, Single Thread CV: %f" % (elapsed))


# In[91]:


# Parallel Thread XGBoost and CV
start = time.time()
model = XGBClassifier(nthread=-1)
results = cross_val_score(model, X, label_encoded_y, cv=kfold, scoring='neg_log_loss', n_jobs=-1)
elapsed = time.time() - start
print("Parallel Thread XGBoost and CV: %f" % (elapsed))


# In[93]:


print(results)


# **FEATURE IMPORTANCE**

# In[ ]:




