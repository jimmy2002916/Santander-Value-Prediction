# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

train_original = pd.read_csv("data/train.csv")
test_original = pd.read_csv("data/test.csv")

#%%
train = train_original
test = test_original

label = 'target' #label
features = list(set(train.columns) - {label,'ID'})
features_for_test = list(set(test.columns) - {'ID'})
test = test[features_for_test]

X = train[features]
y = train[label]

x_train = X
y_train = y
x_test = test

#%%
"""
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""
# In[21]:
# Feature Scaling
"""
print("Feature scaling...")
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
"""

# In[22]:
print("training...")
def Stacking(model,train,y,test,n_fold):
   folds = KFold(n_splits=n_fold,random_state=1)
   test_pred=np.empty((test.shape[0],1),float)
   train_pred=np.empty((0,1),float)
   for train_indices,val_indices in folds.split(train,y.values):
      x_train,x_val=train.iloc[train_indices],train.iloc[val_indices]
      y_train,y_val=y.iloc[train_indices],y.iloc[val_indices]
      x_train = x_train.as_matrix()
      y_train = y_train.as_matrix()
      model.fit(X=x_train,y=y_train)
      train_pred=np.append(train_pred,model.predict(x_val))
      test_pred=np.append(test_pred,model.predict(test))
   return test_pred.reshape(-1,1),train_pred, model

#%%
print("model 1")
from xgboost import XGBRegressor
model1 = XGBRegressor()            

test_pred1 ,train_pred1, model_1 = Stacking(model = model1, n_fold = 5, train=x_train,test=x_test,y=y_train)
train_pred1 = pd.DataFrame(train_pred1)
test_pred1 = pd.DataFrame(test_pred1)
model_1.score(x_train, y_train) #same score!?!?

#%%
print("model 2")
from sklearn.tree import DecisionTreeRegressor
model2 = DecisionTreeRegressor()
 
test_pred2 ,train_pred2, model_2 = Stacking(model = model2, n_fold = 5, train=x_train,test=x_test,y=y_train)
train_pred2 = pd.DataFrame(train_pred2)
test_pred2 = pd.DataFrame(test_pred2)
model_2.score(x_train, y_train) #same score!?!?

#%%
df = pd.concat([train_pred1, train_pred2], axis=1)
df_test = pd.concat([test_pred1, test_pred2], axis=1)

from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(df, y_train)     
#y_pred = model.predict(X_test)

#%%
"""
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=2)
auc = metrics.auc(fpr, tpr)
"""
# In[24]:
print("predicting... ")
final_prediction = model.predict(df_test)

# In[26]:
print("Outputing file...")
test = test_original
ID = test['ID']
target = pd.DataFrame(final_prediction, columns = ['target'])
target[target < 0] = 0
result = pd.concat([ID, target], axis=1)
result.to_csv("submission_stacking_kfold.csv", index = False)

