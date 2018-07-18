# coding: utf-8
import pandas as pd

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

label = 'target' #label
features = list(set(train.columns) - {label,'ID'})

X = train[features]
y = train[label]

#%%
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# In[21]:
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# In[22]:
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X, y)

# In[24]:
features_for_test = list(set(test.columns) - {'ID'})
final_prediction = model.predict(test[features_for_test])

# In[26]:
ID = test['ID']
target = pd.DataFrame(final_prediction, columns = ['target'])
target[target < 0] = 0
result = pd.concat([ID, target], axis=1)
result.to_csv("submission.csv", index = False)

