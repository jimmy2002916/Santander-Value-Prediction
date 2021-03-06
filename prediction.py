# coding: utf-8
import pandas as pd

train_original = pd.read_csv("data/train.csv")
test_original = pd.read_csv("data/test.csv")

train = train_original
test = test_original

label = 'target' #label
features = list(set(train.columns) - {label,'ID'})

X = train[features]
y = train[label]

#for Keras
X = X.as_matrix()
y = y.as_matrix()

#%%
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# In[21]:
# Feature Scaling
print("Feature scaling...")
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# In[22]:
print("training...")
### Change the File output name into the model you use here. 
print("Building ANN")
from sklearn.model_selection import cross_val_score, KFold
from keras.models import Sequential
from sklearn.metrics import accuracy_score
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
seed = 1

def baseline_model():
    model = Sequential()
    model.add(Dense(units = 4991, input_dim = 4991, activation='relu'))
    model.add(Dense(units = 4991, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


model = KerasRegressor(build_fn = baseline_model)        
model.fit(X, y,
          batch_size = 40, 
          epochs = 100
          )
#y_pred = model.predict(X_test)

#%%
"""
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=2)
auc = metrics.auc(fpr, tpr)
"""
# In[24]:
print("predicting... ")
features_for_test = list(set(test.columns) - {'ID'})
test = test[features_for_test]
test = test.as_matrix()
final_prediction = model.predict(test)
accuracy_score(y, final_prediction)


# In[26]:
print("Outputing file...")
test = test_original
ID = test['ID']
target = pd.DataFrame(final_prediction, columns = ['target'])
target[target < 0] = 0
result = pd.concat([ID, target], axis=1)
result.to_csv("submission_keras.csv", index = False)

