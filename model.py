import pandas as pd
import numpy as np
import sklearn

from sklearn.ensemble import RandomForestRegressor

NA_FILL_VAL = 1e-9 

def summarize (y_act, y_pred, info):
	print (info + 'MSE: ' + str (np.mean ((y_act-y_pred)**2)))

df = pd.read_csv ('data2.csv')

df = df.fillna (NA_FILL_VAL)
df = df.drop (['Date', 'next_qtr', 'prev_qtr'], axis=1)

y_train = df [(df ['Seq'] >= 10958) & (df ['Seq'] < 23376)] ['gdp_label']
y_val = df [(df ['Seq'] >= 23376) & (df ['Seq'] < 24106) ] ['gdp_label']
y_test = df [(df ['Seq'] >= 24106) & (df ['Seq'] < 24838) ] ['gdp_label']

X_train = df [(df ['Seq'] >= 10958) & (df ['Seq'] < 23376)]
X_val = df [(df ['Seq'] >= 23376) & (df ['Seq'] < 24106) ]
X_test = df [(df ['Seq'] >= 24106) & (df ['Seq'] < 24838) ]

X_train = X_train.drop (['Seq', 'gdp_label'], axis=1)
X_val = X_val.drop (['Seq', 'gdp_label'], axis=1)
X_test = X_test.drop (['Seq', 'gdp_label'], axis=1)


mdl = RandomForestRegressor (n_estimators=500)

mdl.fit (X_train, y_train)

train_preds = mdl.predict (X_train)
val_preds = mdl.predict (X_val)
test_preds = mdl.predict (X_test)

summarize (y_train, train_preds, 'Training')
summarize (y_val, val_preds, 'Vallidation')
summarize (y_test, test_preds, 'Test')