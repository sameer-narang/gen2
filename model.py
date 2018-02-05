import datetime
import pandas as pd
import numpy as np
import sklearn

from datetime import timedelta
from dateutil.relativedelta import relativedelta
from sklearn.ensemble import RandomForestRegressor

#NA_FILL_VAL = 1e-9
NA_FILL_VAL = None

DATE_FMT = "%m/%d/%Y"

def summarize (y_act, y_pred, info):
	print (info + 'MSE: ' + str (np.mean ((y_act-y_pred)**2)))


df = pd.read_csv ('data2.csv')
df ['Date'] = df ['Date'].apply (datetime.datetime.strptime, args=(DATE_FMT,))
df ['prev_qtr'] = df ['prev_qtr'].apply (datetime.datetime.strptime, args=(DATE_FMT,))
df ['next_qtr'] = df ['next_qtr'].apply (datetime.datetime.strptime, args=(DATE_FMT,))
df ['prev_qtr_start'] = df ['prev_qtr_start'].apply (datetime.datetime.strptime, args=(DATE_FMT,))
df = df [df ['Seq'] > 10000]


y_train = df [(df ['Seq'] >= 10958) & (df ['Seq'] < 23376)] ['gdp_label']
y_val = df [(df ['Seq'] >= 23376) & (df ['Seq'] < 24106) ] ['gdp_label']
y_test = df [(df ['Seq'] >= 24106) & (df ['Seq'] < 24838) ] ['gdp_label']

ts = [
		'GTII10 Govt'
		]

features = [
			'EHGDUS Index',
			'days_to_go',
			'spx_ratio',
			'vix_ratio'
			]

for s in ts:
	suffix = "_ratio"
	df [s + suffix] = np.nan
	features.append (s + suffix)

import pdb; pdb.set_trace ()
for i, row in df.iterrows():
	for s in ts:
  		running_avg_ratio = np.nanmean (df [(df ['Date'] > row ['prev_qtr']) & (df ['Date'] <= row ['next_qtr'])] [s])/ \
  						np.nanmean (df [(df ['Date'] >= row ['prev_qtr_start']) & (df ['Date'] <= row ['prev_qtr'])] [s])
  		if np.isfinite (running_avg_ratio):
  			df.set_value(i, s, running_avg_ratio)

if NA_FILL_VAL:
	df = df.fillna (NA_FILL_VAL)
else:
	df = df.fillna (df.mean ())

df.to_csv ('data_new.csv')

df = df.drop (['Date', 'next_qtr', 'prev_qtr'], axis=1)

X_train = df [(df ['Seq'] >= 10958) & (df ['Seq'] < 23376)] [features]
X_val = df [(df ['Seq'] >= 23376) & (df ['Seq'] < 24106) ] [features]
X_test = df [(df ['Seq'] >= 24106) & (df ['Seq'] < 24838) ] [features]

'''
X_train = X_train.drop (['Seq', 'gdp_label'], axis=1)
X_val = X_val.drop (['Seq', 'gdp_label'], axis=1)
X_test = X_test.drop (['Seq', 'gdp_label'], axis=1)
'''

mdl = RandomForestRegressor (n_estimators=500)
mdl.fit (X_train, y_train)

train_preds = mdl.predict (X_train)
val_preds = mdl.predict (X_val)
test_preds = mdl.predict (X_test)

summarize (y_train, train_preds, 'Training')
summarize (y_val, val_preds, 'Vallidation')
summarize (y_test, test_preds, 'Test')

print ("---------- Regularized below ----------")

mdl2 = RandomForestRegressor (n_estimators=500, max_leaf_nodes=500, \
            max_features=0.3, max_depth=30, min_samples_split=7, \
            min_samples_leaf=4, min_impurity_decrease=0.05)
mdl2.fit (X_train, y_train)

train_preds2 = mdl2.predict (X_train)
val_preds2 = mdl2.predict (X_val)
test_preds2 = mdl2.predict (X_test)

summarize (y_train, train_preds2, 'Training')
summarize (y_val, val_preds2, 'Vallidation')
summarize (y_test, test_preds2, 'Test')