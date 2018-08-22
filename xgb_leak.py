import xgboost as xgb
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold,BaseCrossValidator
from scipy.stats import skew, kurtosis, gmean, ks_2samp
import time
import gc
gc.enable()

col=[   'ba42e41fa','3f4a39818','371da7669','b98f3e0d7','2288333b4',
        '84d9d1228','de4e75360','20aa07010','1931ccfdd','c2dae3a5a',
        'f190486d6', 'c47340d97', 'eeb9cd3aa', '66ace2992', 'e176a204a',
        '491b9ee45', '1db387535', 'c5a231d81', '0572565c2', '024c577b9',
        '15ace8c9f', '23310aa6f', '9fd594eec', '58e2e02e6', '91f701ba2',
        'adb64ff71', '2ec5b290f', '703885424', '26fc93eb7', '6619d81fc',
        '0ff32eb98', '70feb1494', '58e056e12', '1931ccfdd', '1702b5bf0',
        '58232a6fb', '963a49cdc', 'fc99f9426', '241f0f867', '5c6487af1',
        '62e59a501', 'f74e8f13d', 'fb49e4212', '190db8488', '324921c7b',
        'b43a7cfd5', '9306da53f', 'd6bb78916', 'fb0f5dbfe', '6eef030c1'
    ]
col2=['6eef030c1', 'ba42e41fa', '703885424', 'eeb9cd3aa', '3f4a39818',
       '371da7669', 'b98f3e0d7', 'fc99f9426', '2288333b4', '324921c7b',
       '66ace2992', '84d9d1228', '491b9ee45', 'de4e75360', '9fd594eec',
       'f190486d6', '62e59a501', '20aa07010', 'c47340d97', '1931ccfdd',
       'c2dae3a5a', 'e176a204a']

nrows=None
patch='/home/ai/Documents/AI/santa/input/'
patch1='/home/ai/Documents/AI/santa/csv_leak_data/'
patchcsv='/home/ai/Documents/AI/santa/csv/'
def get_data():
    print('Reading data')
    data = pd.read_csv(patch+'train.csv', nrows=nrows)
    test = pd.read_csv(patch+'test.csv', nrows=nrows)
    print('Train shape ', data.shape, ' Test shape ', test.shape)
    return data, test
data,test=get_data()
target = np.log1p(data['target'])
y = data[['ID', 'target']].copy()
del data['target'], data['ID']
sub = test[['ID']].copy()
del test['ID']
data_new=data[col2]
test_new=test[col2]
leak = pd.read_csv(patch1+'train_leak.csv')
data_new['leak'] = leak['compiled_leak'].values
data_new['log_leak'] = np.log1p(leak['compiled_leak'].values)
tst_leak = pd.read_csv(patch1+'test_leak.csv')
test_new['leak'] = tst_leak['compiled_leak']
test_new['log_leak'] = np.log1p(tst_leak['compiled_leak'])
sub['leak'] = tst_leak['compiled_leak']
sub['log_leak'] = np.log1p(tst_leak['compiled_leak'])
y['leak'] = leak['compiled_leak'].values
y['log_leak'] = np.log1p(leak['compiled_leak'].values)
#############################################
def add_statistics(data, test,col):
    # This is part of the trick I think, plus lightgbm has a special process for NaNs
    data.replace(0, np.nan, inplace=True)
    test.replace(0, np.nan, inplace=True)
    
    for df in [data, test]:
        df['nb_nans'] = df[col].isnull().sum(axis=1)
        df['the_median'] = df[col].median(axis=1)
        df['the_mean'] = df[col].mean(axis=1)
        df['the_sum'] = df[col].sum(axis=1)
        df['the_std'] = df[col].std(axis=1)
        df['the_kur'] = df[col].kurtosis(axis=1)
        df['the_max'] = df[col].max(axis=1)
        df['the_skew'] = df[col].skew(axis=1)
        df['the_log_mean']= np.log(df[col].mean(axis=1))
        df['the_count']= df[col].count(axis=1)
        df['the_min'] = df[col].min(axis=1)
        df['the_kurtosis'] = df[col].kurtosis(axis=1)
        
        #df['the_gmean']= gmean(df,axis=1)
        
        
        
        
    data[col].fillna(-1, inplace=True)
    test[col].fillna(-1, inplace=True)
       
    return data, test
##############################################################################
NUM_OF_DECIMALS = 32
data = data.round(NUM_OF_DECIMALS)
test = test.round(NUM_OF_DECIMALS)

train_zeros = pd.DataFrame({'Percent_zero':((data.values)==0).mean(axis=0),
                           'Column' : data.columns})

high_vol_columns = train_zeros['Column'][train_zeros['Percent_zero'] < 0.70].values
low_vol_columns = train_zeros['Column'][train_zeros['Percent_zero'] >= 0.70].values
#########################################################
data = data.replace({0:np.nan})
test = test.replace({0:np.nan})



cluster_sets = {"low":low_vol_columns, "high":high_vol_columns}
for cluster_key in cluster_sets:
    for df in [data,test]:
        df['nb_nan_all'] = df.isnull().sum(axis=1)
        df["count_not0_"+cluster_key] = df[cluster_sets[cluster_key]].count(axis=1)
        df["sum_"+cluster_key] = df[cluster_sets[cluster_key]].sum(axis=1)
        df["var_"+cluster_key] = df[cluster_sets[cluster_key]].var(axis=1)
        df["median_"+cluster_key] = df[cluster_sets[cluster_key]].median(axis=1)
        df["mean_"+cluster_key] = df[cluster_sets[cluster_key]].mean(axis=1)
        df["std_"+cluster_key] = df[cluster_sets[cluster_key]].std(axis=1)
        df["max_"+cluster_key] = df[cluster_sets[cluster_key]].max(axis=1)
        df["min_"+cluster_key] = df[cluster_sets[cluster_key]].min(axis=1)
        df["skew_"+cluster_key] = df[cluster_sets[cluster_key]].skew(axis=1)
        df["kurtosis_"+cluster_key] = df[cluster_sets[cluster_key]].kurtosis(axis=1)

data_more_simplified = data.drop(high_vol_columns,axis=1).drop(low_vol_columns,axis=1)
test_more_simplified = test.drop(high_vol_columns,axis=1).drop(low_vol_columns,axis=1)
statistic_fea=data_more_simplified.columns
#data, test=add_statistics(data, test)
data_new, test_new=add_statistics(data_new, test_new,col2)
#col2=col+['nb_nans', 'the_median', 'the_mean', 'the_sum', 'the_std', 'the_kur','the_max','the_log_mean','the_count']
#data=data[col2]
#test=test[col2]
#col1=['nb_nans', 'the_median', 'the_mean', 'the_sum', 'the_std', 'the_kur','the_max','the_log_mean','the_count']
for c in statistic_fea:
    data_new[c]=data_more_simplified[c]
    test_new[c]=test_more_simplified[c]

###############################################################################
from scipy.sparse import hstack
X_tr = hstack([data_more_simplified,data_new]).tocsr()
X_te = hstack([test_more_simplified,test_new]).tocsr()
sub_preds = np.zeros(test.shape[0])
oof_preds = np.zeros(data.shape[0])
def rmsle(y_true, y_pred):
    errors = (np.log1p(y_true) - np.log1p(y_pred)) ** 2

    return np.sqrt(np.mean(errors))
class KFoldByTargetValue(BaseCrossValidator):
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def _iter_test_indices(self, X, y=None, groups=None):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        sorted_idx_vals = sorted(zip(indices, X), key=lambda x: x[1])
        indices = [idx for idx, val in sorted_idx_vals]

        for split_start in range(self.n_splits):
            split_indeces = indices[split_start::self.n_splits]
            yield split_indeces

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_split
cv = KFoldByTargetValue(n_splits=5, shuffle=True, random_state=101)   
xgb_params = {
        'n_estimators': 360,
        'objective': 'reg:linear',
        'booster': 'gbtree',
        'learning_rate': 0.05,
        'max_depth': 8,
        'min_child_weight': 50,#57,
        'gamma' : 1.2,
        'alpha': 0.05,
        'lambda': 0.0,
        'subsample': 0.67,
        'colsample_bytree': 0.054,
        'colsample_bylevel': 0.50,
        'n_jobs': -1,
        'random_state': 0
    }
fit_params = {
    'eval_metric': 'rmse',
    'verbose': False
}
for i, (train_index, valid_index) in enumerate(cv.split(target)):
    print(f'Fold {i}')
    data_tr=X_tr[train_index,:]
    data_va=X_tr[valid_index,:]
    label_tr=target[train_index]
    label_va=target[valid_index]
    tr_data = xgb.DMatrix(data_tr, label_tr)
    va_data = xgb.DMatrix(data_va, label_va)
    watchlist = [(tr_data, 'train'), (va_data, 'valid')]
    model=xgb.train(xgb_params, tr_data, 2000, watchlist, maximize=False, early_stopping_rounds = 30, verbose_eval=100)
    sub_preds+=model.predict(xgb.DMatrix(X_te), ntree_limit=model.best_ntree_limit)/5    
    oof_preds[valid_index]= model.predict(xgb.DMatrix(data_va),ntree_limit=model.best_ntree_limit)
    print('score_CV',mean_squared_error(target[valid_index], oof_preds[valid_index]) ** .5)
y['predictions'] = oof_preds
y.loc[y['leak'].notnull(), 'predictions'] = np.log1p(y.loc[y['leak'].notnull(), 'leak'])
print('Full Out-Of-Fold score : %9.6f' 
          % (mean_squared_error(target, oof_preds) ** .5))
print('OOF SCORE with LEAK : %9.6f' 
      % (mean_squared_error(target, y['predictions']) ** .5))
    # Store predictions

y[['ID', 'target', 'predictions']].to_csv(patchcsv+'oof_xgb.csv', index=False)
sub['target'] = np.expm1(sub_preds)
sub.loc[sub['leak'].notnull(), 'target'] = sub.loc[sub['leak'].notnull(), 'leak']
sub[['ID', 'target']].to_csv(patchcsv+'xgb_leak.csv', index=False)
    