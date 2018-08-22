#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 16:21:11 2018

@author: pooh
"""
import tqdm
import pandas as pd
def get_rank(x):
    return pd.Series(x).rank(pct=True).values
import numpy as np
patch='/home/ai/Documents/AI/santa/csv/'
data1 = pd.read_csv(patch+'leak_seed.csv')
#data2 = pd.read_csv(patch+'count_feature_leak.csv')
data3 = pd.read_csv(patch+'xgb_leak_seed.csv')
#data4 = pd.read_csv(patch+'flag37_lgb.csv')
#data5 = pd.read_csv(patch+'xgb_leak_flag37.csv')
data1['target'] = (data1['target']*0.5+data3['target']*0.5)#+data3['target']*0.33)
data1.to_csv(patch+'Tru_just_2_times_seed.csv',index = False)

