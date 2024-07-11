#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 20:23:58 2024

@author: jing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle, bz2
import datetime
from collections import OrderedDict
import pylogit as pl
from scipy import stats
import ast
from itertools import permutations
from sympy import symbols, solve
import random
random.seed(42)
np.random.seed(42)
import warnings

warnings.filterwarnings("ignore")

try2 = pd.read_csv('forLogit0701.csv').drop('Unnamed: 0',axis=1)
try4 = try2.groupby(['initial_adl_time_x','CARRIER','ARR_LOCID']).count()[['hub']].reset_index()
try4 = try4[try4['hub']>1]

Custom_Choice_id = 0
convex = 1.1
for_logit = []
for i in range(len(try4)):
    try3 = try2[(try2['initial_adl_time_x']==try4.iloc[i]['initial_adl_time_x'])&(try2['ARR_LOCID']==try4.iloc[i]['ARR_LOCID'])&(try2['CARRIER']==try4.iloc[i]['CARRIER'])]
    try3 = try3.sort_values('EDCT_ON')
    for q in range(len(try3)-1):#q+1 th slot
        try9 = try3.tail(len(try3)-q)
        try9 = try9.sort_values('OAG_S_ARR')
        slot_time = np.sort(list(try9['EDCT_ON']))#sorted
        first_slot_time = slot_time[0]
        slot_time = slot_time[1:]
        sch_time = list(try9['OAG_S_ARR'])#sorted
        
        Custom_Choice_id = Custom_Choice_id + 1
        for j in range(len(try3)-q):
            
            if try9.iloc[j]['min_EDCT_on']<=first_slot_time:
                sch_time_new = sch_time[:j] + sch_time[j+1:]
                delay_convex = 0
                for p in range(len(try3)-q-1):
                    delay_convex = delay_convex + (max(0,-sch_time_new[p]+slot_time[p])/3600)**convex
                if try9.iloc[j]['EDCT_ON']==first_slot_time:
                    latent = int(try9.iloc[j]['OAG_S_ARR']<slot_time[0] - 2*3600)*int(try9.iloc[j]['OAG_S_ARR']>first_slot_time - 1*3600)
                    for_logit.append([Custom_Choice_id,1,j,delay_convex,(max(0,-sch_time[j]+list(try9['EDCT_ON'])[0])/3600)**convex,try9.iloc[j]['initial_adl_time_x'],
                                     try9.iloc[j]['hub'],try9.iloc[j]['Pass_on'],try9.iloc[j]['Seats'],try9.iloc[j]['avg_fare'],try9.iloc[j]['connect'],try9.iloc[j]['buff_time'],try9.iloc[j]['oversea'],try9.iloc[j]['#inter'],try9.iloc[j]['dep_delta'],1,latent])
                else:
                    latent = int(try9.iloc[j]['OAG_S_ARR']<slot_time[0] - 2*3600)*int(try9.iloc[j]['OAG_S_ARR']>first_slot_time - 1*3600)
                    for_logit.append([Custom_Choice_id,0,j,delay_convex,(max(0,-sch_time[j]+list(try9['EDCT_ON'])[0])/3600)**convex,try9.iloc[j]['initial_adl_time_x'],
                                     try9.iloc[j]['hub'],try9.iloc[j]['Pass_on'],try9.iloc[j]['Seats'],try9.iloc[j]['avg_fare'],try9.iloc[j]['connect'],try9.iloc[j]['buff_time'],try9.iloc[j]['oversea'],try9.iloc[j]['#inter'],try9.iloc[j]['dep_delta'],1,latent])
                    
forLogit = pd.DataFrame(for_logit,columns = ['Custom_Choice_id','Choice','swap','delayOthers','delayIt','GDP_time',
                                            'hub','Pass_on','Seats','avg_fare','connect','buff_time','oversea','numberInter','dep_delta','mode_id','latent'])

temp = forLogit.groupby('Custom_Choice_id').sum()
temp = temp[temp['Choice']==1].reset_index()
temp = list(temp['Custom_Choice_id'])
forLogit = forLogit[forLogit['Custom_Choice_id'].isin(temp)]


temp = forLogit.groupby('Custom_Choice_id').count()
temp = temp[temp['Choice']>=2].reset_index()
temp = list(temp['Custom_Choice_id'])
temp = random.sample(temp, 20000)
forLogit = forLogit[forLogit['Custom_Choice_id'].isin(temp)]

forLogit['Connect'] = forLogit.apply(lambda row: int(row.connect),axis=1)
forLogit['Hub'] = forLogit.apply(lambda row: int(row.hub),axis=1)
forLogit['Revenue'] = forLogit.apply(lambda row: row.avg_fare*row.Pass_on,axis=1)
forLogit['SeatInter'] = forLogit.apply(lambda row: row.Seats*row.numberInter,axis=1)

custom_alt_id = 'mode_id'
obs_id_column = 'Custom_Choice_id'
choice_column = 'Choice'

basic_specification = OrderedDict()
basic_names = OrderedDict()

# basic_specification["delayIt"] = [1]
# basic_names["delayIt"] = ['delayIt']

basic_specification["delayOthers"] = [1]
basic_names["delayOthers"] = ['delayOthers']

# basic_specification["swap"] = [1]
# basic_names["swap"] = ['swap']

mnl = pl.create_choice_model(data=forLogit,
                                        alt_id_col=custom_alt_id,
                                        obs_id_col=obs_id_column,
                                        choice_col=choice_column,
                                        specification=basic_specification,
                                        model_type="MNL",
                                        names=basic_names)

mnl.fit_mle(np.zeros(1))
print(mnl.get_statsmodels_summary())

custom_alt_id = 'mode_id'
obs_id_column = 'Custom_Choice_id'
choice_column = 'Choice'

basic_specification = OrderedDict()
basic_names = OrderedDict()

basic_specification["delayOthers"] = [1]
basic_names["delayOthers"] = ['delayOthers']

basic_specification["swap"] = [1]
basic_names["swap"] = ['swap']


mnl = pl.create_choice_model(data=forLogit,
                                        alt_id_col=custom_alt_id,
                                        obs_id_col=obs_id_column,
                                        choice_col=choice_column,
                                        specification=basic_specification,
                                        model_type="MNL",
                                        names=basic_names)

mnl.fit_mle(np.zeros(2))
print(mnl.get_statsmodels_summary())

custom_alt_id = 'mode_id'
obs_id_column = 'Custom_Choice_id'
choice_column = 'Choice'

basic_specification = OrderedDict()
basic_names = OrderedDict()

basic_specification["delayOthers"] = [1]
basic_names["delayOthers"] = ['delayOthers']

basic_specification["swap"] = [1]
basic_names["swap"] = ['swap']

basic_specification["delayIt"] = [1]
basic_names["delayIt"] = ['delayIt']

mnl = pl.create_choice_model(data=forLogit,
                                        alt_id_col=custom_alt_id,
                                        obs_id_col=obs_id_column,
                                        choice_col=choice_column,
                                        specification=basic_specification,
                                        model_type="MNL",
                                        names=basic_names)

mnl.fit_mle(np.zeros(3))
print(mnl.get_statsmodels_summary())


forLogit['test1'] = forLogit['delayIt']*100

custom_alt_id = 'mode_id'
obs_id_column = 'Custom_Choice_id'
choice_column = 'Choice'

basic_specification = OrderedDict()
basic_names = OrderedDict()

basic_specification["delayOthers"] = [1]
basic_names["delayOthers"] = ['delayOthers']

basic_specification["swap"] = [1]
basic_names["swap"] = ['swap']

basic_specification["test1"] = [1]
basic_names["test1"] = ['test1']

mnl = pl.create_choice_model(data=forLogit,
                                        alt_id_col=custom_alt_id,
                                        obs_id_col=obs_id_column,
                                        choice_col=choice_column,
                                        specification=basic_specification,
                                        model_type="MNL",
                                        names=basic_names)

mnl.fit_mle(np.zeros(3))
print(mnl.get_statsmodels_summary())

forLogittt = forLogit[['swap', 'delayOthers', 'delayIt']]
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(forLogittt)

# Apply PCA
pca = PCA()
principal_components = pca.fit_transform(scaled_data)

# Create a DataFrame with the principal components
principal_df = pd.DataFrame(data = principal_components, columns = ['PC1', 'PC2', 'PC3'])

forLogit['one'] = list(principal_df['PC1'])
forLogit['two'] = list(principal_df['PC2'])
forLogit['three'] = list(principal_df['PC3'])

custom_alt_id = 'mode_id'
obs_id_column = 'Custom_Choice_id'
choice_column = 'Choice'

basic_specification = OrderedDict()
basic_names = OrderedDict()

basic_specification["one"] = [1]
basic_names["one"] = ['one']

basic_specification["two"] = [1]
basic_names["two"] = ['two']

basic_specification["three"] = [1]
basic_names["three"] = ['three']

mnl = pl.create_choice_model(data=forLogit,
                                        alt_id_col=custom_alt_id,
                                        obs_id_col=obs_id_column,
                                        choice_col=choice_column,
                                        specification=basic_specification,
                                        model_type="MNL",
                                        names=basic_names)

mnl.fit_mle(np.zeros(3))
mnl.get_statsmodels_summary()


forLogit['test2'] = forLogit['two']*100

custom_alt_id = 'mode_id'
obs_id_column = 'Custom_Choice_id'
choice_column = 'Choice'

basic_specification = OrderedDict()
basic_names = OrderedDict()

basic_specification["one"] = [1]
basic_names["one"] = ['one']

basic_specification["test2"] = [1]
basic_names["test2"] = ['test2']

basic_specification["three"] = [1]
basic_names["three"] = ['three']

mnl = pl.create_choice_model(data=forLogit,
                                        alt_id_col=custom_alt_id,
                                        obs_id_col=obs_id_column,
                                        choice_col=choice_column,
                                        specification=basic_specification,
                                        model_type="MNL",
                                        names=basic_names)

mnl.fit_mle(np.zeros(3))
mnl.get_statsmodels_summary()



