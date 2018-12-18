#!/usr/local/bin/python3
# Code Framework for an End-to-End ML Project

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


RANDOM = 42
np.random.seed(RANDOM)


data = pd.DataFrame()

'''
==================================================
SPLIT DATA (RANDOM OR STRATIFIED) INTO TRAINING AND TEST SETS
'''

# 1) Random split
# from sklearn.model_selection import train_test_split
# train_set, test_set = train_test_split(data, test_size=0.2,
#   random_state=RANDOM)

# 2) Stratified split
# Use one metric to get stratified split
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data['strat_metric']):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

# Drop the column that was used to stratify the data if manufactured
for set_ in (strat_train_set, strat_test_set):
    set_.drop('strat_metric', axis=1, inplace=True)

'''
==================================================
VISUALIZE THE DATA [TO COME]
'''

# Use training set only (with labels) for visualizations
data = strat_train_set.copy()
print(data.describe())

# Categorical Features
for col in data.dtypes[data.dtypes == 'object'].index:
    sns.countplot(data=data, y=col)
    plt.show()

# Numeric Features
plt.figure(figsize=(7, 6))
sns.boxplot(data=data, x='label', y='cat_to_split_on')
plt.show()

plt.figure(figsize=(9, 8))
correlations = data.corr()
mask = np.zeros_like(correlations, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(correlations * 100, annot=True, fmt='.0f', mask=mask, cbar=False)
plt.show()


'''
==================================================
PREPARE DATA FOR PREPROCESSING AND ML ALGORITHMS
'''

# Move labels from training set to own df
data = strat_train_set.drop('label', axis=1)
data_labels = strat_train_set['label'].copy()
