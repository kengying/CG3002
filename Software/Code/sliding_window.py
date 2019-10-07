# 1. Manage imports

import os
import numpy
import pandas as pd
import collections

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

print(tf.version.VERSION)
print(tf.keras.__version__)


# 2. Load Dataset

df = pd.read_csv('../datasets/Activity Recognition from Single Chest-Mounted Accelerometer/Activity Recognition from Single Chest-Mounted Accelerometer/1.csv',
                names=['index', 'acc_x', 'acc_y', 'acc_z', 'activity'], usecols=range(1,5))

df_valid = df[df['activity'].isin(range(1,8))]

print(df_valid)


# 3. Features extraction and data segmentation

def sliding_window(df, width, overlap):
    windows = []
    current_index = 0
    if overlap < 0 or overlap >= 1:
        print("Invalid overlap value.")
        return None
    while True:
        next_index = current_index + width
        if next_index >= len(df):
#             print("End of window sliding.")
            break
        windows.append(df[current_index:next_index])
        current_index += max(int((1-overlap)*width), 1)
    return windows


data_cols = ['acc_x', 'acc_y', 'acc_z']
df_valid[data_cols] = (df_valid[data_cols]-df_valid[data_cols].min())/(df_valid[data_cols].max()-df_valid[data_cols].min())


# Windows generation
sliding_windows = []
for activity, df_group in df_valid.groupby('activity'):
    windows = sliding_window(df_group, 30, 0.5)
    sliding_windows.append(windows)


# # Feature extraction
# feature_extraction = []
# for activity, df_group in df_valid.groupby('activity'):
#     windows = sliding_window(df_group, 4, 0.5)
    
#     # More features here    
#     grp = collections.OrderedDict([
#         ('mean_x', []),
#         ('mean_y', []),
#         ('mean_z', []),
#         ('activity', activity)
#     ])

#     for window in windows:
#         means = window[['acc_x', 'acc_y', 'acc_z']].mean()
#         grp['mean_x'].append(means['acc_x'])
#         grp['mean_y'].append(means['acc_y'])
#         grp['mean_z'].append(means['acc_z'])
        
#     feature_extraction.append(pd.DataFrame(grp))

# print(feature_extraction)

# Just for temporary output testing
# df_group = df_valid.groupby('activity').get_group(1)
# normalized_df_group = (df_group-df_group.min())/(df_group.max()-df_group.min())
# # print(df_group)

# windows = sliding_window(normalized_df_group, 50, 0.5)
# print(windows[0])