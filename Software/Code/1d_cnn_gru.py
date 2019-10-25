# 1. Manage imports

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import h5py

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

# print(tf.version.VERSION)
# print(tf.keras.__version__)

# tf.logging.set_verbosity(tf.logging.ERROR)
# %matplotlib inline

WINDOW_SIZE = 128
OVERLAP = 0.5

def sliding_window(data, width=WINDOW_SIZE, overlap=OVERLAP):
    windows = list()
    current_index = 0
    if overlap < 0 or overlap >= 1:
        print("Invalid overlap value.")
        return None
    while True:
        next_index = current_index + width
        if next_index >= len(data):
            break
        windows.append(data[current_index:next_index])
        current_index += max(int((1-overlap)*width), 1)
    return windows

import glob

def generate_train_test():
    path = '../datasets/here'

    labels=['bunny','cowboy','handmotor','rocket','tapshoulder']
    classes={
        'bunny':1,
        'cowboy':2,
        'handmotor':3,
        'rocket':4,
        'tapshoulder':5
    }

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for label in labels:
        train_dir = path + "/train/" + label
        train_files = glob.glob(train_dir + "/*.csv")
        for filename in train_files:
            data = pd.read_csv(filename, header=None, usecols=range(0,12)).values
            windows = sliding_window(data)
            x_train = x_train + windows
            y_train = y_train + [classes[label]]*len(windows)

        test_dir = path + "/test/" + label
        test_files = glob.glob(test_dir + "/*.csv")
        for filename in test_files:
            data = pd.read_csv(filename, header=None, usecols=range(0,12)).values
            windows = sliding_window(data)
            x_test = x_test + windows
            y_test = y_test + [classes[label]]*len(windows)

    x_train = np.array(x_train)
    y_train = to_categorical(np.array(y_train)-1)
    x_test = np.array(x_test)
    y_test = to_categorical(np.array(y_test)-1)
#     print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, stratify=y_train, test_size=0.33, shuffle= True)
    # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    
    return x_train, y_train, x_test, y_test

# Helper functions

def cm_analysis(cm, labels, ymap=None, figsize=(10,10)):
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap="Blues")
    #plt.savefig(filename)
    plt.show()

from tensorflow.keras.layers import BatchNormalization, LSTM, Activation, Dense, Dropout, GRU,Conv1D,MaxPooling1D

def create_model(x_train, y_train):
    # Define model
    n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
    model = Sequential()
    
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(MaxPooling1D(pool_size=4))
    model.add(BatchNormalization())
    
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())

    model.add(GRU(
        units=64,
        activation='tanh',
        recurrent_activation='sigmoid',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        return_sequences=True,  # because the next layer is also GRU
        dropout=0.0, #0.303
        recurrent_dropout=0.0, #0.458 
        ))   
    model.add(GRU(
        units=64,
        activation='tanh',
        recurrent_activation='sigmoid',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        return_sequences=False,  # because the next layer is dense 
    #     dropout=0.2, # 0.196
         recurrent_dropout=0.2  # 0.073
        ))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_outputs, activation='softmax'))
    
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def evaluate_model(x_train, y_train, x_test, y_test, cnn_model):
    EPOCHS = 50
    BATCH_SIZE = 32
    
    epochs, batch_size = EPOCHS, BATCH_SIZE
    model = cnn_model
    
    x_train_2, x_valid, y_train_2, y_valid = train_test_split(x_train, y_train, stratify=y_train, 
                                                              test_size=0.33, shuffle= True)
    
    # Configure model callbacks including early stopping routine
    PATIENCE_NUM = 10

    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=PATIENCE_NUM)
    model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=0, save_best_only=True)

    # Fit network
#     model.fit(x_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=1)
    history = model.fit(x_train_2, y_train_2, validation_data=(x_valid, y_valid), epochs=epochs, batch_size=batch_size,
                        verbose=0, callbacks=[early_stopping, model_checkpoint])
    saved_model = load_model('best_model.h5')
    
#     # summarize history for accuracy
#     plt.plot(history.history['acc'])
#     plt.plot(history.history['val_acc'])
#     plt.title('model accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'validation'], loc='upper left')
#     plt.show()
#     # summarize history for loss
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('model loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'validation'], loc='upper left')
#     plt.show()
    
#     # Confusion matrix
#     y_pred = saved_model.predict_classes(x_test, batch_size, verbose=0)
#     cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
#     labels=['bunny','cowboy','handmotor','rocket','tapshoulder']
#     cm_analysis(cm, labels, ymap=None, figsize=(10,10))
#     print(cm)
    
    # Evaluate model
    _, train_acc = saved_model.evaluate(x_train, y_train, verbose=0)
    _, test_acc = saved_model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc*100.0, test_acc*100.0))
    return test_acc

# 6. Evaluate the model

# Summarize scores
def summarize_results(scores):
    print(scores)
    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# Main function
def cnn_main():
    x_train, y_train, x_test, y_test = generate_train_test()
    
#     create_model(x_train, y_train).summary()

    REPEATS_NUM = 1

    # Repeat experiment
    scores = list()
    for r in range(REPEATS_NUM):
        cnn_model = create_model(x_train, y_train)
        score = evaluate_model(x_train, y_train, x_test, y_test, cnn_model)
        score = score * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)

    # Summarize results
    summarize_results(scores)

def get_cnn_model():
    x_train, y_train, x_test, y_test = generate_train_test()
    cnn_model = create_model(x_train, y_train)
    
    EPOCHS = 50
    BATCH_SIZE = 32
    
    epochs, batch_size = EPOCHS, BATCH_SIZE
    model = cnn_model
    
    x_train_2, x_valid, y_train_2, y_valid = train_test_split(x_train, y_train, stratify=y_train, 
                                                              test_size=0.33, shuffle= True)
    
    # Configure model callbacks including early stopping routine
    PATIENCE_NUM = 10

    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=PATIENCE_NUM)
    model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=0, save_best_only=True)

    # Fit network
#     model.fit(x_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=1)
    history = model.fit(x_train_2, y_train_2, validation_data=(x_valid, y_valid), epochs=epochs, batch_size=batch_size,
                        verbose=0, callbacks=[early_stopping, model_checkpoint])
    saved_model = load_model('best_model.h5')
    
    return saved_model

# cnn_main()