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

# load a single file as a numpy array
def load_file(filepath):
    dataframe = pd.read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = np.dstack(loaded)
    return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
    # body acceleration
    filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
    # body gyroscope
    filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/y_'+group+'.txt')
    return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
    # load all train
    trainX, trainy = load_dataset_group('train', prefix + '../datasets/UCI HAR Dataset/UCI HAR Dataset/')
    # load all test
    testX, testy = load_dataset_group('test', prefix + '../datasets/UCI HAR Dataset/UCI HAR Dataset/')
    # zero-offset class values
    trainy = trainy - 1
    testy = testy - 1
    # one hot encode y
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX, trainy, testX, testy

# x_train, y_train, x_test, y_test = load_dataset()

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
    EPOCHS = 100
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
    
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
    # Confusion matrix
    y_pred = saved_model.predict_classes(x_test, batch_size, verbose=0)
    cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    labels=['WALKING','WALKING_UPSTAIRS','WALKING_DOWNSTAIRS','SITTING','STANDING','LAYING']
    cm_analysis(cm, labels, ymap=None, figsize=(10,10))
    print(cm)
    
    # Evaluate model
    _, train_acc = saved_model.evaluate(x_train, y_train, verbose=0)
    _, test_acc = saved_model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc*100.0, test_acc*100.0))
    return test_acc

# 6. Evaluate the model

# create_model(x_train, y_train).summary()

# Summarize scores
def summarize_results(scores):
    print(scores)
    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# Main function
def cnn_main():
	x_train, y_train, x_test, y_test = load_dataset()

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
