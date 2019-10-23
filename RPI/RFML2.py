##from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn import preprocessing
from peak_detection import *
from scipy.signal import welch
from scipy.fftpack import fft
import pandas as pd
import numpy as np
import glob

##used http://ataspinar.com/2018/04/04/machine-learning-with-signal-processing-techniques/ as reference

#read signal data files
def read_signals(pathname):
    with open(pathname,'r') as pn:
        data = pn.read().splitlines()
        data = map(lambda x: x.rstrip().lstrip().split(), data)
        data = [list(map(float,line)) for line in data]
        data = np.array(data, dtype = np.float32)
    return data

#read label files
def read_labels(pathname):
    with open(pathname, 'r') as pn:
        act = pn.read().splitlines()
        act = list(map(int,act))
    return np.array(act)

#transform psd_values
def get_psd_values(y_values,T,N,f_s):
    f_values, psd_values = welch(y_values,fs=f_s,nperseg = 128)
    return f_values,psd_values

#transform fft
def get_fft_values(y_values, T,N,f_s):
    f_values = np.linspace(0.0,1.0/(2.0*T),N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values

#transform correlation
def autocorr(x):
    result = np.correlate(x,x,mode='full')
    return result[len(result)//2:]

def get_autocorr_values(y_values, T, N, f_s):
    autocorr_values = autocorr(y_values)
    X_values = np.array([T*jj for jj in range(0,N)])
    return X_values, autocorr_values

def get_first_n_peaks(x,y,num_peaks = 5):
    xt,yt=list(x),list(y)
    if len(xt) >=num_peaks:
        return xt[:num_peaks],yt[:num_peaks]
    else:
        missing_num_peaks = num_peaks-len(xt)
        return xt + [0]*missing_num_peaks, yt+[0]*missing_num_peaks

def get_features(x_values,y_values, mph):
    indicies_peaks = detect_peaks(y_values,mph=mph)
    peaks_x,peaks_y = get_first_n_peaks(x_values[indicies_peaks],y_values[indicies_peaks])
    return peaks_x+peaks_y

def extract_features_labels(df,labels, T,N,f_s,denom):
    percentile = 5
    list_of_features=[]
    list_of_labels = []
    for signal_num in range(0,len(df)):
        features=[]
        list_of_labels.append(labels[signal_num])
        for signal_comp in range(0,df.shape[2]):
            signal = df[signal_num,:,signal_comp]

            signal_min = np.nanpercentile(signal,percentile)
            signal_max = np.nanpercentile(signal,100-percentile)
            mph = signal_min + (signal_max - signal_min)/denom

            features += get_features(*get_psd_values(signal,T,N,f_s),mph)
            features += get_features(*get_fft_values(signal,T,N,f_s),mph)
            features += get_features(*get_autocorr_values(signal,T,N,f_s),mph)
        list_of_features.append(features)
    return np.array(list_of_features), np.array(list_of_labels)

def RFMLmain():
    url= "/home/pi/Desktop/UCI HAR Dataset/train/Inertial Signals/"
    urltest = "/home/pi/Desktop/UCI HAR Dataset/test/Inertial Signals/"

    url_y_train = "/home/pi/Desktop/UCI HAR Dataset/train/y_train.txt"
    url_y_test = "/home/pi/Desktop/UCI HAR Dataset/test/y_test.txt"

    files_train = ['body_acc_x_train.txt', 'body_acc_y_train.txt', 'body_acc_z_train.txt',
                   'body_gyro_x_train.txt', 'body_gyro_y_train.txt', 'body_gyro_z_train.txt',
                   'total_acc_x_train.txt', 'total_acc_y_train.txt', 'total_acc_z_train.txt']

    files_test =  ['body_acc_x_test.txt', 'body_acc_y_test.txt', 'body_acc_z_test.txt',
                   'body_gyro_x_test.txt', 'body_gyro_y_test.txt', 'body_gyro_z_test.txt',
                   'total_acc_x_test.txt', 'total_acc_y_test.txt', 'total_acc_z_test.txt']

    train_signals, test_signals = [],[]

    for input_file in files_train:
        signal = read_signals(url + input_file)
        train_signals.append(signal)
    train_signals = np.transpose(np.array(train_signals),(1,2,0))

    for input_file in files_test:
        signal = read_signals(urltest + input_file)
        test_signals.append(signal)
    test_signals = np.transpose(np.array(test_signals),(1,2,0))

    label_train = read_labels(url_y_train)
    label_test = read_labels(url_y_test)

    #constants used to record signals
    N = 128
    f_s = 50
    t_n = 2.56
    T = t_n/N

    denom = 10
    X_train, y_train = extract_features_labels(train_signals, label_train, T,N,f_s,denom)
    X_test, y_test = extract_features_labels(test_signals, label_test, T,N,f_s,denom)

    ##X_train = preprocessing.normalize(X_train)
    ##X_test = preprocessing.normalize(X_test)
    ##
    ##X_train = preprocessing.scale(X_train)
    ##X_test = preprocessing.scale(X_test)

    clf= RandomForestClassifier(n_estimators = 100)
    clf.fit(X_train,y_train)

    print("Train score: {}".format(clf.score(X_train,y_train)))
    print("10-Fold CV Score: {}".format(np.mean(cross_val_score(clf,X_train,y_train,cv=10))))
    y_pred = clf.predict(X_test)

    print("Test score: {}".format(clf.score(X_test,y_test)))

    print(classification_report(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))

