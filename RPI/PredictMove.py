from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn import preprocessing
from sklearn.externals import joblib
from peak_detection import *
from scipy.signal import welch
from scipy.fftpack import fft
import pandas as pd
import numpy as np
import glob
import pickle

#constants used to record signals
N = 128
f_s = 50
t_n = 2.56
T = t_n/N
denom = 10

#transform psd_values
def get_psd_values(y_values,T,N,f_s):
    f_values, psd_values = welch(y_values,fs=f_s, nperseg = 128)
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

def extract_features_labels_predict(df,T,N,f_s,denom):
    percentile = 5
    list_of_features=[]
    features=[]
    for signal_comp in range(0,df.shape[1]):
        signal = df[:,signal_comp]

        signal_min = np.nanpercentile(signal,percentile)
        signal_max = np.nanpercentile(signal,100-percentile)
        mph = signal_min + (signal_max - signal_min)/denom
        
        features += get_features(*get_psd_values(signal,T,N,f_s),mph)
        features += get_features(*get_fft_values(signal,T,N,f_s),mph)
        features += get_features(*get_autocorr_values(signal,T,N,f_s),mph)
        
    list_of_features.append(features)
    return np.array(list_of_features)
 
def predictMain(X):
    
    X = preprocessing.normalize(X)
    X = extract_features_labels_predict(X,T,N,f_s,denom)

    clf = joblib.load("RFmodel.pkl")
    return (int(clf.predict(X)[0]))
