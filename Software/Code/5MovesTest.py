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
import os
import pandas as pd
import numpy as np
import glob
import pickle

##used http://ataspinar.com/2018/04/04/machine-learning-with-signal-processing-techniques/ as reference



#transform psd_values
def get_psd_values(y_values,T,N,f_s):
    f_values, psd_values = welch(y_values,fs=f_s, nperseg = 64)
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

def extract_features_labels(df,labels,T,N,f_s,denom):
    percentile = 5
    list_of_features=[]
    list_of_labels = []
    for signal_num in range(0,len(df),N//2):
        features=[]
        list_of_labels.append(labels.iat[signal_num,0])
        for signal_comp in range(0,df.shape[1]):
            signal = df[signal_num:signal_num+N,signal_comp].reshape(-1)
            
            signal_min = np.nanpercentile(signal,percentile)
            signal_max = np.nanpercentile(signal,100-percentile)
            mph = signal_min + (signal_max - signal_min)/denom
            
            features += get_features(*get_psd_values(signal,T,N,f_s),mph)
            features += get_features(*get_fft_values(signal,T,N,f_s),mph)
            features += get_features(*get_autocorr_values(signal,T,N,f_s),mph)
            
        list_of_features.append(features)
    return np.array(list_of_features), np.array(list_of_labels)


def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

def main():
    url = "C:/Users/Jon/Documents/CG3002/Software/humandataset"
##    "/dance move"
##    urltest = "C:/Users/Jon/Documents/CG3002/Software/6th"

    all_files = getListOfFiles(url)

##    all_files_test = getListOfFiles(urltest)

    li= []

##    litest = []

    for filename in all_files:
        df = pd.read_csv(filename,index_col= None, header = None)
        li.append(df)

    dataset = pd.concat(li,axis = 0, ignore_index=True)

    dataset.dropna(inplace=True)

##    for filename in all_files_test:
##        df_test = pd.read_csv(filename,index_col= None, header = None)
##        litest.append(df_test)
##
##    dataset_test = pd.concat(litest,axis = 0, ignore_index=True)
##
##    dataset_test.dropna(inplace=True)
    
    data = pd.DataFrame({
            's1_acc_x': dataset.iloc[:,0],
            's1_acc_y': dataset.iloc[:,1],
            's1_acc_z': dataset.iloc[:,2],
            's1_gyro_x': dataset.iloc[:,3],
            's1_gyro_y': dataset.iloc[:,4],
            's1_gyro_z': dataset.iloc[:,5],
            's2_acc_x': dataset.iloc[:,6],
            's2_acc_y': dataset.iloc[:,7],
            's2_acc_z': dataset.iloc[:,8],
            's2_gyro_x': dataset.iloc[:,9],
            's2_gyro_y': dataset.iloc[:,10],
            's2_gyro_z': dataset.iloc[:,11],
            'activity': dataset.iloc[:,12]
         })

##    datatest = pd.DataFrame({
##            's1_acc_x': dataset_test.iloc[:,0],
##            's1_acc_y': dataset_test.iloc[:,1],
##            's1_acc_z': dataset_test.iloc[:,2],
##            's1_gyro_x': dataset_test.iloc[:,3],
##            's1_gyro_y': dataset_test.iloc[:,4],
##            's1_gyro_z': dataset_test.iloc[:,5],
##            's2_acc_x': dataset_test.iloc[:,6],
##            's2_acc_y': dataset_test.iloc[:,7],
##            's2_acc_z': dataset_test.iloc[:,8],
##            's2_gyro_x': dataset_test.iloc[:,9],
##            's2_gyro_y': dataset_test.iloc[:,10],
##            's2_gyro_z': dataset_test.iloc[:,11],
##            'activity': dataset_test.iloc[:,12]
##         })


    X = data[['s1_acc_x','s1_acc_y','s1_acc_z',
              's1_gyro_x','s1_gyro_y','s1_gyro_z',
              's2_acc_x','s2_acc_y','s2_acc_z',
              's2_gyro_x','s2_gyro_y','s2_gyro_z']]
    y = data[['activity']]

##    X_test = datatest[['s1_acc_x','s1_acc_y','s1_acc_z',
##              's1_gyro_x','s1_gyro_y','s1_gyro_z',
##              's2_acc_x','s2_acc_y','s2_acc_z',
##              's2_gyro_x','s2_gyro_y','s2_gyro_z']]
##    y_test = datatest[['activity']]

    N = 64
    f_s = 50
    t_n = 2.56
    T = t_n/N
    denom = 10

    X = preprocessing.normalize(X)
    X, y = extract_features_labels(X, y, T,N,f_s,denom)

##    X_test = preprocessing.normalize(X_test)
##    X_test, y_test = extract_features_labels(X_test, y_test, T,N,f_s,denom)
    X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.3)    

    clf= RandomForestClassifier(n_estimators = 130)
    clf.fit(X_train,y_train)

    print("Train score: {}".format(clf.score(X_train,y_train)))
    print("10-Fold CV Score: {}".format(np.mean(cross_val_score(clf,X_train,y_train,cv=10))))
    y_pred = clf.predict(X_test)

    print("Test score: {}".format(clf.score(X_test,y_test)))

    print(classification_report(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))

    filename = "RFmodel.pkl"
    pickle.dump(clf,open(filename,'wb'))

main()

