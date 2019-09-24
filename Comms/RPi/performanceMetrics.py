# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import division
import pandas as pd
import numpy as np
import sys



feature_columns = ['timestamp','action','goal','time_delta','correct','voltage','current','power']

def read_data(file_path):
    columns = ['timestamp','action','goal','time_delta','correct','voltage','current','power']
    data = pd.read_csv(file_path)
    data = data[columns]
    return data

def calcuateMeanTime(logData):
    timeDelay = logData["time_delta"]
    #print (np.mean(timeDelay))
    return np.mean(timeDelay)

def calculateMedianTime(logData):
    timeDelay = logData["time_delta"]
    #print np.median(timeDelay);
    return np.median(timeDelay)

def calculateMaxTime(logData):
    timeDelay = logData["time_delta"]
    #print np.max(timeDelay);
    return np.max(timeDelay)

def calculateMinTime(logData):
    timeDelay = logData["time_delta"]
    #print np.min(timeDelay);
    return np.min(timeDelay)

def percentageAccuracy(logData):
    correct = logData["correct"]
    correctIdentify = np.count_nonzero(correct == 1)
    falseIdentify = np.count_nonzero(correct == 0)
    percentAccuracy = correctIdentify/(correctIdentify+falseIdentify)*100
    #print percentAccuracy
    return percentAccuracy

def calculateMeanPower(logData):
    power = logData["power"]
    return np.mean(power)

def calculateMeanCurrent(logData):
    current = logData["current"]
    return np.mean(current)

def calculateMeanVoltage(logData):
    voltage = logData["voltage"]
    return np.mean(voltage)
    
def main():
#    file_path='log.csv'
    file_path=sys.argv[1]
    dataset = read_data(file_path)
    data = dataset[feature_columns]
    logData = pd.DataFrame(data, columns=feature_columns)
    
    meanTime = calcuateMeanTime(logData)
    print('Mean Time:               '+ str(meanTime))
    
    medianTime = calculateMedianTime(logData)
    print('Median Time:             '+ str(medianTime))
    
    maxDelay = calculateMaxTime(logData)
    print('MaxDelay:           '+ str(maxDelay))
    
    minDelay = calculateMinTime(logData)
    print('MinDelay:           '+ str(minDelay))
    
    percentAccuracy=percentageAccuracy(logData)
    print('PercentAccuracy:    '+ str(percentAccuracy))
    
    meanVoltage = calculateMeanVoltage(logData)
    print('meanVoltage:    '+ str(meanVoltage))
    
    meanCurrent = calculateMeanCurrent(logData)
    print('meanCurrent:    '+ str(meanCurrent))
    
    meanPower = calculateMeanPower(logData)
    print('meanPower:    '+ str(meanPower))

if __name__ == '__main__':
    if len(sys.argv) == 2:
        main()
    else:
        print("Wrong argument format")
