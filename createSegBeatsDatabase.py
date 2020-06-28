# -*- coding: utf-8 -*-
"""
"""
import os
import glob
import pywt
import numpy as np
import pandas as pd
import pickle
from scipy import signal
from scipy.io import loadmat
#import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
#from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values
#from hrvanalysis import get_time_domain_features, get_frequency_domain_features, get_geometrical_features, get_poincare_plot_features

folder = r'D:\PhysioNet'
outFolder = os.path.join('C:\PhysioNet', 'results')
dataRecords = glob.glob(folder + '\Training_WFDB\*.mat')

colNames = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
labelTypes = ['Normal', 'AF', 'I-AVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']

class SegBeatsDatabase:
  
  def __init__(self, patient='A0001', data=None, header_data=None, numPrincComp=25, percToDrop=0.20, doPlots=True):
    self.patient = patient
    if (data is not None) and (header_data is not None):
      self.patientDF, self.labelsOneHot = self.convertDataFormat(data, header_data)
    else:
      self.patientDF, self.labelsOneHot = self.genDataDFandLabels()
    self.doPlots = doPlots
    self.sampleRate = 500.0 # Hz
    self.maxHeartRate = 180.0
    self.windowSize = int(self.sampleRate / (self.maxHeartRate/60.0))
    self.numPrincComp = numPrincComp
    self.percToDrop = percToDrop
    self.NUM_PTS_IN_BEAT = 250
    self.pca = None

  ### Prepare dataset
  def genDataDFandLabels(self):
    d = os.path.join(folder, 'Training_WFDB', self.patient+'.mat')
    data = loadmat(d)
    patientDF = pd.DataFrame(np.transpose(data['val']), columns=colNames)
    # Read associated header file to get labels
    with open(os.path.join(folder, 'Training_WFDB', self.patient + '.hea')) as f:
      for line in f:
        if line.split()[0] == '#Dx:':
          labels = [lab for lab in line.split()[1].split(',')]
          labelsOneHot = self.encodeHot(labels)
    return patientDF, labelsOneHot

  def convertDataFormat(self, data, header_data):
    patientDF = pd.DataFrame(np.transpose(data), columns=colNames)
    for iline in header_data:
      if iline.startswith('#Dx:'):
        labels = [lab for lab in iline.split()[1].split(',')]
        labelsOneHot = self.encodeHot(labels)
    return patientDF, labelsOneHot

  def encodeHot(self, labels):
    labelOneHot = [0] * len(labelTypes)
    for i, lab in enumerate(labelTypes):
      if lab in labels:
        labelOneHot[i] = 1
    return labelOneHot

  @staticmethod
  def loadDataset(self):
    with open('PhysioNetData.pkl') as f:
      dataDict = pickle.load(f)
    return dataDict
  ########################################
    
  def applyButterworthFilter(self, data):
    ''' Create and applies a Butterworth filter to the data. Returns
        the filtered data as a list'''
    fn = 0.5 * self.sampleRate   # Nyquist frequency
    # According to Samarin, use a bandpass filter with passband of 1 Hz to 20 Hz
    f_low = 1.0/fn
    f_high = 20.0/fn
    # Create Butterworth filter
    b, a = signal.butter(3, [f_low, f_high], 'bandpass')
    w, h = signal.freqz(b, a, worN=2000)
    # Pass the data through the filter and save to outputDF
    filtData = signal.filtfilt(b, a, data)
    # if self.doPlots:
    #   plt.figure()
    #   plt.plot(range(len(filtData)), filtData)
    #   #plt.xlim([0,5*self.sampleRate])
    #   plt.title('Filtered Data')
    return filtData

  def clipOutliersFromFilteredData(self, data):
    Q1 = pd.DataFrame(data).quantile(0.25)[0]
    Q3 = pd.DataFrame(data).quantile(0.75)[0]
    IQR = Q3 - Q1
    data[data < Q1 - 1.5 * IQR] = 0.0
    data[data > Q3 + 1.5 * IQR] = 0.0
    # if self.doPlots:
    #   plt.figure()
    #   plt.plot(range(len(data)), data)
    #   # plt.xlim([0,5*self.sampleRate])
    #   plt.title('Clipped Data')
    return data

  @staticmethod
  def createWavelet(waveletType):
    ''' Creates and return the wavelet object of the type specified '''
    w = pywt.Wavelet(waveletType)
    return w

  def applyDwtAndSquareSignal(self, data, wavelet):
    ''' Applies the DWT to the data and returns squared coefficients '''
    # Apply the discrete wavelet transform
    mode = pywt.Modes.smooth
    (coeff, d) = pywt.dwt(data, wavelet, mode)
    coeff_list = [coeff, None]
    transformedData = pywt.waverec(coeff_list, wavelet)
    # Square the transformed data to accentuate the peaks
    squaredTransformedData = np.power(transformedData, 2)
    # if self.doPlots:
    #   plt.figure()
    #   plt.plot(range(len(squaredTransformedData)), squaredTransformedData)
    #   plt.title('Squared and Transformed Data: '+self.patient)
    #   plt.show()
    return squaredTransformedData
  
  @staticmethod
  def getGradient(data):
    ''' Calculates and returns a list of the gradient at each data point '''
    gradData = np.gradient(data)
    return gradData
  
  def findPeakInds(self, data):
    ''' Finds the R peaks in the squared DWT transformed data and returns
        the corresponding data list indices '''
    # Use the gradient of the data to localize the R peaks
    diffs = np.diff(data)
    meanDiff = np.mean(diffs)
    stdDiff = np.std(diffs)
    threshold = meanDiff + 2.0 * stdDiff
    peaks = data
    peaks[data < threshold] = 0.0
    peakInds = []
    # Find the indices that correspond to the R peaks
    for i in range(0,len(peaks), self.windowSize):
      window = peaks[i:i+self.windowSize]
      if sum(window) > 0.0:
        maxInd, = list(np.where(window == np.max(window)))
        peakInds.append(i + maxInd[0])
    # Remove false peaks
    indsToRemove = []
    for i in range(1,len(peakInds)):
      if peakInds[i] - peakInds[i-1] < self.windowSize:
        indsToRemove.append(peakInds[i]) if peaks[peakInds[i]] <= peaks[peakInds[i-1]] else indsToRemove.append(peakInds[i-1])
    peakInds = [x for x in peakInds if x not in indsToRemove]
    time = [(x/self.sampleRate) for x in range(len(data))]
    rrIntervals = self.calcRRIntervals(peakInds, time)
    # if self.doPlots:
    #   plt.figure()
    #   plt.plot(range(len(peaks)), peaks)
    #   plt.scatter(peakInds, peaks[peakInds], c='r')
    #   plt.title('Identified Peaks: '+self.patient)
    #   plt.savefig(os.path.join(outFolder, self.patient+'_peaks.png'))
    #   plt.show()
    return peakInds
  
  def calcRRIntervals(self, peakInds, time):
    rrIntervals = []
    for i in range(1,len(peakInds)):
      rrIntervals.append(time[peakInds[i]] - time[peakInds[i-1]])
    meanRR = np.mean(rrIntervals)
    stdRR = np.std(rrIntervals)
    rrIntervals = [x for x in rrIntervals if (x >= meanRR - 2.0 * stdRR >= x)]
    return rrIntervals
  
  def resample(self, x, kind='linear'):
    n = self.NUM_PTS_IN_BEAT
    f = interp1d(np.linspace(0, 1, x.size), x, kind)
    return f(np.linspace(0, 1, n))
  
  @staticmethod
  def calcNorm(x):
    norm = np.sqrt(np.square(x).sum(axis=1))
    return norm
  
  @staticmethod
  def calcMedian(dataDF):
    medianBeat = dataDF.median()
    return medianBeat
  
  def removeOutliersFromIndividual(self, dataDF):
    medianBeat = self.calcMedian(dataDF)
    dataMinusMedianDF = dataDF.sub(medianBeat)
    dataMinusMedianDF['norm'] = np.linalg.norm(dataMinusMedianDF.values,axis=1)
    dataMinusMedianDF.sort_values(by=['norm'], inplace=True)
    numDropRows = int(np.round(self.percToDrop * len(dataMinusMedianDF)))
    dataMinusMedianDF = dataMinusMedianDF[:-numDropRows]
    return dataMinusMedianDF
  
  def segmentBeats(self, filtData, peakInds):
    ''' Separates the beats into individual windows and saves the beat data
        to a csv file. The first and last beats are removed to account for
        the filter warm-up period. '''
    # Determine average distance between peaks
    meanDist = int(np.mean(np.diff(peakInds)))
    thirdDist = int(np.round(meanDist/3.0))
    # Initialize data dictionary
    beatsDF = pd.DataFrame(index=[])
    # Loop over the peak indices and segment the beats
    for k, pk in enumerate(peakInds):
      startInd = np.max([0, pk - thirdDist])
      stopInd = np.min([startInd+meanDist, len(filtData)-1])
      beat = filtData[startInd:stopInd]
      # Throw out the first and last peaks
      if k == 0 or k == len(peakInds)-1:
        continue
      else:
        # Resample to 250 points
        resampledBeat = self.resample(beat, kind='linear')
        beatsDF = beatsDF.append(pd.Series(resampledBeat, index=range(self.NUM_PTS_IN_BEAT), name=str(k)))
        beatsDF['ID'] = self.patient
        for i, label in enumerate(labelTypes):
          beatsDF[label] = self.labelsOneHot[i]
    #beatsDF = pd.concat([beatsDF, pd.DataFrame([self.labelsOneHot], columns=labelTypes)], axis=1)
    return beatsDF
  
  def calcPCA(self, dataDF):
    x = dataDF.values
    self.pca = PCA(n_components=self.numPrincComp)
    principalComponents = self.pca.fit_transform(x)
    pcaDF = pd.DataFrame(data = principalComponents)
    return pcaDF
  
  # def calcHrvFeatures(self, rrIntervals):
  #   # From https://github.com/Aura-healthcare/hrvanalysis
  #   # Calculate NN intervals
  #   rr_intervals_without_outliers = remove_outliers(rr_intervals=rrIntervals, low_rri=300, high_rri=2000, verbose=False)
  #   interpolated_rr_intervals = interpolate_nan_values(rr_intervals=rr_intervals_without_outliers, interpolation_method="linear")
  #   nn_intervals_list = remove_ectopic_beats(rr_intervals=interpolated_rr_intervals, method="malik")
  #   interpolated_nn_intervals = interpolate_nan_values(rr_intervals=nn_intervals_list)
  #   # Calculate time-domain features
  #   time_domain_features = get_time_domain_features(interpolated_nn_intervals)
  #   # Calculate frequency-domain features
  #   frequency_domain_features = get_frequency_domain_features(interpolated_nn_intervals, sampling_frequency=128)
  #   # Calculate geometrical features
  #   geometrical_features = get_geometrical_features(interpolated_nn_intervals)
  #   # Calculate Poincare features
  #   poincare_features = get_poincare_plot_features(interpolated_nn_intervals)
  #   return time_domain_features, frequency_domain_features, geometrical_features, poincare_features

  # def plotDatasets(self):
  #   plt.figure()
  #   plt.suptitle('CardioKey Data')
  #   for i in range(len(dataRecords)):
  #     df = pd.read_csv(dataRecords[i], skiprows=[1])
  #     plt.subplot(len(dataRecords),1,i+1)
  #     plt.plot(df['X'], df['CH1'])
  #     plt.xlabel('Time (sec)')
  #     plt.ylabel('Volts')
  #   plt.savefig(os.path.join(outFolder,'dataPlots.png'))
  #   plt.show()
    
  # def plotBeats(self, beatsDF, numPlots=3):
  #   ''' Plots the segmented beats individually '''
  #   for i in range(numPlots):
  #     plt.figure()
  #     beat = beatsDF.iloc[i]
  #     time = np.arange(len(beat)) * (1.0/float(self.sampleRate))
  #     plt.plot(time, beat)
  #     plt.savefig(os.path.join(outFolder, 'beat_'+str(i)+'.png'))
    return
    
############################################################################
if __name__ == "__main__":

  for i, d in enumerate(dataRecords):

    # Get patient ID
    patient = os.path.basename(d)[:-4]
    print(patient)

    # Create SegBeatsDatabase object
    p = SegBeatsDatabase(patient=patient, doPlots=False)

    # Filter the data with a Butterworth filter
    filteredData = p.applyButterworthFilter(p.patientDF['II'])

    # Clip outliers
    filteredData = p.clipOutliersFromFilteredData(filteredData)

    # Create wavelet
    w = p.createWavelet('sym4')

    # Apply the discrete wavelet transform and square the result
    reconData = p.applyDwtAndSquareSignal(filteredData, w)

    # Take the gradient of the wavelet reconstruction
    gradData = p.getGradient(reconData)

    # Find the peaks in the gradient data
    peakInds = p.findPeakInds(gradData)

    # Segment the beats
    beatsDF = p.segmentBeats(filteredData, peakInds)

    # Create the full dataframe
    if i == 0:
      PhysioNetDF = beatsDF
    else:
      PhysioNetDF = PhysioNetDF.append(beatsDF, ignore_index=True)

  # Write PhysioNetDF to file
  PhysioNetDF.to_csv(os.path.join(folder, 'Training_WFDB', 'PhysioNetDF.csv'), index=False)
