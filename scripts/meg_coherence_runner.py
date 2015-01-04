# -*- coding: utf-8; python-indent: 2; -*-
# This script extracts P-values and R^2 values over each frequency band and determines model fits

# Global Vars
# =======

DEV = True #if True: analyze the dev set; if False: analyze the test set ;; DEV is defined on a sentence level using a stepsize of N ;; TEST is the complement of DEV
devsizerecip = 3 # the reciprocal of the dev size, so devsizerecip = 3 means the dev set is 1/3 and the test set is 2/3

VERBOSE = False #Provide some extra output, mainly for development purposes
FUDGE = False # this factor is used to enable and mark dubious code, which can be cleaned up later; purely for development

# Imports
# =======

import time
import cPickle as pickle
import logging as L
L.basicConfig(level=L.ERROR) # INFO)
import time
import numpy
import scipy.stats
import pandas as pd
import os
#import pylab
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
import scipy
import sklearn.linear_model
from sklearn.feature_selection import chi2
import sys
import re
#import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.outliers_influence

import mne
from mne.connectivity import spectral_connectivity, seed_target_indices

# brian's prototype routines
from protoMEEGutils import *
import protoSpectralWinFFTMapper as specfft

# Definitions
# ===========
        
# adjust R2 down for the artificial inflation you get by increasing the number of explanatory features
def adjustR2(R2, numFeatures, numSamples):
  #1/0
  #return R2
  return R2-(1-R2)*(float(numFeatures)/(numSamples-numFeatures-1))

# normalise (z-scale) the scale of variables (for the explanatory ones, so the magnitude of beta values are comparably interpretable)
def mynormalise(A):
  A = scipy.stats.zscore(A)
  A[numpy.isnan(A)] = 0
  return A

# Preprocessing
# =============

#choose a file - I found participant V to be pretty good, and 0.01 to 50Hz filter is pretty conservative #
(megFileTag1, megFile1) = ('V_TSSS_0.01-50Hz_@125', '../MEG_data/v_hod_allRuns_tsss_audiobookPrepro_stPad1_lp50_resamp125_frac10ICAed.set')#_hp0.010000.set')
(megFileTag2, megFile2) = ('A_TSSS_0.01-50Hz_@125', '../MEG_data/aud_hofd_a_allRuns_tsss_audiobookPrepro_stPad1_lp50_resamp125_frac10ICAed_hp0.010000.set')
(megFileTag3, megFile3) = ('C_TSSS_0.01-50Hz_@125', '../MEG_data/aud_hofd_c_allRuns_tsss_audiobookPrepro_stPad1_lp50_resamp125_frac10ICAed_hp0.010000.set')

## put your own properties in here, as a .tab file of similar format (tab delimited, and field names in a first comment line - should be easy to do in excel...)
#to get the V5.tab:
#  python ../scripts/buildtab.py hod_JoeTimes_LoadsaFeaturesV3.tab hod.wsj02to21-comparativized-gcg15-1671-4sm.fullberk.parsed.gcgbadwords > hod_JoeTimes_LoadsaFeaturesV4.tab
#  python ../scripts/addsentid.py hod_JoeTimes_LoadsaFeaturesV4.tab > hod_JoeTimes_LoadsaFeaturesV5.tab
tokenPropsFile = '../MEG_data/hod_JoeTimes_LoadsaFeaturesV5.tab'

# LOAD WORD PROPS
# change dtype to suit the files in your .tab file #
tokenProps = scipy.genfromtxt(tokenPropsFile,
                              delimiter='\t',names=True,
                              dtype="i4,f4,f4,S50,S50,i2,i2,i2,S10,f4,f4,f4,f4,f4,f4,f4,f4,f4,f4,f4,i1,>i4")
# ... and temporarily save as cpickle archive to satisfy the way I programmed the convenience function loadBookMEGWithAudio (it expects to find the same info in a C-pickle file, and so doesn't need to know about number and type of fields)
tokenPropsPickle = tokenPropsFile+'.cpk'
pickle.dump(tokenProps, open(tokenPropsPickle, 'wb'))

triggersOfInterest=['s%d' % i for i in range(1,10)]
refEvent = 'onTime' #,'offTime']
# guess an epoch of -0.5 to +1s should be enough #
epochStart = -1; # stimulus ref event
epochEnd = +2; # 
epochLength = epochEnd-epochStart;

# Epoch Data
# ----------

# Get the goods on subject 1
(contSignalData1, metaData1, trackTrials, tokenPropsOrig, audioSignal, samplingRate, numChannels) = loadBookMEGWithAudio(megFile1, tokenPropsPickle, triggersOfInterest, epochEnd, epochStart, icaComps=False)

# Get the goods on subject 2
(contSignalData2, metaData2, trackTrials, tokenPropsOrig, audioSignal, samplingRate, numChannels) = loadBookMEGWithAudio(megFile2, tokenPropsPickle, triggersOfInterest, epochEnd, epochStart, icaComps=False)

# Get the goods on subject 3
(contSignalData3, metaData3, trackTrials, tokenPropsOrig, audioSignal, samplingRate, numChannels) = loadBookMEGWithAudio(megFile3, tokenPropsPickle, triggersOfInterest, epochEnd, epochStart, icaComps=False)

NUMSUBJS = 3
tokenProps = numpy.concatenate((tokenPropsOrig,tokenPropsOrig,tokenPropsOrig),axis=0)

#channelsOfInterest = [i for i in range(len(metaData1.chanlocs)) if metaData1.chanlocs[i].labels in channelLabels]

# REDUCE TRIALS TO JUST THOSE THAT CONTAIN A REAL WORD (NOT PUNCTUATION, SPACES, ...)
wordTrialsBool = numpy.array([p != '' for p in tokenProps['stanfPOS']])
#print(wordTrialsBool[:10])
# REDUCE TRIALS TO JUST THOSE THAT HAVE A DECENT DEPTH ESTIMATE
parsedTrialsBool = numpy.array([d != -1 for d in tokenProps['syndepth']])
d1TrialsBool = numpy.array([d == 1 for d in tokenProps['syndepth']])
d2TrialsBool = numpy.array([d == 2 for d in tokenProps['syndepth']])
d3TrialsBool = numpy.array([d == 3 for d in tokenProps['syndepth']])
#print(parsedTrialsBool[:10])

sentidlist = numpy.bincount(tokenProps['sentid'][tokenProps['sentid'] != -1]) #gives the sentence length indexed by sentence ID
sentidlist = sentidlist / float(NUMSUBJS) #account for the fact that we've seen each sentence 3 times (once per subject)
sentidlist = sentidlist.astype(int)
validsents = numpy.nonzero((sentidlist > 4) & (sentidlist <= 50))[0]# & (sentidlist >= 5))[0] #only permit sentences with lengths less than 51
#validsents = numpy.nonzero((sentidlist <= 50))[0]# & (sentidlist >= 5))[0] #only permit sentences with lengths less than 51

# Set up the dev and test sets
devitems = numpy.arange(1,max(tokenProps['sentid']),devsizerecip)
if FUDGE:
  devitems = numpy.arange(1,max(tokenProps['sentid']),devsizerecip*2) #split dev set in half
devTrialsBool = numpy.array([s in devitems and s in validsents for s in tokenProps['sentid']])
testTrialsBool = numpy.array([s not in devitems and s in validsents for s in tokenProps['sentid']])

if DEV:
  inDataset = devTrialsBool
else:
  inDataset = testTrialsBool

fitresults = {}
avefitsig_dict = {}
avefit_dict = {}
freqsY = None
trainX = None

channelLabels = ['MEG0322', 'MEG0323', 'MEG0342', 'MEG0343', 'MEG0112', 'MEG0113', 'MEG1532', 'MEG1533', 'MEG1712', 'MEG1713']
channelsOfInterest = [i for i in range(len(metaData1.chanlocs)) if metaData1.chanlocs[i].labels in channelLabels]

#for channelix in chanixes: 
#print 'Compiling data from channel:',channelix
  #need to reshape because severalMagChannels needs to be channel x samples, and 1-D shapes are flattened by numpy
severalMagChannels1 = contSignalData1[channelsOfInterest,:]

(wordTrials1, epochedSignalData1, epochSliceTimepoints, wordTimesAbsolute, numTrials, epochNumTimepoints) = wordTrialEpochify(severalMagChannels1, samplingRate, tokenPropsOrig, trackTrials, refEvent, epochEnd, epochStart)

severalMagChannels2 = contSignalData2[channelsOfInterest,:]
(wordTrials2, epochedSignalData2, epochSliceTimepoints, wordTimesAbsolute, numTrials, epochNumTimepoints) = wordTrialEpochify(severalMagChannels2, samplingRate, tokenPropsOrig, trackTrials, refEvent, epochEnd, epochStart)

severalMagChannels3 = contSignalData3[channelsOfInterest,:]
(wordTrials3, epochedSignalData3, epochSliceTimepoints, wordTimesAbsolute, numTrials, epochNumTimepoints) = wordTrialEpochify(severalMagChannels3, samplingRate, tokenPropsOrig, trackTrials, refEvent, epochEnd, epochStart)

if True:
    epochedSignalData = ( numpy.concatenate((epochedSignalData1,epochedSignalData2,epochedSignalData3), axis=0) )
  
    wordEpochs = epochedSignalData[wordTrialsBool & parsedTrialsBool & inDataset]
    d1wordEpochs = epochedSignalData[wordTrialsBool & d1TrialsBool & inDataset]
    d2wordEpochs = epochedSignalData[wordTrialsBool & d2TrialsBool & inDataset]
    d3wordEpochs = epochedSignalData[wordTrialsBool & d3TrialsBool & inDataset]
    wordFeatures = tokenProps[wordTrialsBool & parsedTrialsBool & inDataset]
    d1wordFeatures = tokenProps[wordTrialsBool & d1TrialsBool & inDataset]
    d2wordFeatures = tokenProps[wordTrialsBool & d2TrialsBool & inDataset]
    d3wordFeatures = tokenProps[wordTrialsBool & d3TrialsBool & inDataset]

    #COHERENCE ANALYSIS
    #freq_decomp = 'wavelet'
    NJOBS = 20 #dignam has 24 processor
    fmin = 4 #minimum frequency of interest (wavelet); 7
    fmax = 50 #maximum frequency of interest (wavelet); 30
    fstep = 1 #stepsize to get from fmin to fmax 
    
    names = channelLabels
    seed = list(numpy.arange(len(names)))*len(names)
    targets = numpy.array(seed)
    seed = numpy.sort(seed)
    #use indices = seed_target_indices for full cross connectivity (not useful for coherence)
    #indices = seed_target_indices(seed, targets)
    #use indices = None for lower triangular connectivity
    indices = None

    ####
    #use morlet wavelet decomposition
    ####
    
    cwt_frequencies = numpy.arange(fmin, fmax, fstep)
    #cwt_n_cycles = cwt_frequencies / 7.
    cwt_n_cycles = cwt_frequencies
    #Depth 1 coherence
    d1con, freqs, times, _, _ = spectral_connectivity(d1wordEpochs, indices=indices,
                                                      method='coh', mode='cwt_morlet', sfreq=samplingRate,
                                                      cwt_frequencies=cwt_frequencies, cwt_n_cycles=cwt_n_cycles, n_jobs=NJOBS, verbose='WARNING')
    #save for later
    with open('d1con.pkl','wb') as f:
      pickle.dump(d1con,f)
      
    #Depth 2 coherence
    d2con, freqs, times, _, _ = spectral_connectivity(d2wordEpochs, indices=indices,
                                                      method='coh', mode='cwt_morlet', sfreq=samplingRate,
                                                      cwt_frequencies=cwt_frequencies, cwt_n_cycles=cwt_n_cycles, n_jobs=NJOBS, verbose='WARNING')
    #save for later
    with open('d2con.pkl','wb') as f:
      pickle.dump(d2con,f)
    #Depth 3 coherence
    d3con, freqs, times, _, _ = spectral_connectivity(d3wordEpochs, indices=indices,
                                                      method='coh', mode='cwt_morlet', sfreq=samplingRate,
                                                      cwt_frequencies=cwt_frequencies, cwt_n_cycles=cwt_n_cycles, n_jobs=NJOBS, verbose='WARNING')
    #save for later
    with open('d3con.pkl','wb') as f:
      pickle.dump(d3con,f)

    n_rows, n_cols = d1con.shape[:2]
    vmin = min(min(d1con.ravel()),min(d2con.ravel()),min(d3con.ravel()))
    vmax = max(max(d1con.ravel()),max(d2con.ravel()),max(d3con.ravel()))

    fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=(15,15))
    plt.suptitle('Between sensor connectivity (coherence)')
    for i in range(n_rows):
      for j in range(n_cols):
        if i == j:
          axes[i, j].set_axis_off()
          continue

        cax = axes[i, j].imshow(d1con[i, j, :], vmin=vmin, vmax = vmax, aspect = 'auto')
        if epochStart < 0:
          axes[i, j].axvline(x=epochStart *-1 * samplingRate, c='black', lw=1)
          #else: onset isn't on graph

        if j == 0:
          axes[i, j].set_ylabel(names[i])
          axes[i, j].set_yticks(numpy.arange(len(cwt_frequencies))[::2])
          axes[i, j].set_yticklabels(cwt_frequencies[::2])
          axes[0, i].set_title(names[i])
        if i == (n_rows - 1):
          axes[i, j].set_xlabel(names[j])
          axes[i, j].set_xticks(numpy.arange(0.0,d1con.shape[3],62.5))
          axes[i, j].set_xticklabels(numpy.arange(epochStart,epochEnd,(epochEnd-epochStart)/(d1con.shape[3]/62.5)))
        axes[i, j].set_ylim([0.0, d1con.shape[2]-1])
    fig.colorbar(cax)
    if DEV:
      plt.savefig('graphics/coh_d1_dev.png')
    else:
      plt.savefig('graphics/coh_d1_test.png')

    fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=(15,15))
    plt.suptitle('Between sensor connectivity (coherence)')
    for i in range(n_rows):
      for j in range(n_cols):
        if i == j:
          axes[i, j].set_axis_off()
          continue

        cax = axes[i, j].imshow(d2con[i, j, :], vmin=vmin, vmax = vmax, aspect = 'auto')
        if epochStart < 0:
          axes[i, j].axvline(x=epochStart *-1 * samplingRate, c='black', lw=1)
          #else: onset isn't on graph

        if j == 0:
          axes[i, j].set_ylabel(names[i])
          axes[i, j].set_yticks(numpy.arange(len(cwt_frequencies))[::2])
          axes[i, j].set_yticklabels(cwt_frequencies[::2])
          axes[0, i].set_title(names[i])
        if i == (n_rows - 1):
          axes[i, j].set_xlabel(names[j])
          axes[i, j].set_xticks(numpy.arange(0.0,d2con.shape[3],62.5))
          axes[i, j].set_xticklabels(numpy.arange(epochStart,epochEnd,(epochEnd-epochStart)/(d2con.shape[3]/62.5)))
        axes[i, j].set_ylim([0.0, d2con.shape[2]-1])
    fig.colorbar(cax)
    if DEV:
      plt.savefig('graphics/coh_d2_dev.png')
    else:
      plt.savefig('graphics/coh_d2_test.png')

    fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=(15,15))
    plt.suptitle('Between sensor connectivity (coherence)')
    for i in range(n_rows):
      for j in range(n_cols):
        if i == j:
          axes[i, j].set_axis_off()
          continue

        cax = axes[i, j].imshow(d3con[i, j, :], vmin=vmin, vmax = vmax, aspect = 'auto')
        if epochStart < 0:
          axes[i, j].axvline(x=epochStart *-1 * samplingRate, c='black', lw=1)
          #else: onset isn't on graph

        if j == 0:
          axes[i, j].set_ylabel(names[i])
          axes[i, j].set_yticks(numpy.arange(len(cwt_frequencies))[::2])
          axes[i, j].set_yticklabels(cwt_frequencies[::2])
          axes[0, i].set_title(names[i])
        if i == (n_rows - 1):
          axes[i, j].set_xlabel(names[j])
          axes[i, j].set_xticks(numpy.arange(0.0,d3con.shape[3],62.5))
          axes[i, j].set_xticklabels(numpy.arange(epochStart,epochEnd,(epochEnd-epochStart)/(d3con.shape[3]/62.5)))
        axes[i, j].set_ylim([0.0, d3con.shape[2]-1])
    fig.colorbar(cax)
    if DEV:
      plt.savefig('graphics/coh_d3_dev.png')
    else:
      plt.savefig('graphics/coh_d3_test.png')
