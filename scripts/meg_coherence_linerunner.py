# -*- coding: utf-8; python-indent: 2; -*-
# This script extracts P-values and R^2 values over each frequency band and determines model fits

# Global Vars
# =======

DEV = True # if True: analyze the dev set; if False: analyze the test set ;; DEV is defined on a sentence level using a stepsize of N ;; TEST is the complement of DEV
devsizerecip = 3 # the reciprocal of the dev size, so devsizerecip = 3 means the dev set is 1/3 and the test set is 2/3
CWTCYCLEPARAM = 2 # an int parameter to control the temporal/freq resolution of the wavelet decomposition; 2 is good freq resolution, 7 is good temporal resolution
SAVEDATA = False #Take the time to write the coherence data to disk
SAVEDIR = '/home/scratch/vanschm/meg/' #If saving to disk, which directory should it go in? '' for current dir, otherwise must end in a slash

VERBOSE = False #Provide some extra output, mainly for development purposes

#channelLabels = ['MEG0322', 'MEG0323', 'MEG0342', 'MEG0343', 'MEG0112', 'MEG0113', 'MEG1532', 'MEG1533', 'MEG1712', 'MEG1713']
#channelLabels = ['MEG0133','MEG1743']
channelLabels = ['MEG0132','MEG1712']
#channelLabels = ['MEG0133','MEG1542']
#channelLabels = ['MEG0122','MEG0132','MEG0223','MEG1513','MEG1712']
# GOODFREQS = the frequencies to significance test for
GOODFREQS = [32,42]

# logFreq_ANC     bigramLogProbBack_COCA  trigramLogProbBack_COCA surprisal2back_COCA
featureName = 'surprisal2back_COCA'


#GOODFREQS = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46]

#coherence analysis settings
plusminus = 2
NJOBS = 20 #dignam has 24 processors
fmin = 4 #minimum frequency of interest (wavelet); 4
fmax = 50 #maximum frequency of interest (wavelet); 50
fstep = 1 #stepsize to get from fmin to fmax
tminsec = 0 #time start to calculate significance over (in seconds)
tmaxsec = 0.5 #time end to calculate significance over (in seconds)
samplingrate = 125

coherence_step = 4 #number of epochs to average over when calculating coherence

#########
# Autocalculated
#########

tmin = int(tminsec*samplingrate + samplingrate)
tmax = int(tmaxsec*samplingrate + samplingrate)

print str(channelLabels)
if DEV:
  print ' Using DEV'
else:
  print ' Using TEST'
print ' Plus/Minus:', str(plusminus), 'Coherence group size:', str(coherence_step)
print ' Testing Feature:', featureName
  
# Imports
# =======

import time
import cPickle as pickle
import logging as L
L.basicConfig(level=L.ERROR) # INFO)
import time
import numpy
import scipy
import scipy.stats
import pandas as pd
import os
#import pylab
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
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

GOODFREQS = numpy.array(GOODFREQS)

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

#def dynpool(conmat, numEpochs):
#  #conmat has the following shape: wordEpochs x chans x chans x freqs x timeSamples
#  poolsize = conmat.shape[0] / float(numEpochs)
#  if poolsize == 

def draw_con_matrix(conmat, fnamecode, vmin=None, vmax=None, method = 'coherence'):
  #draws the connectivity matrix
  if vmin == None:
    vmin = min(conmat.ravel())
  if vmax == None:
    vmax = max(conmat.ravel())

  n_rows, n_cols = conmat.shape[:2]

  fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=(15,15))
  plt.suptitle('Between sensor connectivity (%s)' % (method) )
  for i in range(n_rows):
    for j in range(i+1):
      if i == j:
        axes[i, j].set_axis_off()
        continue

      cax = axes[i, j].imshow(conmat[i, j, :], vmin=vmin, vmax = vmax, aspect = 'auto') #, interpolation='nearest')
      if epochStart < 0:
        axes[i, j].axvline(x=epochStart *-1 * samplingRate, c='black', lw=1)
      #else: onset isn't on graph

      if j == 0:
        axes[i, j].set_ylabel(names[i])
        axes[i, j].set_yticks(numpy.arange(len(cwt_frequencies))[::6])
        axes[i, j].set_yticklabels(cwt_frequencies[::6])
        axes[0, i].set_title(names[i])
      if i == (n_rows - 1):
        axes[i, j].set_xlabel(names[j])
        axes[i, j].set_xticks(numpy.arange(0.0,conmat.shape[3],62.5))
        axes[i, j].set_xticklabels(numpy.arange(epochStart,epochEnd,(epochEnd-epochStart)/(conmat.shape[3]/62.5)))
      axes[i, j].set_ylim([0.0, conmat.shape[2]-1])
  fig.colorbar(cax)
  if DEV:
    plt.savefig('graphics/coh_%s_dev.png' % (fnamecode) )
  else:
    plt.savefig('graphics/coh_%s_test.png' % (fnamecode) )

def draw_tgraph(data, fnamecode, vmin=None, vmax=None, valtype='t'):
  data = data.reshape(-1,1)
  #draws the tgraph
  if vmin == None:
    vmin = min(data.ravel())
  if vmax == None:
    vmax = max(data.ravel())

  fig = plt.figure()

  plt.suptitle('Frequency %s-values (|%s|)' % (valtype,valtype))
  ax = fig.add_subplot(111)

  cax = ax.matshow(data, interpolation='nearest', vmin=vmin, vmax=vmax, cmap='binary')
  plt.gca().invert_yaxis()
  fig.colorbar(cax)

  ax.set_yticks(numpy.arange(len(cwt_frequencies))[::6])
  ax.set_yticklabels(cwt_frequencies[::6])

  if DEV:
    plt.savefig('graphics/%sgraph_%s_dev.png' % (valtype,fnamecode) )
  else:
    plt.savefig('graphics/%sgraph_%s_test.png' % (valtype,fnamecode) )

def word_connectivity(wordEpochs,indices,step=2):
  wordconnectivity = numpy.empty((int(wordEpochs.shape[0]/step),wordEpochs.shape[1],wordEpochs.shape[1],len(cwt_frequencies),epochLength*samplingRate))
  # this array is wordEpochs x chans x chans x freqs x timeSamples
  print 'wordconnectivity',wordconnectivity.shape
  total = wordEpochs.shape[0]-wordEpochs.shape[0]%step
  for i in range(0,total/step):
    word = wordEpochs[step*i:step*(i+1)]
    if i == 0:
      print 'word',word.shape
    if step == 1:
      word = word.reshape((1,word.shape[0],word.shape[1]))
    if i == 0:
      if step == 1:
        print 'reshaped word',word.shape
      print 'cwt_frequencies',cwt_frequencies.shape
      print 'cwt_n_cycles',cwt_n_cycles.shape
    if i % 200 == 0:
      print 'Epoch %d/%d (%d)' % (i,total/step,total)
    wordconnectivity[i], freqs, times, _, _ = spectral_connectivity(word, indices=indices,
                                                                    method='coh', mode='cwt_morlet', sfreq=samplingRate,
                                                                    cwt_frequencies=cwt_frequencies, cwt_n_cycles=cwt_n_cycles, n_jobs=NJOBS, verbose='WARNING')
  return(wordconnectivity)


def find_pairs(num):
  pairs = []
  for i in range(num-1):
    for j in range(i+1,num):
      pairs.append( (i,j) )
  return(pairs)

def run_ttest(wordesfreqtimes,goodcols,goodfreq):
  #find goodcols
  #create lists of the average coherence for each word (row in wft) according to the given frequency and the given electrode pair
  #run an unpaired t-test for each list derived from wft
  tdata = numpy.zeros((len(wordesfreqtimes),cwt_frequencies.shape[0]))
  e1 = names.index(goodcols[0])
  e2 = names.index(goodcols[1])
  if e2 < e1:
    er = e2
    ec = e1
  else:
    er = e1
    ec = e2
  for di,dataset in enumerate(wordesfreqtimes):
    #d1, d2, d3
    for word in dataset:
      #the, boy, walked
      for fi,freq in word[er,ec]:
        #given the word and the two electrodes of interest, deal with the freq x time matrix
        tdata[di,fi] = numpy.mean(freq)
  for pair in find_pairs(tdata.shape[0]):
    pass
    

# Preprocessing
# =============

#choose a file - I found participant V to be pretty good, and 0.01 to 50Hz filter is pretty conservative #
(megFileTag1, megFile1) = ('V_TSSS_0.01-50Hz_@125', '../MEG_data/v_hod_allRuns_tsss_audiobookPrepro_stPad1_lp50_resamp125_frac10ICAed.set')#_hp0.010000.set')
(megFileTag2, megFile2) = ('A_TSSS_0.01-50Hz_@125', '../MEG_data/aud_hofd_a_allRuns_tsss_audiobookPrepro_stPad1_lp50_resamp125_frac10ICAed_hp0.010000.set')
(megFileTag3, megFile3) = ('C_TSSS_0.01-50Hz_@125', '../MEG_data/aud_hofd_c_allRuns_tsss_audiobookPrepro_stPad1_lp50_resamp125_frac10ICAed_hp0.010000.set')

## put your own properties in here, as a .tab file of similar format (tab delimited, and field names in a first comment line - should be easy to do in excel...)
#to get the V5.tab:
#  python ../scripts/buildtab.py hod_JoeTimes_LoadsaFeaturesV3.tab hod.wsj02to21-comparativized-gcg15-1671-4sm.fullberk.parsed.gcgbadwords > hod_JoeTimes_LoadsaFeaturesV4.tab
#  python ../scripts/expandtab.py hod_JoeTimes_LoadsaFeaturesV4.tab > hod_JoeTimes_LoadsaFeaturesV5.tab
tokenPropsFile = '../MEG_data/hod_JoeTimes_LoadsaFeaturesV5.tab'

# LOAD WORD PROPS
# change dtype to suit the files in your .tab file #
tokenProps = scipy.genfromtxt(tokenPropsFile,
                              delimiter='\t',names=True,
                              dtype="i4,f4,f4,S50,S50,i2,i2,i2,S10,f4,f4,f4,f4,f4,f4,f4,f4,f4,f4,f4,i4,>i4,i4,i4")
# ... and temporarily save as cpickle archive to satisfy the way I programmed the convenience function loadBookMEGWithAudio (it expects to find the same info in a C-pickle file, and so doesn't need to know about number and type of fields)
tokenPropsPickle = tokenPropsFile+'.cpk'
pickle.dump(tokenProps, open(tokenPropsPickle, 'wb'))

triggersOfInterest=['s%d' % i for i in range(1,10)]
refEvent = 'onTime' #,'offTime']
# guess an epoch of -0.5 to +1s should be enough #
#epochStart = -1; # stimulus ref event
#epochEnd = +2; #
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
d4TrialsBool = numpy.array([d == 4 for d in tokenProps['syndepth']])
fintegTrialsBool = numpy.array([d < 0 for d in tokenProps['futdeltadepth']]) #location of hypothesized integration cost
integTrialsBool = numpy.array([d < 0 for d in tokenProps['deltadepth']]) #location of hypothesized integration cost
storTrialsBool = numpy.array([d > 0 for d in tokenProps['deltadepth']]) #location of hypothesized storage cost
maintTrialsBool = numpy.array([d == 0 for d in tokenProps['deltadepth']]) #location of hypothesized maintenance cost

sentidlist = numpy.bincount(tokenProps['sentid'][tokenProps['sentid'] != -1]) #gives the sentence length indexed by sentence ID
sentidlist = sentidlist / float(NUMSUBJS) #account for the fact that we've seen each sentence 3 times (once per subject)
sentidlist = sentidlist.astype(int)
validsents = numpy.nonzero((sentidlist > 4) & (sentidlist <= 50))[0]# & (sentidlist >= 5))[0] #only permit sentences with lengths less than 51
#validsents = numpy.nonzero((sentidlist <= 50))[0]# & (sentidlist >= 5))[0] #only permit sentences with lengths less than 51

# Set up the dev and test sets
devitems = numpy.arange(1,max(tokenProps['sentid']),devsizerecip)
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


epochedSignalData = ( numpy.concatenate((epochedSignalData1,epochedSignalData2,epochedSignalData3), axis=0) )

#acquire datasets

wordEpochs = epochedSignalData[wordTrialsBool & parsedTrialsBool & inDataset]
d1wordEpochs = epochedSignalData[wordTrialsBool & d1TrialsBool & inDataset]
d2wordEpochs = epochedSignalData[wordTrialsBool & d2TrialsBool & inDataset]
d3wordEpochs = epochedSignalData[wordTrialsBool & d3TrialsBool & inDataset]
integwordEpochs = epochedSignalData[wordTrialsBool & integTrialsBool & inDataset]
storwordEpochs = epochedSignalData[wordTrialsBool & storTrialsBool & inDataset]
d2iwordEpochs = epochedSignalData[wordTrialsBool & integTrialsBool & inDataset & d2TrialsBool]
d2fiwordEpochs = epochedSignalData[wordTrialsBool & fintegTrialsBool & inDataset & d2TrialsBool]
d2swordEpochs = epochedSignalData[wordTrialsBool & storTrialsBool & inDataset & d2TrialsBool]
d1mwordEpochs = epochedSignalData[wordTrialsBool & maintTrialsBool & inDataset & d1TrialsBool]
d2mwordEpochs = epochedSignalData[wordTrialsBool & maintTrialsBool & inDataset & d2TrialsBool]
d3mwordEpochs = epochedSignalData[wordTrialsBool & maintTrialsBool & inDataset & d3TrialsBool]
#wordFeatures = tokenProps[wordTrialsBool & parsedTrialsBool & inDataset]
mwordEpochs = epochedSignalData[wordTrialsBool & maintTrialsBool & inDataset]
mwordFeatures = tokenProps[wordTrialsBool & maintTrialsBool & inDataset]
#d1wordFeatures = tokenProps[wordTrialsBool & d1TrialsBool & inDataset]
#d2wordFeatures = tokenProps[wordTrialsBool & d2TrialsBool & inDataset]
#d3wordFeatures = tokenProps[wordTrialsBool & d3TrialsBool & inDataset]

myFeatures = mwordFeatures[featureName]
myOrder = numpy.argsort(myFeatures.ravel())
myFeatures = myFeatures[myOrder] #sort the features
mwordEpochs = mwordEpochs[myOrder] #sort the epochs based on associated feature values

myFeatures = myFeatures[:len(myFeatures)/coherence_step * coherence_step] #chop off extra epochs
myMeanFeatures = numpy.mean(myFeatures.reshape(-1,coherence_step), axis=1) #group features by coherenced epochs

#COHERENCE ANALYSIS
#freq_decomp = 'wavelet'

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

freqs = cwt_frequencies = numpy.arange(fmin, fmax, fstep)
findices = []
for freq in GOODFREQS:
  findices.append(numpy.where(cwt_frequencies == freq)[0][0])
#cwt_n_cycles = cwt_frequencies / 7.
cwt_n_cycles = cwt_frequencies / float(CWTCYCLEPARAM)

print 'Calculating coherence'
#Depth 1 coherence
#d1con, freqs, times, _, _ = spectral_connectivity(d1mwordEpochs[0:200], indices=indices,
#                                                  method='coh', mode='cwt_morlet', sfreq=samplingRate,
#                                                  cwt_frequencies=cwt_frequencies, cwt_n_cycles=cwt_n_cycles, n_jobs=NJOBS, verbose='WARNING')

print 'mwordEpochs',mwordEpochs.shape
mwcon = word_connectivity(mwordEpochs, indices=indices, step = coherence_step)
print 'mwcon',mwcon.shape
mcon = numpy.mean(mwcon,axis=0)

msimplea = numpy.empty((mwcon.shape[0],len(cwt_frequencies)))
msimple = numpy.empty((mwcon.shape[0],len(cwt_frequencies)))

#wordEpochs x chans x chans x freqs x timeSamples

for i in range(msimple.shape[0]):
  for fi in range(msimple.shape[1]):
    msimplea[i,fi] = numpy.mean(mwcon[i,1,0,fi,tmin:tmax])
if plusminus != 0:
  #average over multiple frequencies
  for i in range(msimple.shape[0]):
    for fi in range(msimple.shape[1]):
      msimple[i,fi] = numpy.mean(msimplea[i,max(0,fi-plusminus):min(msimple.shape[1],fi+plusminus+1)])
else:
  msimple = msimplea

msimple = msimple.transpose() # freqs x epochs

if DEV:
  for fi in range(msimple.shape[0]):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(myMeanFeatures,msimple[fi])
    print fi+fmin,'Hz:', 'm:',slope, 'b:',intercept, 'R2:',r_value, 'sigma:',std_err, 'p:',p_value
else:
  for fi in GOODFREQS-fmin:
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(myMeanFeatures,msimple[fi])
    print fi+fmin,'Hz:', 'm:',slope, 'b:',intercept, 'R2:',r_value, 'sigma:',std_err, 'p:',p_value
