# -*- coding: utf-8; python-indent: 2; -*-
# This script extracts P-values and R^2 values over each frequency band and determines model fits

# Global Vars
# =======

DEV = True #if True: analyze the dev set; if False: analyze the test set ;; DEV is defined on a sentence level using a stepsize of N ;; TEST is the complement of DEV
devsizerecip = 3 # the reciprocal of the dev size, so devsizerecip = 3 means the dev set is 1/3 and the test set is 2/3
CWTCYCLEPARAM = 2 # an int parameter to control the temporal/freq resolution of the wavelet decomposition; 2 is good freq resolution, 7 is good temporal resolution
SAVEDATA = False #Take the time to write the coherence data to disk
SAVEDIR = '/home/scratch/vanschm/meg/' #If saving to disk, which directory should it go in? '' for current dir, otherwise must end in a slash

VERBOSE = False #Provide some extra output, mainly for development purposes
FUDGE = False # this factor is used to enable and mark dubious code, which can be cleaned up later; purely for development
DRAW = True #Whether or not to draw the coherence connectivity matrix

#channelLabels = ['MEG0322', 'MEG0323', 'MEG0342', 'MEG0343', 'MEG0112', 'MEG0113', 'MEG1532', 'MEG1533', 'MEG1712', 'MEG1713']
channelLabels = ['MEG0223','MEG1513']
#channelLabels = ['MEG0133','MEG1542']
#channelLabels = ['MEG0122','MEG0132','MEG0223','MEG1513','MEG1712']
# GOODFREQS = the frequencies to significance test for
GOODFREQS = [9,10,40,41,42]
D3 = True

#coherence analysis settings
NJOBS = 20 #dignam has 24 processors
fmin = 4 #minimum frequency of interest (wavelet); 4
fmax = 50 #maximum frequency of interest (wavelet); 50
fstep = 1 #stepsize to get from fmin to fmax
tminsec = 0 #time start to calculate significance over (in seconds)
tmaxsec = 1 #time end to calculate significance over (in seconds)
samplingrate = 125

coherence_step = 4 #number of epochs to average over when calculating coherence

plusminus = 0

if DRAW and not D3:
  print "Can't draw if depth 3 not calculated"
  raise

#########
# Autocalculated
#########

tmin = int(tminsec*samplingrate + samplingrate)
tmax = int(tmaxsec*samplingrate + samplingrate)

if DEV:
  print ' Using DEV'
else:
  print ' Using TEST'
  if FUDGE:
    print ' Cannot fudge numbers on test set...'
    raise

if FUDGE:
  print ' Fudging numbers...'
  
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
maintTrialsBool = numpy.array([d == 0 for d in tokenProps['deltadepth']]) #location of hypothesized storage cost

sentidlist = numpy.bincount(tokenProps['sentid'][tokenProps['sentid'] != -1]) #gives the sentence length indexed by sentence ID
sentidlist = sentidlist / float(NUMSUBJS) #account for the fact that we've seen each sentence 3 times (once per subject)
sentidlist = sentidlist.astype(int)
validsents = numpy.nonzero((sentidlist > 4) & (sentidlist <= 50))[0]# & (sentidlist >= 5))[0] #only permit sentences with lengths less than 51
#validsents = numpy.nonzero((sentidlist <= 50))[0]# & (sentidlist >= 5))[0] #only permit sentences with lengths less than 51

# Set up the dev and test sets
devitems = numpy.arange(1,max(tokenProps['sentid']),devsizerecip)
#if FUDGE:
#  devitems = numpy.arange(1,max(tokenProps['sentid']),devsizerecip*2) #split dev set in half
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
#d1wordFeatures = tokenProps[wordTrialsBool & d1TrialsBool & inDataset]
#d2wordFeatures = tokenProps[wordTrialsBool & d2TrialsBool & inDataset]
#d3wordFeatures = tokenProps[wordTrialsBool & d3TrialsBool & inDataset]

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
if FUDGE:
  d1mwordEpochs = d1mwordEpochs[0:200]
  d2mwordEpochs = d2mwordEpochs[0:200]
  d3mwordEpochs = d3mwordEpochs[0:200]
print 'd1mwordEpochs',d1mwordEpochs.shape
print 'd2mwordEpochs',d2mwordEpochs.shape
print 'd3mwordEpochs',d3mwordEpochs.shape

d1wcon = word_connectivity(d1mwordEpochs, indices=indices, step = coherence_step)
d2wcon = word_connectivity(d2mwordEpochs, indices=indices, step = coherence_step)
if D3:
  d3wcon = word_connectivity(d3mwordEpochs, indices=indices, step = coherence_step)
print 'd1wcon',d1wcon.shape
print 'd2wcon',d2wcon.shape
if D3:
  print 'd3wcon',d3wcon.shape

#sample only the smallest set to enable t-testing
#smallest = min(d1wcon.shape[0],d2wcon.shape[0],d3wcon.shape[0])

#shapes.index(min(shapes))

#d1wcon = dynpool(d1wcon,smallest)
#d2wcon = dynpool(d2wcon,smallest)
#d3wcon = dynpool(d3wcon,smallest)

#Can't do this because d1,d2,d3 don't have the same number of samples
#d32 = d3wcon - d2wcon
#d21 = d2wcon - d1wcon
#
#d32simple = numpy.empty((d32.shape[0],len(cwt_frequencies)))
#d21simple = numpy.empty((d21.shape[0],len(cwt_frequencies)))
#
#for i in range(d32simple.shape[0]):
#  for fi in range(d32simple.shape[1]):
#    d32simple[i,fi] = numpy.mean(d32[i,1,0,fi])
#for i in range(d21simple.shape[0]):
#  for fi in range(d21simple.shape[1]):
#    d21simple[i,fi] = numpy.mean(d21[i,1,0,fi])
#print 'd3-d2:',numpy.mean(d32simple,axis=0)
#print 'd2-d1:',numpy.mean(d21simple,axis=0)
#for col in range(d21simple.shape[1]):
#  print cwt_frequencies[col], 'Hz:', scipy.stats.ttest_ind(d21simple[:,col],d32simple[:,col])

d1mcon = numpy.mean(d1wcon,axis=0)
d2mcon = numpy.mean(d2wcon,axis=0)
if D3:
  d3mcon = numpy.mean(d3wcon,axis=0)

d1simple = numpy.empty((d1wcon.shape[0],len(cwt_frequencies)))
d2simple = numpy.empty((d2wcon.shape[0],len(cwt_frequencies)))
if D3:
  d3simple = numpy.empty((d3wcon.shape[0],len(cwt_frequencies)))

#wordEpochs x chans x chans x freqs x timeSamples

#print numpy.mean(d1wcon[0,0,0,0])
#print numpy.mean(d1wcon[0,0,1,0])
print d1wcon[0,1,0,0,0:10]
print d1wcon[1,1,0,0,0:10]
print d1wcon[2,1,0,0,0:10]
print d2wcon[0,1,0,0,0:10]
if D3:
  print d3wcon[0,1,0,0,0:10]
#print numpy.mean(d1wcon[0,1,0,0])
#print numpy.mean(d1wcon[0,1,1,0])

for i in range(d1simple.shape[0]):
  for fi in range(d1simple.shape[1]):
    d1simple[i,fi] = numpy.mean(d1wcon[i,1,0,fi,tmin:tmax])
for i in range(d2simple.shape[0]):
  for fi in range(d2simple.shape[1]):
    d2simple[i,fi] = numpy.mean(d2wcon[i,1,0,fi,tmin:tmax])
if D3:
  for i in range(d3simple.shape[0]):
    for fi in range(d3simple.shape[1]):
      d3simple[i,fi] = numpy.mean(d3wcon[i,1,0,fi,tmin:tmax])
print 'd1:',numpy.mean(d1simple,axis=0)[[GOODFREQS-fmin]], '::', d1simple.shape
print 'd2:',numpy.mean(d2simple,axis=0)[[GOODFREQS-fmin]], '::', d2simple.shape
if D3:
  print 'd3:',numpy.mean(d3simple,axis=0)[[GOODFREQS-fmin]], '::', d3simple.shape

conpkg = {'conmats':[],'freqs':freqs,'electrodes':channelLabels}

if SAVEDATA:
  print 'Writing coherence metrics to disk'

  conpkg['conmats'] = d1wcon
  with open(SAVEDIR+'cohd1m.pkl','wb') as f:
    pickle.dump(conpkg,f)
  conpkg['conmats'] = d2wcon
  with open(SAVEDIR+'cohd2m.pkl','wb') as f:
    pickle.dump(conpkg,f)
  if D3:
    conpkg['conmats'] = d3wcon
    with open(SAVEDIR+'cohd3m.pkl','wb') as f:
      pickle.dump(conpkg,f)
  
for col in GOODFREQS-fmin: #range(d1simple.shape[1]):  
  print 'd2-d1:', cwt_frequencies[col], 'Hz:', numpy.mean(d2simple,axis=0)[col],'-', numpy.mean(d1simple,axis=0)[col],'p=', scipy.stats.f_oneway(d1simple[:,col],d2simple[:,col])
  if D3:
    print 'd3-d2:', cwt_frequencies[col], 'Hz:', numpy.mean(d3simple,axis=0)[col],'-', numpy.mean(d2simple,axis=0)[col],'p=', scipy.stats.f_oneway(d2simple[:,col],d3simple[:,col])

##save for later
#with open('d1con.pkl','wb') as f:
#  pickle.dump(d1con,f)
#  
##Depth 2 coherence
#d2con, freqs, times, _, _ = spectral_connectivity(d2wordEpochs, indices=indices,
#                                                  method='coh', mode='cwt_morlet', sfreq=samplingRate,
#                                                  cwt_frequencies=cwt_frequencies, cwt_n_cycles=cwt_n_cycles, n_jobs=NJOBS, verbose='WARNING')
##save for later
#with open('d2con.pkl','wb') as f:
#  pickle.dump(d2con,f)
##Depth 3 coherence
#d3con, freqs, times, _, _ = spectral_connectivity(d3wordEpochs, indices=indices,
#                                                  method='coh', mode='cwt_morlet', sfreq=samplingRate,
#                                                  cwt_frequencies=cwt_frequencies, cwt_n_cycles=cwt_n_cycles, n_jobs=NJOBS, verbose='WARNING')
##save for later
#with open('d3con.pkl','wb') as f:
#  pickle.dump(d3con,f)
#Storage coherence
#d2scon, freqs, times, _, _ = spectral_connectivity(d2swordEpochs, indices=indices,
#                                                  method='coh', mode='cwt_morlet', sfreq=samplingRate,
#                                                  cwt_frequencies=cwt_frequencies, cwt_n_cycles=cwt_n_cycles, n_jobs=NJOBS, verbose='WARNING')
#d2mcon, freqs, times, _, _ = spectral_connectivity(d2mwordEpochs, indices=indices,
#                                                  method='coh', mode='cwt_morlet', sfreq=samplingRate,
#                                                  cwt_frequencies=cwt_frequencies, cwt_n_cycles=cwt_n_cycles, n_jobs=NJOBS, verbose='WARNING')
#print 'freqs',freqs
#d3mcon, freqs, times, _, _ = spectral_connectivity(d3mwordEpochs, indices=indices,
#                                                  method='coh', mode='cwt_morlet', sfreq=samplingRate,
#                                                  cwt_frequencies=cwt_frequencies, cwt_n_cycles=cwt_n_cycles, n_jobs=NJOBS, verbose='WARNING')
#d2icon, freqs, times, _, _ = spectral_connectivity(d2iwordEpochs, indices=indices,
#                                                  method='coh', mode='cwt_morlet', sfreq=samplingRate,
#                                                  cwt_frequencies=cwt_frequencies, cwt_n_cycles=cwt_n_cycles, n_jobs=NJOBS, verbose='WARNING')
#d2ficon, freqs, times, _, _ = spectral_connectivity(d2fiwordEpochs, indices=indices,
#                                                  method='coh', mode='cwt_morlet', sfreq=samplingRate,
#                                                  cwt_frequencies=cwt_frequencies, cwt_n_cycles=cwt_n_cycles, n_jobs=NJOBS, verbose='WARNING')
#save for later
#with open('storcon.pkl','wb') as f:
#  pickle.dump(storcon,f)
#Integration coherence
#integcon, freqs, times, _, _ = spectral_connectivity(integwordEpochs, indices=indices,
#                                                  method='coh', mode='cwt_morlet', sfreq=samplingRate,
#                                                  cwt_frequencies=cwt_frequencies, cwt_n_cycles=cwt_n_cycles, n_jobs=NJOBS, verbose='WARNING')
#save for later
#with open('integcon.pkl','wb') as f:
#  pickle.dump(integcon,f)

#vmin = min(min(d1con.ravel()),min(d2con.ravel()),min(d3con.ravel()))
#vmax = max(max(d1con.ravel()),max(d2con.ravel()),max(d3con.ravel()))
#draw_con_matrix(d1con, 'd1', vmin, vmax)
#draw_con_matrix(d2con, 'd2', vmin, vmax)
#draw_con_matrix(d3con, 'd3', vmin, vmax)
#draw_con_matrix(d4con, 'd4', vmin, vmax)

#Do storage and integration analyses
#integcon = d2icon - d2mcon
#fintegcon = d2ficon - d2mcon
#storcon = d2scon - d2mcon
if D3:
  maintcon = d3mcon - d2mcon

#conpkg = {'conmats':[],'freqs':freqs,'electrodes':channelLabels}
#conpkg['conmats'].append(d32simple)
#conpkg['conmats'].append(d21simple)
#if DEV:
#  pklname = 'con_dev.pkl'
#else:
#  pklname = 'con_test.pkl'
#with open(pklname,'wb') as f:
#    pickle.dump(conpkg,f)

if DRAW:
  vmin = None
  vmax = None
  
  print 'Computing means'
  #+/- 2 Hz, 0-.5s, 25 & 40 Hz
  third = int(maintcon.shape[3]/3.)
  sixth = int(third * .5)
  twelfth = int(sixth * .5)
  print maintcon.shape
  #for findex,freq in zip(findices,GOODFREQS):
    #print 'index:', findex, 'freq:',freq
    #print str(freq)+':', numpy.mean(maintcon[1,0,findex-plusminus:findex+plusminus+1,third-sixth:third+sixth],axis=1)
  #  print str(freq)+':', numpy.mean(maintcon[1,0,findex-plusminus:findex+plusminus+1,third-sixth:third+sixth])
  #mean1 = numpy.mean(maintcon[1,0,findex1-plusminus:findex1+plusminus+1,third+twelfth:third+sixth],axis=1)
  #print mean1.shape
  #mean2 = numpy.mean(maintcon[1,0,findex2-plusminus:findex2+plusminus+1,third+twelfth:third+sixth],axis=1)
  #print '22:', mean1
  #print '40:', mean2
  #print '22:',numpy.mean(numpy.mean(maintcon[1,0,findex1-2:findex1+3,third:third+sixth],axis=1))
  #print '22:',
  #print '40:',numpy.mean(numpy.mean(maintcon[1,0,findex2-2:findex2+3,third:third+sixth],axis=1))
  #print '40:',numpy.mean(maintcon[1,0,findex2-2:findex2+3,third:third+sixth])
  
  #vmin = min(min(storcon.ravel()),min(integcon.ravel()),min(fintegcon.ravel()))
  #vmax = max(max(storcon.ravel()),max(integcon.ravel()),max(fintegcon.ravel()))
  #vmin = -0.07
  #vmax = 0.07
  
  #draw_con_matrix(fintegcon, 'fintegration', vmin, vmax)
  #draw_con_matrix(integcon, 'integration', vmin, vmax)
  #draw_con_matrix(storcon, 'storage', vmin, vmax)
  if DEV:
    draw_con_matrix(maintcon, 'maintenance', vmin, vmax)
  else:
    print 'No cheating allowed; graphing suppressed'
  
  #run_ttest((d1wcon,d2wcon,d3wcon),Tchannels,GOODFREQ)
