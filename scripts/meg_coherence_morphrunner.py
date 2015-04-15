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
CHECK_NORMALITY = False #Print normality plots

channelLabels = ['MEG0122', 'MEG0132', 'MEG1512', 'MEG1522', 'MEG1532', 'MEG1642', 'MEG1722', 'MEG1712'] #left-side longitudinal pairings
#channelLabels = ['MEG1712', 'MEG1732', 'MEG1932', 'MEG1922', 'MEG2132', 'MEG2342', 'MEG2512', 'MEG2532'] #posterior left-right longitudinal pairings
#channelLabels = ['MEG0133','MEG1743']
#channelLabels = ['MEG0132','MEG1712']
#channelLabels = ['MEG1732','MEG2543']
#channelLabels = ['MEG0133','MEG1542']
#channelLabels = ['MEG0122','MEG0132','MEG0223','MEG1513','MEG1712']
# GOODFREQS = the frequencies to significance test for
GOODFREQS = [10]
GOODFREQS2 = [10]
GOODFREQS3 = [10]

#GOODFREQS = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46]
DEPTH = True #test depth or something else?
TEST2 = True
TEST3 = True
OPTIMAL = True #test u-values of chosen freqs to determine the best freq
SCAN = True #scan all freqs instead of just the specified ones

if not DEV:
  if DRAW:
    print 'No cheating allowed; graphing suppressed'
    DRAW = False
  if OPTIMAL:
    print 'No cheating allowed; optimal search suppressed'
    OPTIMAL = False
  if SCAN:
    print 'No cheating allowed; scanning suppressed'
    SCAN = False

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

#if DRAW and (not TEST3 or not TEST2):
#  print "Can't draw if depth 2 and depth 3 not calculated"
#  raise

#########
# Autocalculated
#########

tmin = int(tminsec*samplingrate + samplingrate)
tmax = int(tmaxsec*samplingrate + samplingrate)

channelNumbers = [chan[-4:] for chan in channelLabels]
print str(channelLabels)
if DEV:
  print ' Using DEV'
else:
  print ' Using TEST'
print ' Plus/Minus:', str(plusminus), 'Coherence group size:', str(coherence_step)
if SCAN:
  print ' Scanning all frequencies'
if OPTIMAL:
  print ' Using optimal search based on u-test p-values'
  
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import ticker
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

if SCAN:
  GOODFREQS = numpy.arange(fmin,fmax)
  GOODFREQS2 = numpy.arange(fmin,fmax)
  GOODFREQS3 = numpy.arange(fmin,fmax)
else:
  GOODFREQS = numpy.array(GOODFREQS)
  GOODFREQS2 = numpy.array(GOODFREQS2)
  GOODFREQS3 = numpy.array(GOODFREQS3)


#LaTeX fonts
#plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#plt.rc('text', usetex=True)

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

def mycmap(data,ugraph):
  #overlay ugraph on data as inverted alpha level
  #cdict = {'red': [(0.0, 0.0, 0.0),
  #                 (0.5, 1.0, 1.0),
  #                 (1.0, 1.0, 1.0)],
  #         'green': [(0.0, 0.0, 0.0),
  #                   (0.5, 1.0, 1.0),
  #                   (1.0, 0.0, 0.0)],
  #         'blue': [(0.0, 1.0, 1.0),
  #                  (0.5, 1.0, 1.0),
  #                  (1.0, 0.0, 0.0)],
  #         'alpha': [(0.0, 1.0, 1.0),
  #                   (0.5, 1.0, 1.0),
  #                   (0.0, 1.0, 1.0)]}
  #my_cmap = LinearSegmentedColormap('my_colormap',cdict,256)
  #plt.register_cmap(cmap=my_cmap)
  #                  
  #tmp = plt.cm.my_cmap(data)
  small = numpy.min(data.ravel())
  big = numpy.max(data.ravel())
  tmp = plt.cm.jet((data - small)/(big-small)) #scale data 0-1 first, to see full range of color
  for i in xrange(tmp.shape[0]):
    for j in xrange(tmp.shape[1]):
      #newmap[i,j] = tmp[i,j]
      #newmap[i,j,3] = 1-0.9*min(1,ugraph[i])
      tmp[i,j,3] = 1-min(0.8,7*ugraph[i])
  return tmp

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
  for i in xrange(n_rows):
    for j in xrange(i+1):
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

def draw_single_con_matrix(conmat, fnamecode, vmin=None, vmax=None, method = 'coherence'):
  #draws the connectivity matrix after overlaying an alpha filter based on the p-values in the ugraph
  if vmin == None:
    vmin = min(conmat.ravel())
  if vmax == None:
    vmax = max(conmat.ravel())

  n_rows, n_cols = conmat.shape[:2]

  #fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=(15,15))
  #plt.rc('text', usetex=True)
  #plt.rc('font', family='serif')
  
  fig, ax = plt.subplots(figsize=(4.,3.5))
  im = ax.imshow(conmat[1,0], vmin = vmin, vmax = vmax, aspect = 'auto')
  #plt.suptitle('Sensor coherence')
  #for i in range(n_rows):
  #  for j in range(i+1):
  #    if i == j:
  #      axes[i, j].set_axis_off()
  #      continue
  #
  #    cax = axes[i, j].imshow(conmat[i, j, :], vmin=vmin, vmax = vmax, aspect = 'auto') #, interpolation='nearest')
  if epochStart < 0:
    ax.axvline(x=epochStart *-1 * samplingRate, c='black', lw=1)
  #else: onset isn't on graph

  #ax.set_ylabel(names[1])
  ax.set_yticks(numpy.arange(len(cwt_frequencies))[::6])
  ax.set_yticklabels(cwt_frequencies[::6])
  #ax.set_title(names[1])
  #ax.set_xlabel(names[0])
  ax.set_xticks(numpy.arange(0.0,conmat.shape[3],62.5))
  ax.set_xticklabels(numpy.arange(epochStart,epochEnd,(epochEnd-epochStart)/(conmat.shape[3]/62.5)))
  ax.set_ylim([0.0, conmat.shape[2]-1])
  plt.colorbar(im)
  if DEV:
    plt.savefig('graphics/coh_%s_dev.png' % (fnamecode) )
  else:
    plt.savefig('graphics/coh_%s_test.png' % (fnamecode) )

def draw_trans_con_matrix(conmat, ugraph, fnamecode, vmin=None, vmax=None, method = 'coherence'):
  #draws the connectivity matrix after overlaying an alpha filter based on the p-values in the ugraph
  if vmin == None:
    vmin = min(conmat.ravel())
  if vmax == None:
    vmax = max(conmat.ravel())

  n_rows, n_cols = conmat.shape[:2]

  #fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=(15,15))
  #plt.rc('text', usetex=True)
  #plt.rc('font', family='serif')
  
  fig, ax = plt.subplots(figsize=(3.,3.))
  im = ax.imshow(mycmap(conmat[1,0],ugraph), vmin = vmin, vmax = vmax, aspect = 'auto')
  #plt.suptitle('Sensor coherence')
  #for i in range(n_rows):
  #  for j in range(i+1):
  #    if i == j:
  #      axes[i, j].set_axis_off()
  #      continue
  #
  #    cax = axes[i, j].imshow(conmat[i, j, :], vmin=vmin, vmax = vmax, aspect = 'auto') #, interpolation='nearest')
  if epochStart < 0:
    ax.axvline(x=epochStart *-1 * samplingRate, c='black', lw=1)
  #else: onset isn't on graph

  #ax.set_ylabel('Frequency (Hz)')
  ax.set_yticks(numpy.arange(len(cwt_frequencies))[::3])
  ax.set_yticklabels(cwt_frequencies[::3])
  #ax.set_title(names[1])
  #ax.set_xlabel('Time (s)')
  ax.set_xticks(numpy.arange(0.0,conmat.shape[3],62.5))
  ax.set_xticklabels(numpy.arange(epochStart,epochEnd,(epochEnd-epochStart)/(conmat.shape[3]/62.5)))
  ax.set_ylim([0.0, conmat.shape[2]-1])
  
  cb = plt.colorbar(im, format='%1.2g')
  tick_locator = ticker.MaxNLocator(nbins=7)
  cb.locator = tick_locator
  cb.update_ticks()
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
  for i in xrange(0,total/step):
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
  for i in xrange(num-1):
    for j in xrange(i+1,num):
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
(megFileTag1, megFile1) = ('V_TSSS_0.01-50Hz_@125', '../MEG_data/v_hod_allRuns_tsss_audiobookPrepro_stPad1_lp50_resamp125_frac10ICAed_hp0.010000.set')
(megFileTag2, megFile2) = ('A_TSSS_0.01-50Hz_@125', '../MEG_data/aud_hofd_a_allRuns_tsss_audiobookPrepro_stPad1_lp50_resamp125_frac10ICAed_hp0.010000.set')
(megFileTag3, megFile3) = ('C_TSSS_0.01-50Hz_@125', '../MEG_data/aud_hofd_c_allRuns_tsss_audiobookPrepro_stPad1_lp50_resamp125_frac10ICAed_hp0.010000.set')

## put your own properties in here, as a .tab file of similar format (tab delimited, and field names in a first comment line - should be easy to do in excel...)
#to get the V7.tab:
#  python ../scripts/buildtab.py hod_JoeTimes_LoadsaFeaturesV3.tab hod.wsj02to21-comparativized-gcg15-1671-4sm.fullberk.parsed.gcgbadwords > hod_JoeTimes_LoadsaFeaturesV4.tab
#  python ../scripts/expandtab.py hod_JoeTimes_LoadsaFeaturesV4.tab > hod_JoeTimes_LoadsaFeaturesV5.tab
#  python ../scripts/addparserops.py hod_JoeTimes_LoadsaFeaturesV5.tab hod.wsj02to21-comparativized-gcg15-1671-4sm.fullberk.parsed.gcgparseops > hod_JoeTimes_LoadsaFeaturesV6.tab
#  python ../scripts/addtotsurp.py hod_JoeTimes_LoadsaFeaturesV6.tab hod.totsurp > hod_JoeTimes_LoadsaFeaturesV7.tab
#  V8 just adds filler-gap-sensitive surprisal, so run the fg parser and rerun the totsurp script (after changing the header name)
#  python ../scripts/addmorphs.py hod_JoeTimes_LoadsaFeaturesV8.tab hod.pretty_morph_annotations > hod_JoeTimes_LoadsaFeaturesV9.tab
tokenPropsFile = '../MEG_data/hod_JoeTimes_LoadsaFeaturesV9.tab'

# LOAD WORD PROPS
# change dtype to suit the files in your .tab file #
tokenProps = scipy.genfromtxt(tokenPropsFile,
                              delimiter='\t',names=True,
                              dtype="i4,f4,f4,S50,S50,i2,i2,i2,S10,f4,f4,f4,f4,f4,f4,f4,f4,f4,f4,f4,i4,>i4,i4,i4,S4,i8,i8,i4")
# ... and temporarily save as cpickle archive to satisfy the way I programmed the convenience function loadBookMEGWithAudio (it expects to find the same info in a C-pickle file, and so doesn't need to know about number and type of fields)
tokenPropsPickle = tokenPropsFile+'.cpk'
pickle.dump(tokenProps, open(tokenPropsPickle, 'wb'))

triggersOfInterest=['s%d' % i for i in xrange(1,10)]
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
m1TrialsBool = numpy.array([m == 1 for m in tokenProps['morphs']])
m2TrialsBool = numpy.array([m == 2 for m in tokenProps['morphs']])
m3TrialsBool = numpy.array([m == 3 for m in tokenProps['morphs']])
mmTrialsBool = numpy.array([m > 1 for m in tokenProps['morphs']])

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

channelsOfInterest = [i for i in xrange(len(metaData1.chanlocs)) if metaData1.chanlocs[i].labels in channelLabels]

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

wordEpochs = epochedSignalData[wordTrialsBool & inDataset]
m1wordEpochs = epochedSignalData[wordTrialsBool & m1TrialsBool & inDataset]
m2wordEpochs = epochedSignalData[wordTrialsBool & m2TrialsBool & inDataset]
m3wordEpochs = epochedSignalData[wordTrialsBool & m3TrialsBool & inDataset]
mmwordEpochs = epochedSignalData[wordTrialsBool & mmTrialsBool & inDataset]

#conjEpochs = epochedSignalData[wordTrialsBool & parsedTrialsBool & conjBool & inDataset]
#notConjEpochs = epochedSignalData[wordTrialsBool & parsedTrialsBool & notConjBool & inDataset]

if DEPTH:
  testEpochs1 = m1wordEpochs
  testEpochsm = mmwordEpochs
  if TEST3:
    testEpochs2 = m2wordEpochs
else:
  raise
#  testEpochs1 = notConjEpochs
#  testEpochsm = conjEpochs
#  testEpochs2 = None

if FUDGE:
  testEpochs1 = testEpochs1[:40]
  testEpochsm = testEpochsm[:40]
  if TEST3:
    testEpochs2 = testEpochs2[:40]
  
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

print 'testEpochs1',testEpochs1.shape
print 'testEpochsm',testEpochsm.shape
if TEST3:
  print 'testEpochs2',testEpochs2.shape

m1wcon = word_connectivity(testEpochs1, indices=indices, step = coherence_step)
mmwcon = word_connectivity(testEpochsm, indices=indices, step = coherence_step)
if TEST3:
  m2wcon = word_connectivity(testEpochs2, indices=indices, step = coherence_step)
print 'm1wcon',m1wcon.shape
print 'mmwcon',mmwcon.shape
if TEST3:
  print 'm2wcon',m2wcon.shape

m1mcon = numpy.mean(m1wcon,axis=0)
mmmcon = numpy.mean(mmwcon,axis=0)
if TEST3:
  m2mcon = numpy.mean(m2wcon,axis=0)

m1simplea = numpy.empty((m1wcon.shape[0],len(cwt_frequencies)))
m2simplea = numpy.empty((mmwcon.shape[0],len(cwt_frequencies)))
if TEST3:
  m3simplea = numpy.empty((m2wcon.shape[0],len(cwt_frequencies)))
m1simple = numpy.empty((m1wcon.shape[0],len(cwt_frequencies)))
m2simple = numpy.empty((mmwcon.shape[0],len(cwt_frequencies)))
if TEST3:
  m3simple = numpy.empty((m2wcon.shape[0],len(cwt_frequencies)))

#wordEpochs x chans x chans x freqs x timeSamples

#print numpy.mean(d1wcon[0,0,0,0])
#print numpy.mean(d1wcon[0,0,1,0])
#print d1wcon[0,1,0,0,0:10]
#print d1wcon[1,1,0,0,0:10]
#print d1wcon[2,1,0,0,0:10]
#print mmwcon[0,1,0,0,0:10]
#print m2wcon[0,1,0,0,0:10]
#print numpy.mean(d1wcon[0,1,0,0])
#print numpy.mean(d1wcon[0,1,1,0])

for i in xrange(m1simple.shape[0]):
  for fi in xrange(m1simple.shape[1]):
    m1simplea[i,fi] = numpy.mean(m1wcon[i,1,0,fi,tmin:tmax])
for i in xrange(m2simple.shape[0]):
  for fi in xrange(m2simple.shape[1]):
    m2simplea[i,fi] = numpy.mean(mmwcon[i,1,0,fi,tmin:tmax])
if TEST3:
  for i in xrange(m3simple.shape[0]):
    for fi in xrange(m3simple.shape[1]):
      m3simplea[i,fi] = numpy.mean(m2wcon[i,1,0,fi,tmin:tmax])
if plusminus != 0:
  #average over multiple frequencies
  for i in xrange(m1simple.shape[0]):
    for fi in xrange(m1simple.shape[1]):
      m1simple[i,fi] = numpy.mean(m1simplea[i,max(0,fi-plusminus):min(m1simple.shape[1],fi+plusminus+1)])
  for i in xrange(m2simple.shape[0]):
    for fi in xrange(m2simple.shape[1]):
      m2simple[i,fi] = numpy.mean(m2simplea[i,max(0,fi-plusminus):min(m2simple.shape[1],fi+plusminus+1)])
  if TEST3:
    for i in xrange(m3simple.shape[0]):
      for fi in xrange(m3simple.shape[1]):
        m3simple[i,fi] = numpy.mean(m3simplea[i,max(0,fi-plusminus):min(m3simple.shape[1],fi+plusminus+1)])
else:
  m1simple = m1simplea
  m2simple = m2simplea
  if TEST3:
    m3simple = m3simplea

conpkg = {'conmats':[],'freqs':freqs,'electrodes':channelLabels}

tgraph2 = numpy.empty((len(cwt_frequencies),))
ugraph2 = numpy.empty((len(cwt_frequencies),))
if TEST3:
  tgraph3 = numpy.empty((len(cwt_frequencies),))
  ugraph3 = numpy.empty((len(cwt_frequencies),))

if SAVEDATA:
  print 'Writing coherence metrics to disk'

  conpkg['conmats'] = m1wcon
  with open(SAVEDIR+'cohm1m.pkl','wb') as f:
    pickle.dump(conpkg,f)
  conpkg['conmats'] = mmwcon
  with open(SAVEDIR+'cohm2m.pkl','wb') as f:
    pickle.dump(conpkg,f)
  if TEST3:
    conpkg['conmats'] = m2wcon
    with open(SAVEDIR+'cohm3m.pkl','wb') as f:
      pickle.dump(conpkg,f)

if DEV:
  for fi in xrange(len(cwt_frequencies)):
    tgraph2[fi] = abs(scipy.stats.f_oneway(m1simple[:,fi],m2simple[:,fi])[0])
    ugraph2[fi] = abs(scipy.stats.mannwhitneyu(m1simple[:,fi],m2simple[:,fi])[1])
    if TEST3:
      tgraph3[fi] = abs(scipy.stats.f_oneway(m2simple[:,fi],m3simple[:,fi])[0])
      ugraph3[fi] = abs(scipy.stats.mannwhitneyu(m2simple[:,fi],m3simple[:,fi])[1])
if OPTIMAL:
  if TEST2:
    for col in numpy.where(ugraph2 < 0.05)[0].ravel():
      print 'm2-m1:', cwt_frequencies[col], 'Hz:', numpy.mean(m2simple,axis=0)[col],'-', numpy.mean(m1simple,axis=0)[col],'p=', scipy.stats.f_oneway(m1simple[:,col],m2simple[:,col]), 'u=',scipy.stats.mannwhitneyu(m1simple[:,col],m2simple[:,col])
      print 'variance:',numpy.var(numpy.concatenate((m1simple[:,col],m2simple[:,col]),axis=0)), 'stddev:',numpy.std(numpy.concatenate((m1simple[:,col],m2simple[:,col]),axis=0)), 'stddevA:',numpy.std(m2simple[:,col]), 'stddevB:',numpy.std(m1simple[:,col])
  if TEST3:
    for col in numpy.where(ugraph3 < 0.05)[0].ravel():
      print 'm3-m2:', cwt_frequencies[col], 'Hz:', numpy.mean(m3simple,axis=0)[col],'-', numpy.mean(m2simple,axis=0)[col],'p=', scipy.stats.f_oneway(m2simple[:,col],m3simple[:,col]),'u=', scipy.stats.mannwhitneyu(m2simple[:,col],m3simple[:,col])
else:
  for col in GOODFREQS-fmin:
    if TEST2:
      if DEV or (not DEV and col in GOODFREQS2-fmin):
        print 'm2-m1:', cwt_frequencies[col], 'Hz:', numpy.mean(m2simple,axis=0)[col],'-', numpy.mean(m1simple,axis=0)[col],'p=', scipy.stats.f_oneway(m1simple[:,col],m2simple[:,col]), 'u=',scipy.stats.mannwhitneyu(m1simple[:,col],m2simple[:,col])
        print 'variance:',numpy.var(numpy.concatenate((m1simple[:,col],m2simple[:,col]),axis=0)), 'stddev:',numpy.std(numpy.concatenate((m1simple[:,col],m2simple[:,col]),axis=0)), 'stddevA:',numpy.std(m2simple[:,col]), 'stddevB:',numpy.std(m1simple[:,col])
        if CHECK_NORMALITY:
          scipy.stats.probplot(m1simple[:,col], dist="norm", plot=plt)
          plt.savefig('graphics/norm1.png')
          plt.close("all")
          scipy.stats.probplot(m2simple[:,col], dist="norm", plot=plt)
          plt.savefig('graphics/norm2.png')
          plt.close("all")
    if TEST3:
      if DEV or (not DEV and col in GOODFREQS3-fmin):
        print 'm3-m2:', cwt_frequencies[col], 'Hz:', numpy.mean(m3simple,axis=0)[col],'-', numpy.mean(m2simple,axis=0)[col],'p=', scipy.stats.f_oneway(m2simple[:,col],m3simple[:,col]),'u=', scipy.stats.mannwhitneyu(m2simple[:,col],m3simple[:,col])
        print 'variance:',numpy.var(numpy.concatenate((m1simple[:,col],m2simple[:,col]),axis=0)), 'stddev:',numpy.std(numpy.concatenate((m1simple[:,col],m2simple[:,col]),axis=0))
        if CHECK_NORMALITY:
          scipy.stats.probplot(m3simple[:,col], dist="norm", plot=plt)
          plt.savefig('graphics/norm3.png')
          plt.close("all")    
      
##save for later
#with open('m1con.pkl','wb') as f:
#  pickle.dump(m1con,f)
#  
##Depth 2 coherence
#m2con, freqs, times, _, _ = spectral_connectivity(m2wordEpochs, indices=indices,
#                                                  method='coh', mode='cwt_morlet', sfreq=samplingRate,
#                                                  cwt_frequencies=cwt_frequencies, cwt_n_cycles=cwt_n_cycles, n_jobs=NJOBS, verbose='WARNING')
##save for later
#with open('m2con.pkl','wb') as f:
#  pickle.dump(m2con,f)
##Depth 3 coherence
#m3con, freqs, times, _, _ = spectral_connectivity(m3wordEpochs, indices=indices,
#                                                  method='coh', mode='cwt_morlet', sfreq=samplingRate,
#                                                  cwt_frequencies=cwt_frequencies, cwt_n_cycles=cwt_n_cycles, n_jobs=NJOBS, verbose='WARNING')
##save for later
#with open('m3con.pkl','wb') as f:
#  pickle.dump(m3con,f)
#Storage coherence
#m2scon, freqs, times, _, _ = spectral_connectivity(m2swordEpochs, indices=indices,
#                                                  method='coh', mode='cwt_morlet', sfreq=samplingRate,
#                                                  cwt_frequencies=cwt_frequencies, cwt_n_cycles=cwt_n_cycles, n_jobs=NJOBS, verbose='WARNING')
#mmmcon, freqs, times, _, _ = spectral_connectivity(m2mwordEpochs, indices=indices,
#                                                  method='coh', mode='cwt_morlet', sfreq=samplingRate,
#                                                  cwt_frequencies=cwt_frequencies, cwt_n_cycles=cwt_n_cycles, n_jobs=NJOBS, verbose='WARNING')
#print 'freqs',freqs
#m2mcon, freqs, times, _, _ = spectral_connectivity(m3mwordEpochs, indices=indices,
#                                                  method='coh', mode='cwt_morlet', sfreq=samplingRate,
#                                                  cwt_frequencies=cwt_frequencies, cwt_n_cycles=cwt_n_cycles, n_jobs=NJOBS, verbose='WARNING')
#m2icon, freqs, times, _, _ = spectral_connectivity(m2iwordEpochs, indices=indices,
#                                                  method='coh', mode='cwt_morlet', sfreq=samplingRate,
#                                                  cwt_frequencies=cwt_frequencies, cwt_n_cycles=cwt_n_cycles, n_jobs=NJOBS, verbose='WARNING')
#m2ficon, freqs, times, _, _ = spectral_connectivity(m2fiwordEpochs, indices=indices,
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

#vmin = min(min(m1con.ravel()),min(m2con.ravel()),min(m3con.ravel()))
#vmax = max(max(m1con.ravel()),max(m2con.ravel()),max(m3con.ravel()))
#draw_con_matrix(m1con, 'm1', vmin, vmax)
#draw_con_matrix(m2con, 'm2', vmin, vmax)
#draw_con_matrix(m3con, 'm3', vmin, vmax)
#draw_con_matrix(d4con, 'd4', vmin, vmax)

#Do storage and integration analyses
#integcon = m2icon - mmmcon
#fintegcon = m2ficon - mmmcon
#storcon = m2scon - mmmcon

#conpkg = {'conmats':[],'freqs':freqs,'electrodes':channelLabels}
#conpkg['conmats'].append(m32simple)
#conpkg['conmats'].append(m21simple)
#if DEV:
#  pklname = 'con_dev.pkl'
#else:
#  pklname = 'con_test.pkl'
#with open(pklname,'wb') as f:
#    pickle.dump(conpkg,f)

if DRAW:
  if TEST3:
    maintcon3 = m2mcon - m1mcon
  maintcon2 = mmmcon - m1mcon

  vmin = None
  vmax = None
  
  print 'Computing means'
  #+/- 2 Hz, 0-.5s, 25 & 40 Hz
  third = int(maintcon2.shape[3]/3.)
  sixth = int(third * .5)
  twelfth = int(sixth * .5)
  print maintcon2.shape
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
  namemod = str(plusminus)+'_'+str(channelNumbers[0])+'-'+str(channelNumbers[1])
  if DEV:
    if TEST3:
      draw_single_con_matrix(maintcon3, 'heat2'+namemod, vmin, vmax)
      draw_con_matrix(maintcon3, 'heatgrid2'+namemod, vmin, vmax)
      draw_trans_con_matrix(maintcon3, ugraph3, 'heattrans2'+namemod, vmin, vmax)
      draw_tgraph(tgraph3, 'tgraph2'+namemod, 0, None, 't')
      draw_tgraph(ugraph3, 'ugraph2'+namemod, None, 0.05, 'u')
    draw_single_con_matrix(maintcon2, 'heatm'+namemod, vmin, vmax)
    draw_con_matrix(maintcon2, 'heatgridm'+namemod, vmin, vmax)
    draw_trans_con_matrix(maintcon2, ugraph2, 'heattransm'+namemod, vmin, vmax)
    draw_tgraph(tgraph2, 'tgraphm'+namemod, 0, None, 't')
    draw_tgraph(ugraph2, 'ugraphm'+namemod, None, 0.05, 'u')
  #run_ttest((m1wcon,mmwcon,m2wcon),Tchannels,GOODFREQ)
