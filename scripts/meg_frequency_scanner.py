# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# Imports
# =======

# <codecell>

#disable autosave functionality;
#This script is taxing enough, and autosaving tends to push it over the edge
# plus, autosaving seems to zero out the file before restoring it from the backup
# this means an autosave causing a crash will actually delete the file rather than saving it!!!
#%autosave 0

# <codecell>

#basic imports

#%pylab inline
import time
import pickle
import logging as L
L.basicConfig(level=L.ERROR) # INFO)
import time
import numpy
#import #pylab
import scipy.stats
import os
import pylab
import sklearn
import scipy
import sklearn.linear_model
import re

pylab.rcParams['figure.figsize'] = 10,10 #change the default image size for this session
pylab.ion()

# <codecell>

#custom imports
#%cd ../scripts

# brian's prototype routines
from protoMEEGutils import *
import protoSpectralWinFFTMapper as specfft

# <markdowncell>

# Definitions
# ===========

# <codecell>

# OUTLINE, 19th Nov 2014
#
# script for initial "signs of life" analysis of single MEG
#
# load in a meg file in EEGlab format
# load in the word properties
# choose a "languagey" channel (left-temporal, clean, with expected ERF patterns) 
# plot some ERFs (e.g. all nouns vs all preps) as sanity check
# do the earlier analysis of R^2 and betas due to lexical access vars, up to X words back 
# hand over to Marten, so he can do the analysis based on syntactic embedding...


#### SUBROUTINES ####

# plot the time-scale for ERFs and other epoch figures 
def commonPlotProps():
        #zeroSample = (abs(epochStart)/float(epochLength)*epochNumTimepoints)
        #pylab.plot((0,epochNumTimepoints),(0,0),'k--')
        #pylab.ylim((-2.5e-13,2.5e-13)) #((-5e-14,5e-14)) # better way just to get the ones it choose itself?
        #pylab.plot((zeroSample,zeroSample),(0,0.01),'k--')
        pylab.xticks(numpy.linspace(0,epochNumTimepoints,7),epochStart+(numpy.linspace(0,epochNumTimepoints,7)/samplingRate))
        pylab.xlabel('time (s) relative to auditory onset') #+refEvent)
        pylab.xlim((62,313))
        pylab.show()
        pylab.axhline(0, color='k', linestyle='--')
        pylab.axvline(125, color='k', linestyle='--')
        
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

# <markdowncell>

# Preprocessing
# =============

# <markdowncell>

# Input Params
# ----------

# <codecell>

#change to the data directory to load in the data
#%cd ../MEG_data

#*# MARTY #*# choose a file - I found participant V to be pretty good, and 0.01 to 50Hz filter is pretty conservative #*#
(megFileTag1, megFile1) = ('V_TSSS_0.01-50Hz_@125', '../MEG_data/v_hod_allRuns_tsss_audiobookPrepro_stPad1_lp50_resamp125_frac10ICAed_hp0.010000.set')
(megFileTag2, megFile2) = ('A_TSSS_0.01-50Hz_@125', '../MEG_data/aud_hofd_a_allRuns_tsss_audiobookPrepro_stPad1_lp50_resamp125_frac10ICAed_hp0.010000.set')
(megFileTag3, megFile3) = ('C_TSSS_0.01-50Hz_@125', '../MEG_data/aud_hofd_c_allRuns_tsss_audiobookPrepro_stPad1_lp50_resamp125_frac10ICAed_hp0.010000.set')

## put your on properties in here, as a .tab file of similar format (tab delimited, and field names in a first comment line - should be easy to do in excel...)
#to get the V5.tab:
#  python ../scripts/buildtab.py hod_JoeTimes_LoadsaFeaturesV3.tab hod.wsj02to21-comparativized-gcg15-1671-4sm.fullberk.parsed.gcgbadwords > hod_JoeTimes_LoadsaFeaturesV4.tab
#  python ../scripts/addsentid.py hod_JoeTimes_LoadsaFeaturesV4.tab > hod_JoeTimes_LoadsaFeaturesV5.tab
tokenPropsFile = '../MEG_data/hod_JoeTimes_LoadsaFeaturesV5.tab'

# WHICH CHANNELS TO LOOK AT AS ERFS
#*# MARTY #*# decide which channels to use - channels of interest are the first few you can look at in an ERF, and then from them you can choose one at a time with "channelToAnalyse" for the actual regression analysis #*#
#channelLabels = ['MEG0111', 'MEG0121', 'MEG0131', 'MEG0211', 'MEG0212', 'MEG0213', 'MEG0341']
#?# this way of doing things was a slightly clumsy work-around, cos I didn't have enough memory to epoch all 306 channels at one time

# LOAD WORD PROPS
#*# MARTY #*# change dtype to suit the files in your .tab file #*#
tokenProps = scipy.genfromtxt(tokenPropsFile,
                              delimiter='\t',names=True,
                              dtype="i4,f4,f4,S50,S50,i2,i2,i2,S10,f4,f4,f4,f4,f4,f4,f4,f4,f4,f4,f4,i1,>i4")
# ... and temporarily save as cpickle archive to satisfy the way I programmed the convenience function loadBookMEGWithAudio (it expects to find the same info in a C-pickle file, and so doesn't need to know about number and type of fields)
tokenPropsPickle = tokenPropsFile+'.cpk'
cPickle.dump(tokenProps, open(tokenPropsPickle, 'wb'))

# <markdowncell>

# Trial Params
# ------------

# <codecell>

triggersOfInterest=['s%d' % i for i in range(1,10)]
refEvent = 'onTime' #,'offTime']
#*# MARTY #*# guess an epoch of -0.5 to +1s should be enough #*#
epochStart = -1; # stimulus ref event
epochEnd = +2; # 
epochLength = epochEnd-epochStart;
baseline = False #[-1,0]

# <markdowncell>

# Epoch Data
# ----------

# <codecell>

# Get the goods on subject 1
(contSignalData1, metaData1, trackTrials, tokenPropsOrig, audioSignal, samplingRate, numChannels) = loadBookMEGWithAudio(megFile1, tokenPropsPickle, triggersOfInterest, epochEnd, epochStart, icaComps=False)

# Get the goods on subject 2
(contSignalData2, metaData2, trackTrials, tokenPropsOrig, audioSignal, samplingRate, numChannels) = loadBookMEGWithAudio(megFile2, tokenPropsPickle, triggersOfInterest, epochEnd, epochStart, icaComps=False)

# Get the goods on subject 3
(contSignalData3, metaData3, trackTrials, tokenPropsOrig, audioSignal, samplingRate, numChannels) = loadBookMEGWithAudio(megFile3, tokenPropsPickle, triggersOfInterest, epochEnd, epochStart, icaComps=False)

tokenProps = numpy.concatenate((tokenPropsOrig,tokenPropsOrig,tokenPropsOrig),axis=0)

#channelsOfInterest = [i for i in range(len(metaData1.chanlocs)) if metaData1.chanlocs[i].labels in channelLabels]

# REDUCE TRIALS TO JUST THOSE THAT CONTAIN A REAL WORD (NOT PUNCTUATION, SPACES, ...)
wordTrialsBool = numpy.array([p != '' for p in tokenProps['stanfPOS']])
#print(wordTrialsBool[:10])
# REDUCE TRIALS TO JUST THOSE THAT HAVE A DECENT DEPTH ESTIMATE
parsedTrialsBool = numpy.array([d != -1 for d in tokenProps['syndepth']])
#print(parsedTrialsBool[:10])

# <codecell>

# Set up the dev and test sets
devsizerecip = 3 # the reciprocal of the dev size, so devsizerecip = 3 means the dev set is 1/3 and the test set is 2/3
devitems = numpy.arange(1,max(tokenProps['sentid']),devsizerecip)
devTrialsBool = numpy.array([s in devitems for s in tokenProps['sentid']])
testTrialsBool = numpy.array([s not in devitems for s in tokenProps['sentid']])

inDataset = devTrialsBool
freqsource = None # determines how the frequency bands are defined #Can be 'weiss', 'wiki', or an interpolation of the two 


for channelix in range(metaData1.chanlocs):
  severalMagChannels1 = contSignalData1[channelix,:]
  (wordTrials1, epochedSignalData1, epochSliceTimepoints, wordTimesAbsolute, numTrials, epochNumTimepoints) = wordTrialEpochify(severalMagChannels1, samplingRate, tokenPropsOrig, trackTrials, refEvent, epochEnd, epochStart)

# <codecell>
#del contSignalData1
#del severalMagChannels1

# <codecell>

  severalMagChannels2 = contSignalData2[channelix,:]
  (wordTrials2, epochedSignalData2, epochSliceTimepoints, wordTimesAbsolute, numTrials, epochNumTimepoints) = wordTrialEpochify(severalMagChannels2, samplingRate, tokenPropsOrig, trackTrials, refEvent, epochEnd, epochStart)

# <codecell>

#del contSignalData2
#del severalMagChannels2

# <codecell>

  severalMagChannels3 = contSignalData3[channelix,:]
  (wordTrials3, epochedSignalData3, epochSliceTimepoints, wordTimesAbsolute, numTrials, epochNumTimepoints) = wordTrialEpochify(severalMagChannels3, samplingRate, tokenPropsOrig, trackTrials, refEvent, epochEnd, epochStart)

# <codecell>

#del contSignalData3
#del severalMagChannels3

# <codecell>

  epochedSignalData = numpy.concatenate((epochedSignalData1,epochedSignalData2,epochedSignalData3), axis=0)
  #print(epochedSignalData.shape)
  #print(tokenProps.shape)
  sys.stderr.write('epochedSignalData.shape: '+str(epochedSignalData.shape)+'\n')
  raise #need to determine whether we need to reshape with a center column or retain two dimensional...?

# <codecell>

#print tokenProps.shape,epochedSignalData.shape
  wordEpochs = epochedSignalData[wordTrialsBool & parsedTrialsBool & inDataset] #NB: Might not be 
  wordFeatures = tokenProps[wordTrialsBool & parsedTrialsBool & inDataset]
#print wordFeatures.shape, wordEpochs.shape

# <markdowncell>

# Spectral decomposition
# ---------------------

# <codecell>

#test reshape outcomes
#a = np.arange(18).reshape((3,2,3))
#print(a) #target: 3 epochs, 2 channels, 3 frequency_features
#b = np.arange(18).reshape((3,6))
#print(b) #fft output: 3 epochs, 2 channels x 3 frequency_features
#c = b.reshape((3,2,3))
#print(c) #reshaped output: 3 epochs, 2 channels, 3 frequency_features

# <codecell>

# The FFT script collapses across channels
# index0: epoch
# index1: channels x fft_feature_types x frequencies
# solution: reshape the output as (epochs,channels,-1)

#print 'wordEpochs: ',wordEpochs.shape
# Spectrally decompose the epochs
  (mappedTrialFeatures, spectralFrequencies) = specfft.mapFeatures(wordEpochs,samplingRate,windowShape='hann',featureType='amp')
# Reshape output to get epochs x channels x frequency
  mappedTrialFeatures = mappedTrialFeatures.reshape((wordEpochs.shape[0],wordEpochs.shape[1],-1))
#print 'FFT output: ', mappedTrialFeatures.shape, spectralFrequencies.shape

# <codecell>

#print(spectralFrequencies)

  if freqsource == 'weiss':
      #Weiss et al. 05
      theta = numpy.nonzero( (spectralFrequencies >= 4) & (spectralFrequencies <= 7) )
      beta1 = numpy.nonzero( (spectralFrequencies >= 13) & (spectralFrequencies <= 18) )
      beta2 = numpy.nonzero( (spectralFrequencies >= 20) & (spectralFrequencies <= 28) )
      gamma = numpy.nonzero( (spectralFrequencies >= 30) & (spectralFrequencies <= 34) )
  elif freqsource == 'wiki':
      # end of http://en.wikipedia.org/wiki/Theta_rhythm
      delta = numpy.nonzero( (spectralFrequencies >= 0.1) & (spectralFrequencies <= 3) )
      theta = numpy.nonzero( (spectralFrequencies >= 4) & (spectralFrequencies <= 7) )
      alpha = numpy.nonzero( (spectralFrequencies >= 8) & (spectralFrequencies <= 15) )
      beta = numpy.nonzero( (spectralFrequencies >= 16) & (spectralFrequencies <= 31) )
      gamma = numpy.nonzero( (spectralFrequencies >= 32) & (spectralFrequencies <= 100) )
  else:
      #Interpolate between weiss and wiki
      #print(numpy.nonzero((spectralFrequencies >= 4) & (spectralFrequencies <= 7)))
      theta = numpy.nonzero( (spectralFrequencies >= 4) & (spectralFrequencies <= 7) )
      alpha = numpy.nonzero( (spectralFrequencies >= 8) & (spectralFrequencies < 13) )
      beta1 = numpy.nonzero( (spectralFrequencies >= 13) & (spectralFrequencies <= 18) )
      beta2 = numpy.nonzero( (spectralFrequencies >= 20) & (spectralFrequencies <= 28) )
      gamma = numpy.nonzero( (spectralFrequencies >= 30) & (spectralFrequencies <= 34) )
#print(theta)
#print(theta[0])

# <markdowncell>

# Select channel for analysis
# --------------

# <codecell>

#print(channelLabels)
#channelToAnalyse = 3 # index of the channels above to actually run regression analysis on
#print 'Analyzing:', channelLabels[channelToAnalyse]

# <markdowncell>

# Run Regression Analysis
# ---------------------------

# <codecell>

# REGULARISATION VALUES TO TRY (e.g. in Ridge GCV)
regParam = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 1e+2, 2e+2, 5e+2, 1e+3, 2e+3, 5e+3]

# SELECT AND DESCRIBE THE REGRESSORS WE'RE CHOOSING TO USE
# this strings should match the names of the fields in tokenProps
#*# MARTY #*# here you should list the features you're choosing from your .tab file (just one?) #*#
features = [
 #'logFreq_ANC',
 #'surprisal2back_COCA',
 #'bigramEntropy_COCA_here',
 'syndepth'
]
#*# MARTY #*# ... this has shorthand versions of the variable names, for display, and also has to include the "position" one that this version of the script inserts by default #*#
labelMap = {
 #'logFreq_ANC': 'freq',
 #'surprisal2back_COCA': 'surprisal',
 #'bigramEntropy_COCA_here': 'entropy',
 #'sentenceSerial': 'position',
 'syndepth': 'depth'
}
legendLabels = features

# <codecell>

# SLOT REGRESSORS IN ONE BY ONE
explanatoryFeatures = numpy.zeros((wordFeatures.shape)) # dummy
#explanatoryFeatures = numpy.array([])
for feature in features:
        print feature
        explanatoryFeatures = numpy.vstack((explanatoryFeatures, wordFeatures[feature]))
explanatoryFeatures = explanatoryFeatures[1:].T # strip zeros out again

# PLOT EFFECTS X EPOCHS BACK
#*# MARTY #*# I guess you don't want to do the history thing (though is good initially for sanity check), so can leave this at 0 #*#
epochHistory = 0

# <codecell>

modelTrainingFit = []
modelTestCorrelation = []
modelParameters = []
legendLabels = features
#tmpFeatures = explanatoryFeatures.copy()
#tmpLegend = legendLabels[:]
#for epochsBack in range(1,epochHistory+1):
#        epochFeatures = numpy.zeros(tmpFeatures.shape)
#        epochFeatures[epochsBack:,:] = tmpFeatures[:-epochsBack,:]
#        explanatoryFeatures = numpy.hstack((explanatoryFeatures,epochFeatures))
#        legendLabels = legendLabels + [l+'-'+str(epochsBack) for l in tmpLegend]

## put in sentence serial - can't leave in history, cos is too highly correlated across history...
#explanatoryFeatures = numpy.vstack((explanatoryFeatures.T, wordFeatures['sentenceSerial'])).T
#features.append('sentenceSerial')
#legendLabels.append('sentenceSerial')

# <codecell>

# STEP THROUGH EACH TIME POINT IN THE EPOCH, RUNNING REGRESSION FOR EACH ONE
for t in theta[0]:#range(1):
        #print 'fitting at timepoint',t
        # NOTES # tried a load of different versions, and straight linear regression does as well as any of them, measured in terms of R^2

        # WHICH VARIETY OF REGRESSION TO USE?
        #*# MARTY #*# I get pretty similar results with all three of those below. The most generic (ie fewest extra assumptions) is normal LinearRegression. I guess RidgeCV should do best in terms of R^2, but has discontinuities in betas, as different regularisation parameters are optimal at each time step. LassoLars is something of a compromise. #*#
        #lm = sklearn.linear_model.LinearRegression(fit_intercept=True, normalize=True)
        #lm = sklearn.linear_model.RidgeCV(fit_intercept=True, normalize=True, alphas=regParam) #, 10000, 100000])
        lm = sklearn.linear_model.LassoLars(alpha=0.0001) #(alpha=1.0, fit_intercept=True, verbose=False, normalize=True, precompute='auto', max_iter=500, eps=2.2204460492503131e-16, copy_X=True)

        # NORMALISE THE EXPLANATORY VARIABLES? (for comparable beta magnitude interpretation)
        #*# MARTY #*# choose whether to scale inputs #*#
        trainX = mynormalise(explanatoryFeatures)
        trainY = mynormalise(mappedTrialFeatures[:,channelToAnalyse,t])
        #trainX = mynormalise(explanatoryFeatures)
        #trainY = mynormalise(wordEpochs[:,channelToAnalyse,t])
        #trainX = explanatoryFeatures
        #trainY = wordEpochs[:,channelToAnalyse,t]

        trainedLM = lm.fit(trainX,trainY)
        modelParameters.append(lm)
        #print(lm.score(trainX,trainY),trainX.shape[1], trainX.shape[0])
        #modelTrainingFit.append(adjustR2(lm.score(trainX,trainY), trainX.shape[1], trainX.shape[0]))
        modelTrainingFit.append(lm.score(trainX,trainY)) #for a single feature, no punishment is necessary

# <codecell>

print(modelTrainingFit)
print(numpy.sort(modelTrainingFit)[::-1])
print 'ave fit: ', numpy.mean(modelTrainingFit)
print 'max fit: ', numpy.max(modelTrainingFit)

# <markdowncell>

# Graph results
# ============

# <codecell>

# DETERMINE IF THERE IS CORRELATION BETWEEN THE EXPLANATORY VARIABLES
betaMatrix = numpy.array([p.coef_ for p in modelParameters])
print(betaMatrix.shape)
neatLabels = [l.replace(re.match(r'[^-]+',l).group(0), labelMap[re.match(r'[^-]+',l).group(0)]) for l in legendLabels if re.match(r'[^-]+',l).group(0) in labelMap]
legendLabels = numpy.array(legendLabels)
#numFeaturesDisplay = len(legendLabels)
neatLabels = numpy.array(neatLabels)

# <codecell>

# DO BIG SUMMARY PLOT OF FEATURE CORRELATIONS, R^2 OVER TIMECOURSE, BETAS OVER TIME COURSE, AND ERF/ERP
f = pylab.figure(figsize=(10,10))
s = pylab.subplot(2,2,1)
pylab.title('R-squared '+str(trainedLM))
pylab.plot(modelTrainingFit, linewidth=2)
commonPlotProps()
s = pylab.subplot(2,2,2)
if betaMatrix.shape[1] > 7:
        pylab.plot(betaMatrix[:,:7], '-', linewidth=2)
        pylab.plot(betaMatrix[:,7:], '--', linewidth=2)
else:
        pylab.plot(betaMatrix, '-', linewidth=2)

pylab.legend(neatLabels)
#pylab.legend(legendLabels)
pylab.title('betas for all (normed) variables')
commonPlotProps()


s = pylab.subplot(3,3,2)
pylab.title('correlations between explanatory variables')
pylab.imshow(numpy.abs(numpy.corrcoef(explanatoryFeatures.T)),interpolation='nearest', origin='upper') # leave out the dummy one
pylab.clim(0,1)
pylab.yticks(range(len(neatLabels)),neatLabels)
pylab.ylim((-0.5,len(neatLabels)-0.5))
pylab.xticks(range(len(neatLabels)),neatLabels, rotation=90)
pylab.xlim((-0.5,len(neatLabels)-0.5))
pylab.colorbar()

#fontP = FontProperties()
#fontP.set_size('small')
#legend([s], "title", prop = fontP)

#s = pylab.subplot(2,2,4)
#pylab.plot(numpy.mean(epochedSignalData[wordTrialsBool,channelToAnalyse],axis=0).T, linewidth=2)
#pylab.title('ERF')
#commonPlotProps()

#print 'history %d, mean model fit over -0.5s to +1.0s: %.5f, max is %.5f' % (epochHistory, numpy.mean(modelTrainingFit[62:250]), numpy.max(modelTrainingFit[62:250]))
#pylab.savefig('meg_testfig_%s.png' % (channelLabels[channelToAnalyse]))

pylab.show()

# <codecell>


