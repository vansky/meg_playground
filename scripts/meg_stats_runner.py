# -*- coding: utf-8; python-indent: 2; -*-
# <nbformat>3.0</nbformat>
# This script extracts P-values and R^2 values over each frequency band and determines model fits

# <markdowncell>

# Imports
# =======

# <codecell>

#disable autosave functionality;
#This script is taxing enough, and autosaving tends to push it over the edge
# plus, autosaving seems to zero out the file before restoring it from the backup
# this means an autosave causing a crash will actually delete the file rather than saving it!!!
#%autosave 0


#basic imports
DEV = True
CLUSTER = True
COMBINE_SUBJS = False

#%pylab inline
import time
import pickle
import logging as L
L.basicConfig(level=L.ERROR) # INFO)
import time
import numpy
import scipy.stats
import pandas as pd
import os
#import pylab
import sklearn
from sklearn import preprocessing
import scipy
import sklearn.linear_model
import sys
import re
#import statsmodels.api as sm
import statsmodels.formula.api as smf

#pylab.rcParams['figure.figsize'] = 10,10 #change the default image size for this session
#pylab.ion()

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
#def commonPlotProps():
        #zeroSample = (abs(epochStart)/float(epochLength)*epochNumTimepoints)
        #pylab.plot((0,epochNumTimepoints),(0,0),'k--')
        #pylab.ylim((-2.5e-13,2.5e-13)) #((-5e-14,5e-14)) # better way just to get the ones it choose itself?
        #pylab.plot((zeroSample,zeroSample),(0,0.01),'k--')
#        pylab.xticks(numpy.linspace(0,epochNumTimepoints,7),epochStart+(numpy.linspace(0,epochNumTimepoints,7)/samplingRate))
#        pylab.xlabel('time (s) relative to auditory onset') #+refEvent)
#        pylab.xlim((62,313))
#        pylab.show()
#        pylab.axhline(0, color='k', linestyle='--')
#        pylab.axvline(125, color='k', linestyle='--')
        
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

#choose a file - I found participant V to be pretty good, and 0.01 to 50Hz filter is pretty conservative #
(megFileTag1, megFile1) = ('V_TSSS_0.01-50Hz_@125', '../MEG_data/v_hod_allRuns_tsss_audiobookPrepro_stPad1_lp50_resamp125_frac10ICAed.set')#_hp0.010000.set')
(megFileTag2, megFile2) = ('A_TSSS_0.01-50Hz_@125', '../MEG_data/aud_hofd_a_allRuns_tsss_audiobookPrepro_stPad1_lp50_resamp125_frac10ICAed_hp0.010000.set')
(megFileTag3, megFile3) = ('C_TSSS_0.01-50Hz_@125', '../MEG_data/aud_hofd_c_allRuns_tsss_audiobookPrepro_stPad1_lp50_resamp125_frac10ICAed_hp0.010000.set')

## put your on properties in here, as a .tab file of similar format (tab delimited, and field names in a first comment line - should be easy to do in excel...)
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
cPickle.dump(tokenProps, open(tokenPropsPickle, 'wb'))

# <markdowncell>

# Trial Params
# ------------

# <codecell>

triggersOfInterest=['s%d' % i for i in range(1,10)]
refEvent = 'onTime' #,'offTime']
# guess an epoch of -0.5 to +1s should be enough #
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

NUMSUBJS = 3
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

if DEV:
  inDataset = devTrialsBool
else:
  inDataset = testTrialsBool
freqsource = None # determines how the frequency bands are defined #Can be 'weiss', 'wiki', or an interpolation of the two 

if CLUSTER: #??!!NB: clustering goes here
  clustershape = (3,3) #bottom 6 sectors
  xdivs = np.arange(clustershape[0])
  ydivs = np.arange(clustershape[1])

  #find xmax/min and ymax/min in first loop
  minx = None
  maxx = None
  miny = None
  maxy = None
  for sensor in metaData1.chanlocs:
    #need to remap what X and Y are from the EEGLab defaults
    thisx = -sensor.Y
    thisy = sensor.X
    if not minx or thisx < minx:
      minx = thisx
    if not maxx or thisx > maxx:
      maxx = thisx
    if not miny or thisy < miny:
      miny = thisy
    if not maxy or thisy > maxy:
      maxy = thisy

  xdivs *= (xmax+xmin)/float(clustershape[0])
  ydivs *= (ymax+ymin)/float(clustershape[1])
  #assign sensor labels to clusters in second loop
  sensorCluster = {}
  for sensorix,sensor in enumerate(metaData1.chanlocs):
    thisx = -sensor.Y
    thisy = sensor.X
    xpos = 0
    ypos = 0
    for i,x in enumerate(xdivs[1:]):
      #find the xdim of the cluster
      if thisx <= x:
        xpos = i
        break
    for i,y in enumerate(ydivs[1:]):
      #find the ydim of the cluster
      if thisy <= y:
        ypos = i
        break
    sensorCluster[sensorix] = (xpos, ypos) 

fitresults = {}
#for i in range(NUMSUBJS):
#  fitresults[i] = {}
avefitsig_dict = {}
avefit_dict = {}
#lm_dict = {}
#df_dict = {}
#maxfitresults = {}
freqsY = None
trainX = None
clusterpower = {}
clustersize = {}
for channelix in range(metaData1.chanlocs.shape[0]-1): #minus 1 because the last 'channel' is MISC
  print 'Compiling data from channel:',channelix
  #need to reshape because severalMagChannels needs to be channel x samples, and 1-D shapes are flattened by numpy
  severalMagChannels1 = contSignalData1[channelix,:].reshape((1,-1))

#  channelLabels = ['MEG0111', 'MEG0121', 'MEG0131', 'MEG0211', 'MEG0212', 'MEG0213', 'MEG0341']

  (wordTrials1, epochedSignalData1, epochSliceTimepoints, wordTimesAbsolute, numTrials, epochNumTimepoints) = wordTrialEpochify(severalMagChannels1, samplingRate, tokenPropsOrig, trackTrials, refEvent, epochEnd, epochStart)

  severalMagChannels2 = contSignalData2[channelix,:].reshape((1,-1))
  (wordTrials2, epochedSignalData2, epochSliceTimepoints, wordTimesAbsolute, numTrials, epochNumTimepoints) = wordTrialEpochify(severalMagChannels2, samplingRate, tokenPropsOrig, trackTrials, refEvent, epochEnd, epochStart)

  severalMagChannels3 = contSignalData3[channelix,:].reshape((1,-1))
  (wordTrials3, epochedSignalData3, epochSliceTimepoints, wordTimesAbsolute, numTrials, epochNumTimepoints) = wordTrialEpochify(severalMagChannels3, samplingRate, tokenPropsOrig, trackTrials, refEvent, epochEnd, epochStart)

# <codecell>

  if True:
#  if COMBINE_SUBJS:
    epochedSignalData = ( numpy.concatenate((epochedSignalData1,epochedSignalData2,epochedSignalData3), axis=0) )
#  else:
    #epochedSignalDataCore = (epochedSignalData1,epochedSignalData2,epochedSignalData3)
  #for i,epochedSignalData in enumerate(epochedSignalDataCore):
  
    #print tokenProps.shape,epochedSignalData.shape
    sys.stderr.write(str(epochedSignalData.shape)+' '+str(parsedTrialsBool.shape) +' ' + str(inDataset.shape)+'\n')#& parsedTrialsBool & inDataset]
    wordEpochs = epochedSignalData[wordTrialsBool & parsedTrialsBool & inDataset]
    wordFeatures = tokenProps[wordTrialsBool & parsedTrialsBool & inDataset]
  # The FFT script collapses across channels
  # index0: epoch
  # index1: channels x fft_feature_types x frequencies
  # solution: reshape the output as (epochs,channels,-1)
  
  
  # Spectrally decompose the epochs
    #freqres = 32 (narrow window) freqres = 0 (wide, epoch-length window) freqres=220 (medium window?)
    (mappedTrialFeatures, spectralFrequencies) = specfft.mapFeatures(wordEpochs,samplingRate,windowShape='hann',featureType='amp',freqRes=0)
  # Reshape output to get epochs x channels x frequency
    mappedTrialFeatures = mappedTrialFeatures.reshape((wordEpochs.shape[0],wordEpochs.shape[1],-1))
  #  print 'FFT output: ', mappedTrialFeatures.shape, spectralFrequencies.shape
  #  raise
  
  
  #print(spectralFrequencies)
    freqbands = {}
    if freqsource == 'weiss':
        #Weiss et al. 05
        freqbands['theta'] = numpy.nonzero( (spectralFrequencies >= 4) & (spectralFrequencies <= 7) )
        freqbands['beta1'] = numpy.nonzero( (spectralFrequencies >= 13) & (spectralFrequencies <= 18) )
        freqbands['beta2'] = numpy.nonzero( (spectralFrequencies >= 20) & (spectralFrequencies <= 28) )
        freqbands['gamma'] = numpy.nonzero( (spectralFrequencies >= 30) & (spectralFrequencies <= 34) )
    elif freqsource == 'wiki':
        # end of http://en.wikipedia.org/wiki/Theta_rhythm
        freqbands['delta'] = numpy.nonzero( (spectralFrequencies >= 0.1) & (spectralFrequencies <= 3) )
        freqbands['theta'] = numpy.nonzero( (spectralFrequencies >= 4) & (spectralFrequencies <= 7) )
        freqbands['alpha'] = numpy.nonzero( (spectralFrequencies >= 8) & (spectralFrequencies <= 15) )
        freqbands['beta'] = numpy.nonzero( (spectralFrequencies >= 16) & (spectralFrequencies <= 31) )
        freqbands['gamma'] = numpy.nonzero( (spectralFrequencies >= 32) & (spectralFrequencies <= 100) )
    else:
        #Interpolate between weiss and wiki
        #print(numpy.nonzero((spectralFrequencies >= 4) & (spectralFrequencies <= 7)))
        freqbands['theta'] = numpy.nonzero( (spectralFrequencies >= 4) & (spectralFrequencies <= 7) )
        freqbands['alpha'] = numpy.nonzero( (spectralFrequencies >= 8) & (spectralFrequencies < 13) )
        freqbands['beta1'] = numpy.nonzero( (spectralFrequencies >= 13) & (spectralFrequencies <= 18) )
        freqbands['beta2'] = numpy.nonzero( (spectralFrequencies >= 20) & (spectralFrequencies <= 28) )
        freqbands['gamma'] = numpy.nonzero( (spectralFrequencies >= 30) & (spectralFrequencies <= 34) )

    # REGULARISATION VALUES TO TRY (e.g. in Ridge GCV)
    regParam = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 1e+2, 2e+2, 5e+2, 1e+3, 2e+3, 5e+3]
  
  # SELECT AND DESCRIBE THE REGRESSORS WE'RE CHOOSING TO USE
  # this strings should match the names of the fields in tokenProps
  # here you should list the features you're choosing from your .tab file (just one?)#
    features = [
     'logFreq_ANC',
     #'surprisal2back_COCA',
     'bigramLogProbBack_COCA',
     #'bigramEntropy_COCA_here',
     'sentenceSerial',
     'runSerial',
     'storySerial',
     'syndepth'
    ]

    # SLOT REGRESSORS IN ONE BY ONE
    explanatoryFeatures = numpy.zeros((wordFeatures.shape)) # dummy
    #explanatoryFeatures = numpy.array([])
    for feature in features:
      explanatoryFeatures = numpy.vstack((explanatoryFeatures, wordFeatures[feature]))
    explanatoryFeatures = explanatoryFeatures[1:].T # strip zeros out again
  
  # STEP THROUGH EACH TIME POINT IN THE EPOCH, RUNNING REGRESSION FOR EACH ONE
    avefitsig = {}
    avefit = {}
    #fitlm = {}
    #df_banddict = {}
    #bandmaxfits = {}

    if CLUSTER:
      #accumulate the cluster data before running the analysis
      # This is impractical because we can't hold all the data in memory at once...
#      for freq in mappedTrialFeatures[0,0,:]:
#        if not freqsY:
#          freqsY = mappedTrialFeatures[:,0,freq]
#        else:
#          freqsY = concatenate(freqsY,mappedTrialFeatures[:,0,freq],axis=1)
      clusterid = sensorCluster[channelix]
      if clusterid not in clusterpower:
        clusterpower[clusterid] = {}
        clustersize[clusterid] = 0
      clustersize[clusterid] += 1
      for band in freqbands:
        for freq in freqbands[band]:
          if not freqsY:                                                     
            freqsY = mappedTrialFeatures[:,0,freq]                           
          else:                                                              
            freqsY = concatenate(freqsY,mappedTrialFeatures[:,0,freq],axis=1)
        if band not in clusterpower[clusterid]:
          clusterpower[clusterid][band] = numpy.mean(mynormalise(freqsY),axis=1).reshape(-1,1)
        else:
          clusterpower[clusterid][band] = concatenate(clusterpower[clusterid][band],numpy.mean(mynormalise(freqsY),axis=1).reshape(-1,1))

        if not trainX:
          trainX = pd.DataFrame(data = mynormalise(explanatoryFeatures), columns = features)
      continue #don't run the analysis yet because we haven't accumulated over the whole cluster
      
    for band in freqbands:
        modelTrainingFit = []
        modelTestCorrelation = []
        modelParameters = []
        legendLabels = features

        trainX = pd.DataFrame(data = mynormalise(explanatoryFeatures), columns = features)
        
        freqsY = None
        for freq in freqbands[band]:
          # WHICH VARIETY OF REGRESSION TO USE?
          # I get pretty similar results with all three of those below. The most generic (ie fewest extra assumptions) is normal LinearRegression. I guess RidgeCV should do best in terms of R^2, but has discontinuities in betas, as different regularisation parameters are optimal at each time step. LassoLars is something of a compromise. #
          #lm = sklearn.linear_model.LinearRegression(fit_intercept=True, normalize=True)
          #lm = sklearn.linear_model.RidgeCV(fit_intercept=True, normalize=True, alphas=regParam) #, 10000, 100000])
          #lm = sklearn.linear_model.LassoLars(alpha=0.0001) #(alpha=1.0, fit_intercept=True, verbose=False, normalize=True, precompute='auto', max_iter=500, eps=2.2204460492503131e-16, copy_X=True)
  
          # NORMALISE THE EXPLANATORY VARIABLES (for comparable beta magnitude interpretation)
          if not freqsY:
            freqsY = mappedTrialFeatures[:,0,freq]
          else:
            freqsY = concatenate(freqsY,mappedTrialFeatures[:,0,freq],axis=1)

        trainX['Y'] = numpy.mean(mynormalise(freqsY),axis=1)

#        df = pd.DataFrame({'Y':trainY,'X':trainX})
#        fitlm = pd.ols(y = df['Y'], x = df['X']) # intercept=True)

        #lm = sm.OLS(trainY,trainX)
        myform = 'Y ~ '+'+'.join(features)
#        print myform, freqsY.shape, trainX.shape
        lm = smf.ols(formula=myform,data=trainX,missing='drop')
        bandlm = lm.fit_regularized(alpha=0.0001) #does a Lasso fit with same regularizer as above
        signif_of_fit = bandlm.pvalues #stat['p-value']
        goodness_of_fit0 = bandlm.rsquared
        goodness_of_fit1 = bandlm.rsquared_adj
        #trainedLM = lm.fit(trainX,trainY)
        #modelParameters.append(lm)
        #print(lm.score(trainX,trainY),trainX.shape[1], trainX.shape[0])
        #modelTrainingFit.append(adjustR2(lm.score(trainX,trainY), trainX.shape[1], trainX.shape[0]))
        #modelTrainingFit.append(lm.score(trainX,trainY)) #for a single feature, no punishment is necessary
        avefitsig[band] = signif_of_fit
        avefit[band] = (goodness_of_fit0,goodness_of_fit1)
        #fitlm[band] = bandlm
        #df_banddict[band] = trainX

#        print bandlm.summary()
#        print 'R2:',goodness_of_fit
#        raise
        
        #bandmaxfits[band] = numpy.max(modelTrainingFit)
  
    avefitsig_dict[ metaData1.chanlocs[channelix].labels ] = avefitsig
    avefit_dict[ metaData1.chanlocs[channelix].labels ] = avefit
#    lm_dict[ metaData1.chanlocs[channelix].labels ] = fitlm
#    df_dict[ metaData1.chanlocs[channelix].labels ] = df_banddict
    
    #maxfitresults[ metaData1.chanlocs[channelix].labels ] = bandmaxfits
  
  
  #print(modelTrainingFit)
  #print(numpy.sort(modelTrainingFit)[::-1])
  #print 'ave fit: ', numpy.mean(modelTrainingFit)
  #print 'max fit: ', numpy.max(modelTrainingFit)
if CLUSTER:
  #now we've got all the clusters, so analyze them
  myform = 'Y ~ '+'+'.join(list(trainX.columns))
  for clusterid in clusterpower:
    avefit = {}
    avefitsig = {}
    for band in clusterpower[clusterid]:
      trainX['Y'] = numpy.mean(mynormalise(clusterpower[clusterid][band]),axis=1)
      lm = smf.ols(formula=myform,data=trainX,missing='drop')
      bandlm = lm.fit_regularized(alpha=0.0001) #does a Lasso fit with same regularizer as above
      signif_of_fit = bandlm.pvalues #stat['p-value']
      goodness_of_fit0 = bandlm.rsquared
      goodness_of_fit1 = bandlm.rsquared_adj
      avefitsig[band] = signif_of_fit
      avefit[band] = (goodness_of_fit0,goodness_of_fit1)
    avefitsig_dict[ clusterid ] = avefitsig
    avefit_dict[ clusterid ] = avefit
    
  
fitresults = {'r2':avefit_dict,'p':avefitsig_dict} #,'lm':lm_dict,'df':df_dict}
fname = 'signifresults.multifactor'
if DEV:
  fname = fname + '.dev'
else:
  fname = fname + '.test'
if CLUSTER:
  fname = fname + '.cluster'

  
cPickle.dump(fitresults, open(fname+'.cpk', 'wb'))

