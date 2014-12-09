# -*- coding: utf-8; python-indent: 2; -*-
# This script extracts P-values and R^2 values over each frequency band and determines model fits

# Global Vars
# =======

DEV = True #if True: analyze the dev set; if False: analyze the test set ;; DEV is defined on a sentence level using a stepsize of N ;; TEST is the complement of DEV
devsizerecip = 3 # the reciprocal of the dev size, so devsizerecip = 3 means the dev set is 1/3 and the test set is 2/3

freqsource = None # determines how the frequency bands are defined #Can be 'weiss', 'wiki', or (any other value) an interpolation of the two

CLUSTER = False #if True: analyses are done over each channel; if False: channels are clustered prior to the analysis
clustershape = (3,3) #Clustering is done based on sensor position; given (x,y), imagine a grid with x rows and y cols 

SHORTTEST = True #Tests only the MEG analogues to EEG's Pz; Requires CLUSTER = False

LASSO = False #False #if True: use Lasso regression; if False: use ridge regression
FINDVALS = False #If True, searches for the optimal regression alpha value; Requires LASSO = False
regParam = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 1e+2, 2e+2, 5e+2, 1e+3, 2e+3, 5e+3] #the alpha values to try for FINDVALS

RMOUTLIERS = True #Should outliers be removed? Removes outliers based on each (channel, frequency band)
OUTDEF = 3 #outliers are defined as those stimuli with dep vars whose value is <= this many std deviations of the mean

TESTFACTORS = False #Get the correlation matrix of the predictors; Get VIF of each predictor to test for multicollinearity
VERBOSE = False #Provide some extra output, mainly for development purposes
FUDGE = False # this factor is used to enable and mark dubious code, which can be cleaned up later; purely for development

if CLUSTER and FINDVALS:
  raise #we don't have clustering able to optimize alpha in ridge regression yet.
if CLUSTER and SHORTTEST:
  raise #can't combine cluster analysis and Pz analysis.
if CLUSTER and TESTFACTORS:
  #haven't implemented collinearity checks in cluster analysis
  # (and the checks are independent of the analysis),
  # so either do TESTFACTORS with SHORTTEST or don't do them
  print 'TESTFACTORS=True, but CLUSTER analysis only outputs a correlation matrix. It doesn\'t do VIF testing.'
if LASSO and FINDVALS:
  raise #only ridge permits exploring alpha vals

# Imports
# =======

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
from sklearn.feature_selection import chi2
import sys
import re
#import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.outliers_influence

#pylab.rcParams['figure.figsize'] = 10,10 #change the default image size for this session
#pylab.ion()

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
cPickle.dump(tokenProps, open(tokenPropsPickle, 'wb'))

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
#print(parsedTrialsBool[:10])

sentidlist = numpy.bincount(tokenProps['sentid'][tokenProps['sentid'] != -1]) #gives the sentence length indexed by sentence ID
sentidlist = sentidlist / float(NUMSUBJS) #account for the fact that we've seen each sentence 3 times (once per subject)
sentidlist = sentidlist.astype(int)
validsents = numpy.nonzero((sentidlist > 4) & (sentidlist <= 50))[0]# & (sentidlist >= 5))[0] #only permit sentences with lengths less than 51
#validsents = numpy.nonzero((sentidlist <= 50))[0]# & (sentidlist >= 5))[0] #only permit sentences with lengths less than 51

#validsents = numpy.nonzero(sentidlist)[0]
#for i in range(max(sentlens.keys())):
#  sys.stderr.write(str(i)+':'+str(sentlens.get(i,0))+'\n')
#tmp = sentidlist > 200 #0 and s < 100]
#print numpy.nonzero(tmp)
#for i,t in enumerate(tmp):
#  if t:
#  sys.stderr.write(str(s)+'\n')
#sentids = numpy.nonzero(sentidlist)[0]
#print 'Sentence Lengths:', sentidlist[sentids]
#print 'Sentence Lengths:', sentidlist[]
#tmp = sentidlist[sentids] < 100
#sentlenhist = numpy.bincount(sentidlist[numpy.nonzero(sentidlist <= 50)])#[sentids][tmp])
#sentlenhist_readable = numpy.nonzero(sentlenhist)[0]
#print 'Numsents:', sum(sentlenhist[sentlenhist_readable]), ' 95%:', sum(sentlenhist[sentlenhist_readable])*.95
#prevs = 0
#numsents = 0
#for s in sentlenhist_readable:
#  numsents += sentlenhist[sentlenhist_readable]
#  if numsents > 590:
#    print 'Cutoff:',prevs
#  prevs = s
#print 'Sentence Length Counts:', zip(sentlenhist_readable,sentlenhist[sentlenhist_readable])
#raise

#sys.stderr.write(str(validsents)+'\n')

# Set up the dev and test sets
devitems = numpy.arange(1,max(tokenProps['sentid']),devsizerecip)
if FUDGE:
  devitems = numpy.arange(1,max(tokenProps['sentid']),devsizerecip*2) #split dev set in half
devTrialsBool = numpy.array([s in devitems and s in validsents for s in tokenProps['sentid']])
testTrialsBool = numpy.array([s not in devitems and s in validsents for s in tokenProps['sentid']])


#badsents = numpy.nonzero((sentidlist > 50))[0]
#baddev = []
#badtest = []
#for s in badsents:
#  if s in devitems:
#    baddev.append((s,sentidlist[s]))
#  else:
#    badtest.append((s,sentidlist[s]))
#sys.stderr.write('Bad Dev'+'\n'+str(baddev)+'\n'+'Bad Test'+'\n'+str(badtest)+'\n')
#raise

if DEV:
  inDataset = devTrialsBool
else:
  inDataset = testTrialsBool

if CLUSTER:
  # given an (x,y) specification, define x rows and y cols for clustering
  xdivs = numpy.arange(float(clustershape[1])+1) # columns are defined by slicing at x values
  ydivs = numpy.arange(float(clustershape[0])+1) # rows are defined by slicing at y values

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

  if VERBOSE:
    print minx, maxx, miny, maxy
    print 'x: '+str(xdivs)+' y: '+str(ydivs)
  xdivs *= float(maxx-minx)/float(clustershape[0])
  xdivs += minx
  ydivs *= float(maxy-miny)/float(clustershape[1])
  ydivs += miny
  if VERBOSE:
    print 'x: '+str(xdivs)+' y: '+str(ydivs)

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

if CLUSTER:
  print 'Cluster IDs: ', str(set(sensorCluster.values()))
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
FIRSTCLUSTER = True
if SHORTTEST:
  chanixes = [i for i,c in enumerate(metaData1.chanlocs) if c.labels[:-1] == 'MEG224']
else:
  chanixes = range(metaData1.chanlocs.shape[0]-1) #minus 1 because the last 'channel' is MISC

for channelix in chanixes: 
  print 'Compiling data from channel:',channelix
  #need to reshape because severalMagChannels needs to be channel x samples, and 1-D shapes are flattened by numpy
  severalMagChannels1 = contSignalData1[channelix,:].reshape((1,-1))

#  channelLabels = ['MEG0111', 'MEG0121', 'MEG0131', 'MEG0211', 'MEG0212', 'MEG0213', 'MEG0341']

  (wordTrials1, epochedSignalData1, epochSliceTimepoints, wordTimesAbsolute, numTrials, epochNumTimepoints) = wordTrialEpochify(severalMagChannels1, samplingRate, tokenPropsOrig, trackTrials, refEvent, epochEnd, epochStart)

  severalMagChannels2 = contSignalData2[channelix,:].reshape((1,-1))
  (wordTrials2, epochedSignalData2, epochSliceTimepoints, wordTimesAbsolute, numTrials, epochNumTimepoints) = wordTrialEpochify(severalMagChannels2, samplingRate, tokenPropsOrig, trackTrials, refEvent, epochEnd, epochStart)

  severalMagChannels3 = contSignalData3[channelix,:].reshape((1,-1))
  (wordTrials3, epochedSignalData3, epochSliceTimepoints, wordTimesAbsolute, numTrials, epochNumTimepoints) = wordTrialEpochify(severalMagChannels3, samplingRate, tokenPropsOrig, trackTrials, refEvent, epochEnd, epochStart)

  if True:
    epochedSignalData = ( numpy.concatenate((epochedSignalData1,epochedSignalData2,epochedSignalData3), axis=0) )

    wordEpochs = epochedSignalData[wordTrialsBool & parsedTrialsBool & inDataset]
    wordFeatures = tokenProps[wordTrialsBool & parsedTrialsBool & inDataset]

  # Spectrally decompose the epochs
  #  freqres = 32 (narrow window) freqres = 0 (wide, epoch-length window) freqres=220 (medium window?)
    (mappedTrialFeatures, spectralFrequencies) = specfft.mapFeatures(wordEpochs,samplingRate,windowShape='hann',featureType='amp',freqRes=0)

  # The FFT script collapses across channels
  # index0: epoch
  # index1: channels x fft_feature_types x frequencies
  # solution: reshape the output as (epochs,channels,-1)
  
  # Reshape output to get epochs x channels x frequency
    mappedTrialFeatures = mappedTrialFeatures.reshape((wordEpochs.shape[0],wordEpochs.shape[1],-1))
  #  print 'FFT output: ', mappedTrialFeatures.shape, spectralFrequencies.shape
  #  raise
  
  # define spectral frequency bands
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
  
  # SELECT AND DESCRIBE THE REGRESSORS WE'RE CHOOSING TO USE
  # these strings should match the names of the fields in tokenProps
  # here you should list the features you're choosing from your .tab file #
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

    for feature in features:
      explanatoryFeatures = numpy.vstack((explanatoryFeatures, wordFeatures[feature]))
    explanatoryFeatures = explanatoryFeatures[1:].T # strip zero dummy out again
  
  # STEP THROUGH EACH TIME POINT IN THE EPOCH, RUNNING REGRESSION FOR EACH ONE
    avefitsig = {}
    avefit = {}
    #fitlm = {}
    #df_banddict = {}
    #bandmaxfits = {}

    if CLUSTER:
      #accumulate the cluster data before running the analysis
      
#      This first commented approach is impractical because we can't hold all the data in memory at once...
#      for freq in mappedTrialFeatures[0,0,:]:
#        if not freqsY:
#          freqsY = mappedTrialFeatures[:,0,freq]
#        else:
#          freqsY = concatenate((freqsY,mappedTrialFeatures[:,0,freq]),axis=1)
      clusterid = sensorCluster[channelix]
      if clusterid not in clusterpower:
        clusterpower[clusterid] = {}
        clustersize[clusterid] = 0
      clustersize[clusterid] += 1
      for band in freqbands:
        for ix,freq in enumerate(freqbands[band]):
          if ix == 0:
            freqsY = mappedTrialFeatures[:,0,freq]
          else:
            freqsY = numpy.concatenate((freqsY,mappedTrialFeatures[:,0,freq]),axis=1)
        if band not in clusterpower[clusterid]:
          clusterpower[clusterid][band] = numpy.mean(mynormalise(freqsY),axis=1).reshape(-1,1)
        else:
          clusterpower[clusterid][band] = numpy.concatenate((clusterpower[clusterid][band],numpy.mean(mynormalise(freqsY),axis=1).reshape(-1,1)),axis=1)

      if FIRSTCLUSTER:
        trainX = pd.DataFrame(data = mynormalise(explanatoryFeatures), columns = features)
        FIRSTCLUSTER = False
      continue #don't run the analysis yet because we haven't accumulated over the whole cluster      

    # If we've made it this far, we're not clustering...
    for band in freqbands:
        if VERBOSE or FINDVALS or TESTFACTORS: 
          sys.stderr.write('Fitting: '+band+'\n')
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
            freqsY = numpy.concatenate(freqsY,mappedTrialFeatures[:,0,freq],axis=1)

        Y = numpy.mean(mynormalise(freqsY),axis=1)
        if not FINDVALS:
          if RMOUTLIERS:
            trainX = trainX[numpy.abs(Y-Y.mean())<=(OUTDEF*Y.std())] #only retain non-outliers
            Y = mynormalise(Y[numpy.abs(Y-Y.mean())<=(OUTDEF*Y.std())])
            trainX = mynormalise(trainX) 
            if TESTFACTORS and band == 'alpha':
              #Get VIF to test for multicollinearity
              for col in range(len(features)):
                #test each factor for multicollinearity with all the others
                vif = statsmodels.stats.outliers_influence.variance_inflation_factor(trainX,col)
                print str(features[col])+': '+str(vif)+'\n'
          trainX = pd.DataFrame(data = mynormalise(trainX), columns = features)
          if TESTFACTORS and band == 'alpha':
            print trainX.corr()
          trainX['Y'] = Y
        else:
          Y = Y.reshape(-1,1)
        if VERBOSE:
          print trainX.shape

        myform = 'Y ~ '+'+'.join(features)

        if LASSO:
          lm = smf.ols(formula=myform,data=trainX,missing='drop')
          bandlm = lm.fit_regularized(alpha=0.0001,L1_wt=1.0) #does a Lasso fit with regularizer parameterized by alpha
          signif_of_fit = bandlm.pvalues #stat['p-value']
          goodness_of_fit0 = bandlm.rsquared
          goodness_of_fit1 = bandlm.rsquared_adj
        else:
          if not FINDVALS:
            lm = smf.ols(formula=myform,data=trainX,missing='drop')
            bandlm = lm.fit_regularized(alpha=0.005,L1_wt=0.0) #does a Ridge fit with regularizer parameterized by alpha
            #bandlm = lm.fit_regularized(alpha=0.0001,L1_wt=0.0) #does a Ridge fit with regularizer parameterized by alpha
            signif_of_fit = bandlm.pvalues
            goodness_of_fit0 = bandlm.rsquared
            goodness_of_fit1 = bandlm.rsquared_adj

          else:
            #we're just finding alpha via ridge; sklearn won't give us significance, so just fill output with None
            lm = sklearn.linear_model.RidgeCV(fit_intercept=True, normalize=True, alphas=regParam)
            lm.fit(trainX,Y)
            sys.stderr.write('alpha: '+str(lm.alpha_)+'\n')
            signif_of_fit = None
            goodness_of_fit0 = None
            goodness_of_fit1 = None
          

          
        #print(lm.score(trainX,trainY),trainX.shape[1], trainX.shape[0])
        #modelTrainingFit.append(adjustR2(lm.score(trainX,trainY), trainX.shape[1], trainX.shape[0]))
        #modelTrainingFit.append(lm.score(trainX,trainY)) #for a single feature, no punishment is necessary
        avefitsig[band] = signif_of_fit
        avefit[band] = (goodness_of_fit0,goodness_of_fit1)

        #makes output file HUGE
        #fitlm[band] = bandlm
        #df_banddict[band] = trainX

#        print bandlm.summary()
#        raise
        
        #bandmaxfits[band] = numpy.max(modelTrainingFit)
  
    avefitsig_dict[ metaData1.chanlocs[channelix].labels ] = avefitsig
    avefit_dict[ metaData1.chanlocs[channelix].labels ] = avefit
#    lm_dict[ metaData1.chanlocs[channelix].labels ] = fitlm
#    df_dict[ metaData1.chanlocs[channelix].labels ] = df_banddict
    
    #maxfitresults[ metaData1.chanlocs[channelix].labels ] = bandmaxfits
  
  
if CLUSTER:
  #now we've got all the clusters, so analyze them
  myform = 'Y ~ '+'+'.join(list(trainX.columns))
  for clusterid in clusterpower:
    avefit = {}
    avefitsig = {}
    for band in clusterpower[clusterid]:
      Y = numpy.mean(mynormalise(clusterpower[clusterid][band]),axis=1)
      if RMOUTLIERS:
        trainXnoout = trainX[numpy.abs(Y-Y.mean())<=(OUTDEF*Y.std())] #only retain non-outliers
        Y = mynormalise(Y[numpy.abs(Y-Y.mean())<=(OUTDEF*Y.std())])
        trainXnoout = pd.DataFrame(data = mynormalise(trainXnoout), columns = features)
        if band == 'alpha' and TESTFACTORS:
          print trainXnoout.corr()
        trainXnoout['Y'] = Y
        #trainXnoout = trainX[numpy.abs(trainX['Y']-trainX['Y'].mean())<=(OUTDEF*trainX['Y'].std())] #only retain non-outliers
        lm = smf.ols(formula=myform,data=trainXnoout,missing='drop')
      else:
        trainX['Y'] = Y
        lm = smf.ols(formula=myform,data=trainX,missing='drop')
      if LASSO:
        bandlm = lm.fit_regularized(alpha=0.005) #does a Lasso fit with regularizer parameterized by alpha
      else:
        bandlm = lm.fit_regularized(alpha=0.005, L1_wt=0.0) #does a Ridge fit with regularizer parameterized by alpha
      signif_of_fit = bandlm.pvalues #stat['p-value']
      goodness_of_fit0 = bandlm.rsquared
      goodness_of_fit1 = bandlm.rsquared_adj
      avefitsig[band] = signif_of_fit
      avefit[band] = (goodness_of_fit0,goodness_of_fit1)
      if VERBOSE:
        print clusterid
        print bandlm.summary()
    avefitsig_dict[ clusterid ] = avefitsig
    avefit_dict[ clusterid ] = avefit
    
  
fitresults = {'r2':avefit_dict,'p':avefitsig_dict} #,'lm':lm_dict,'df':df_dict}
if CLUSTER:
  fitresults.update({'clustersize':clustersize})
fname = 'sigresults'
if DEV:
  fname = fname + '.dev'
else:
  fname = fname + '.test'
if CLUSTER:
  fname = fname + '.cluster'
if SHORTTEST:
  fname = fname + '.short'
else:
  fname = fname + '.full'
  
cPickle.dump(fitresults, open(fname+'.cpk', 'wb'))

