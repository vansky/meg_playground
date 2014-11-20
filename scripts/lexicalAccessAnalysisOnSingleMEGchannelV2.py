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



#### SETTINGS ####

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

pylab.ion()

# brian's prototype routines
from protoMEEGutils import *

# INPUT PARAMS
#*# MARTY #*# choose a file - I found participant V to be pretty good, and 0.01 to 50Hz filter is pretty conservative #*#
(megFileTag, megFile) = ('V_TSSS_0.01-50Hz_@125', '/usr/data/meg/upmc/v_hod_allRuns_tsss_audiobookPrepro_stPad1_lp50_resamp125_frac10ICAed_hp0.010000.set')

#*# MARTY #*# put your on properties in here, as a .tab file of similar format (tab delimited, and field names in a first comment line - should be easy to do in excel...) #*#
tokenPropsFile = '/home/bmurphy/projects/heartOfDarkness/wordTimesAnnots/hod_JoeTimes_LoadsaFeaturesV3.tab' # best yet! have automatic Stanford tagging, several word length and freq measures, and also the 2 and 3 token back-grams

# WHICH CHANNELS TO LOOK AT AS ERFS
#*# MARTY #*# decide which channels to use - channels of interest are the first few you can look at in an ERF, and then from them you can choose one at a time with "channelToAnalyse" for the actual regression analysis #*#
channelsOfInterest = [2,5,8,14,35]
channelLabels = ['MEG0111', 'MEG0121', 'MEG0131', 'MEG0211', 'MEG0341'] # this could be read out of metadata of the .set file
channelToAnalyse = 4 # index of the channels above to actually run regression analysis on, so 4=MEG0341
#?# this way of doing things was a slightly clumsy work-around, cos I didn't have enough memory to epoch all 306 channels at one time

# LOAD WORD PROPS
tokenProps = scipy.genfromtxt
#*# MARTY #*# change dtype to suit the files in your .tab file #*#
(tokenPropsFile,delimiter='\t',names=True,dtype="i4,f4,f4,S50,S50,i2,i2,i2,S10,f4,f4,f4,f4,f4,f4,f4,f4,f4,f4,f4")
# ... and temporarily save as cpickle archive to satisfy the way I programmed the convenience function loadBookMEGWithAudio (it expects to find the same info in a C-pickle file, and so doesn't need to know about number and type of fields)
tokenPropsPickle = tokenPropsFile+'.cpk'
cPickle.dump(tokenProps, open(tokenPropsPickle, 'wb'))

# TRIAL PARAMS
triggersOfInterest=['s%d' % i for i in range(1,10)]
refEvent = 'onTime' #,'offTime']
#*# MARTY #*# guess an epoch of -0.5 to +1s should be enough #*#
epochStart = -1; # stimulus ref event
epochEnd = +2; # 
epochLength = epochEnd-epochStart;
baseline = False #[-1,0]



#### GET THE MEG DATA IN ####

print time.ctime()

# LOAD DATA
#setID = os.path.splitext(os.path.basename(megFile))[0]
#sessionID = setID[:setID.find('allRuns')-1]
(contSignalData, metaData, trackTrials, tokenProps, audioSignal, samplingRate, numChannels) = loadBookMEGWithAudio(megFile, tokenPropsPickle, triggersOfInterest, epochEnd, epochStart, icaComps=False)
# ... TRIM DOWN ...
# I had to do this, cos I ran out of memory, but could epochify all 306 channels otherwise
severalMagChannels = contSignalData[channelsOfInterest,:] # 011, 012, 013, 021, 034
# see at end of this script, and MEG presentation PDF, for more examples

# BREAK INTO TRIAL EPOCHS
(wordTrials, epochedSignalData, epochSliceTimepoints, wordTimesAbsolute, numTrials, epochNumTimepoints) = wordTrialEpochify(severalMagChannels, samplingRate, tokenProps, trackTrials, refEvent, epochEnd, epochStart)

# PLOTS FOR SANITY CHECK
pylab.figure()
pylab.subplot(2,1,1)
pylab.plot(audioSignal)
pylab.title('audio signal')

pylab.subplot(2,1,2)
pylab.plot(contSignalData[0,:])
pylab.title('first MEG signal')

pylab.figure()
pylab.title('ERF over all tokens, selected channels')
pylab.plot( numpy.mean(epochedSignalData,axis=0).T)
pylab.legend(channelLabels, loc=4)
commonPlotProps()



#### RUN REGRESSION ANALYSIS ####

# TODO: work out what this does????

# REDUCE TRIALS TO JUST THOSE THAT CONTAIN A REAL WORD (NOT PUNCTUATION, SPACES, ...)
wordTrialsBool = numpy.array([p != '' for p in tokenProps['stanfPOS']])
wordEpochs = epochedSignalData[wordTrialsBool]
wordFeatures = tokenProps[wordTrialsBool]

# REGULARISATION VALUES TO TRY (e.g. in Ridge GCV)
regParam = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 1e+2, 2e+2, 5e+2, 1e+3, 2e+3, 5e+3]

# SELECT AND DESCRIBE THE REGRESSORS WE'RE CHOOSING TO USE
# this strings should match the names of the fields in tokenProps
#*# MARTY #*# here you should list the features you're choosing from your .tab file (just one?) #*#
features = [
 'logFreq_ANC',
 'surprisal2back_COCA',
 'bigramEntropy_COCA_here',
]
#*# MARTY #*# ... this has shorthand versions of the variable names, for display, and also has to include the "position" one that this version of the script inserts by default #*#
labelMap = {
 'logFreq_ANC': 'freq',
 'surprisal2back_COCA': 'surprisal',
 'bigramEntropy_COCA_here': 'entropy',
 'sentenceSerial': 'position',
}
legendLabels = features

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

modelTrainingFit = []
modelTestCorrelation = []
modelParameters = []
legendLabels = features
tmpFeatures = explanatoryFeatures.copy()
tmpLegend = legendLabels[:]
for epochsBack in range(1,epochHistory+1):
	epochFeatures = numpy.zeros(tmpFeatures.shape)
	epochFeatures[epochsBack:,:] = tmpFeatures[:-epochsBack,:]
	explanatoryFeatures = numpy.hstack((explanatoryFeatures,epochFeatures))
	legendLabels = legendLabels + [l+'-'+str(epochsBack) for l in tmpLegend]

# put in sentence serial - can't leave in history, cos is too highly correlated across history...
explanatoryFeatures = numpy.vstack((explanatoryFeatures.T, wordFeatures['sentenceSerial'])).T
features.append('sentenceSerial')
legendLabels.append('sentenceSerial')

# STEP THROUGH EACH TIME POINT IN THE EPOCH, RUNNING REGRESSION FOR EACH ONE
for t in range(epochNumTimepoints):
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
	trainY = mynormalise(wordEpochs[:,channelToAnalyse,t])
	#trainX = explanatoryFeatures
	#trainY = wordEpochs[:,channelToAnalyse,t]

	trainedLM = lm.fit(trainX,trainY)
	modelParameters.append(lm)
	modelTrainingFit.append(adjustR2(lm.score(trainX,trainY), trainX.shape[1], trainX.shape[0]))

# DETERMINE IF THERE IS CORRELATION BETWEEN THE EXPLANATORY VARIABLES
betaMatrix = numpy.array([p.coef_ for p in modelParameters])
neatLabels = [l.replace(re.match(r'[^-]+',l).group(0), labelMap[re.match(r'[^-]+',l).group(0)]) for l in legendLabels if re.match(r'[^-]+',l).group(0) in labelMap]
legendLabels = numpy.array(legendLabels)
#numFeaturesDisplay = len(legendLabels)
neatLabels = numpy.array(neatLabels)

# DO BIG SUMMARY PLOT OF FEATURE CORRELATIONS, R^2 OVER TIMECOURSE, BETAS OVER TIME COURSE, AND ERF/ERP
f = pylab.figure()
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
s = pylab.subplot(2,2,3)
pylab.title('correlations between explanatory variables')
pylab.imshow(numpy.abs(numpy.corrcoef(explanatoryFeatures.T)),interpolation='nearest', origin='upper') # leave out the dummy one
pylab.clim(0,1)
pylab.yticks(range(len(neatLabels)),neatLabels)
pylab.ylim((-0.5,len(neatLabels)-0.5))
pylab.xticks(range(len(neatLabels)),neatLabels, rotation=90)
pylab.xlim((-0.5,len(neatLabels)-0.5))
pylab.colorbar()
s = pylab.subplot(2,2,4)
pylab.plot(numpy.mean(epochedSignalData[wordTrialsBool,channelToAnalyse],axis=0).T, linewidth=2)
pylab.title('ERF')
commonPlotProps()

pylab.savefig('meg_testfig.png')
print 'history %d, mean model fit over -0.5s to +1.0s: %.5f, max is %.5f' % (epochHistory, numpy.mean(modelTrainingFit[62:250]), numpy.max(modelTrainingFit[62:250]))





pylab.show()


#sensorGroups = (
#	('LeftAntTempMags', [2,5,8,14,35]), # mags @ locations 011, 012, 013, 021, 034
#	('LeftTempPNPgrads', [12, 15, 21, 165, 174]), #new improved PNP! ['MEG0213', 'MEG0222', 'MEG0243', 'MEG1522', 'MEG1613']
#	('LeftTempNPNgrads', [7, 13, 16, 22, 163, 166, 175]) #new improved NPN!
#)




