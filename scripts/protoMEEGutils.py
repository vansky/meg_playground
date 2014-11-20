import protoReadExtractEEGlab 
import protoReadExtractEEGlabICAcomps
import protoEpochifyContinuousEEG
#import protoSplineMapper
#import protoWaveletMapper
import cPickle
import time
import numpy
import mvpa2.suite
import mvpa2
import scipy
import scipy.linalg
#import pylab


# # # # # # #
# 
# TO DO
# 
# ESSENTIAL:
# !!!! # FIX !!! fact that all trackTrial times are 4ms late (=1 sample)


# LOAD AUDIOBOOK MEG SET (INCLUDING AUDIO SIGNAL), AND TOKEN-TRIAL PROPERTIES
def loadBookMEGWithAudio(eegFile, wordPropsFile, triggersOfInterest, epochEnd=1, epochStart=0, icaComps=False):
	print 'loading word properties and timing from pickled file',wordPropsFile
	wordProps=cPickle.load(open(wordPropsFile, 'rb')) # created with autoHofDWordPropsV2.py

	if not icaComps:
		print '... loading signal data from eeglab file',eegFile
		(rawSignalData, metaData, trackTrials, epochedSignalData, epochSliceTimepoints) = protoReadExtractEEGlab.load(eegFile, triggersOfInterest, epochEnd, epochStart)
	else:
		print '... loading ICAed data from eeglab file',eegFile
		(rawSignalData, metaData, trackTrials, epochedSignalData, epochSliceTimepoints) = protoReadExtractEEGlabICAcomps.load(eegFile, triggersOfInterest, epochEnd, epochStart)

	del epochedSignalData # will have to recreate anywaynfoldCvedSelectedClassifier.ca.stats.sets[0][0]
	audioSignal = rawSignalData[-1,:]
	rawSignalData = rawSignalData[:-1,:] # chuck last channel, which is audio

	samplingRate = metaData.srate
	numChannels = rawSignalData.shape[0] # original dimensionality in metaData.nbchan

	print time.ctime()
	return (rawSignalData, metaData, trackTrials, wordProps, audioSignal, samplingRate, numChannels)


# BREAK MEG DATASET INTO EPOCHS, BASED ON TOKEN TIMING AND EPOCH SPECS
def wordTrialEpochify(rawSignalData, samplingRate, wordProps, trackTrials, refEvent='start', epochEnd=1, epochStart=0):
	# derive absolute timing of each token trial (using timing of the start of each track from eeg triggers + the latency of each word onset relative to
	print 'determining experiment-specific word timing, of reference point:',refEvent,time.ctime()
	wordTimesAbsolute = []
	for tokenEntry in wordProps:
		refTime = tokenEntry[refEvent]
		wordTimesAbsolute.append(refTime+(trackTrials['timepoint'][tokenEntry['track']-1]/samplingRate))
	print time.ctime()

	# re-epoch the raw data, relative to each word onset
	print '... epoching data on the basis of these',len(wordTimesAbsolute),refEvent,'events, with epoch (s):',epochStart,epochEnd,time.ctime()
	(wordTrials, epochedSignalData, epochSliceTimepoints) = protoEpochifyContinuousEEG.epochify(rawSignalData, samplingRate, wordTimesAbsolute, epochEnd, epochStart)
	numTrials = epochedSignalData.shape[0] # or len(trials)
	epochNumTimepoints = epochedSignalData.shape[2] # length of each epoch in timepoints (aka samples)

	print time.ctime()
	return (wordTrials, epochedSignalData, epochSliceTimepoints, wordTimesAbsolute, numTrials, epochNumTimepoints)


# MAKE INTO pyMVPA DATASET, Z-SCORING
def pymvpaDatasetZscored(features, targets, numFolds, foldMode='inter', trialwiseFolds=None):
	print 'making folded, z-scored, pyMVPA dataset',time.ctime()
	if trialwiseFolds:
		print '... predetermined folds'
		chunks=trialwiseFolds
	elif foldMode.lower().startswith('inter'): # interleaved
		print '...',numFolds,'wise interleaved folds'
		chunks=numpy.array(range(len(targets)))%numFolds+1 
	elif foldMode.lower().startswith('seq'): # sequential
		print '...',numFolds,'wise sequential folds'
		chunks=numpy.array(range(len(targets)))%numFolds+1 
		chunks.sort()
	dataset = mvpa2.suite.dataset_wizard(samples=features, targets=targets, chunks=chunks)

	print '... zscoring',time.ctime()
	mvpa2.suite.zscore(dataset)

	#print dataset.summary()

	print time.ctime()
	return dataset


# FACTORISE/DIMENSIONALITY-REDUCE M/EEG SIGNALS
def factoriseSignals(signalData, method='svd'):
	if method.lower()=='svd':
		print 'get SVD of signals',time.ctime()
		U,sigma,Vh = scipy.linalg.svd(rawSignalData.transpose(), full_matrices=False)
		rawSignalData = U.transpose() #[:pcaCompsN]
		## MORE EFFICIENT to use PCA.fit on a subsample (e.g. every 10th sample) and then apply to rest

	numChannels = rawSignalData.shape[1] # original dimensionality in metaData.nbchan

	print time.ctime()
	return (rawSignalData, numChannels)


# FACTORISE/DIMENSIONALITY-REDUCE PYMVPA DATASET FEATURES
def factoriseFeatures(features, method='svd'):
	if method.lower()=='svd':
		print 'get SVD of features',time.ctime()
		U,sigma,Vh = scipy.linalg.svd(features, full_matrices=False)
		features = U
		## MORE EFFICIENT to use PCA.fit on a subsample (e.g. every 10th sample) and then apply to rest

	print time.ctime()
	return (features)


# VISUALISE ERPF GLOBAL AND CONDITIONS
def bunchERPs2conditions(epochedSignalData, numChannels, metaData, wordProps, propOfInterest, catsOfInterest, epochStart, epochEnd, samplingRate, ERPchannels):
	print 'visualise epochs by sensor type, epoch type',time.ctime()
	# make list of channels of each of the three types
	mags1 = []
	grads2 = []
	grads3  = []
	for c in range(numChannels):
		if(('%s'% metaData.chanlocs[c].labels).endswith('1')):
			mags1.append(c)
		elif(('%s'% metaData.chanlocs[c].labels).endswith('2')):
			grads2.append(c)
		else:
			grads3.append(c)

	# display details
	epochSamples = epochedSignalData.shape[2]
	epochLength = epochEnd-epochStart
	ticks = numpy.around(numpy.linspace(0,epochSamples,(epochLength/.1)+1))
	tickLabels = epochStart+(numpy.linspace(0,epochSamples,(epochLength/.1)+1)/samplingRate)
	channelLabels = [str(metaData.chanlocs[c].labels) for c in ERPchannels]
	magMultFactor = 2 # compared numpy.std(allWordERPs[mags1].flatten()) to numpy.std(allWordERPs[grads2].flatten()), which gave ~3, but was more peaky so reduced to 2

	# ERPs (aka ERFs) for all sensors
	allERPs = numpy.mean(epochedSignalData,axis=0)
	allWordERPs = numpy.mean(epochedSignalData[wordProps[propOfInterest] != '',:,:],axis=0)
	allRelevantERPs = numpy.mean(epochedSignalData[numpy.logical_or(wordProps[propOfInterest] == catsOfInterest[0],wordProps[propOfInterest] == catsOfInterest[1]),:,:],axis=0)
	mags1WordERPs = allWordERPs[mags1,:]
	grads2WordERPs = allWordERPs[grads2,:]
	grads3WordERPs = allWordERPs[grads3,:]

	# compare sensor types over all word epochs
	pylab.figure()
	pylab.subplot(2,2,1)
	pylab.plot(allWordERPs.T)
	pylab.xticks(ticks,tickLabels)
	pylab.xlim(ticks[0],ticks[-1])
	pylab.title('all words, all channels')
	pylab.subplot(2,2,2)
	pylab.plot(mags1WordERPs.T)
	pylab.xticks(ticks,tickLabels)
	pylab.xlim(ticks[0],ticks[-1])
	pylab.title('magnetometers')
	pylab.subplot(2,2,3)
	pylab.plot(grads2WordERPs.T)
	pylab.xticks(ticks,tickLabels)
	pylab.xlim(ticks[0],ticks[-1])
	pylab.title('gradiometers (2)')
	pylab.subplot(2,2,4)
	pylab.plot(grads3WordERPs.T)
	pylab.xticks(ticks,tickLabels)
	pylab.xlim(ticks[0],ticks[-1])
	pylab.title('gradiometers (3)')

	pylab.figure()
	pylab.plot(allERPs.T)
	pylab.xticks(ticks,tickLabels)
	pylab.xlim(ticks[0],ticks[-1])
	pylab.title('all epochs')

	pylab.figure()
	pylab.plot(allWordERPs.T)
	pylab.xticks(ticks,tickLabels)
	pylab.xlim(ticks[0],ticks[-1])
	pylab.title('word epochs')

	pylab.figure()
	pylab.plot(allRelevantERPs.T)
	pylab.xticks(ticks,tickLabels)
	pylab.xlim(ticks[0],ticks[-1])
	pylab.title((propOfInterest,catsOfInterest))

	# show each sensor individually
	pylab.figure()
	for s in range(numChannels/3):
		sensorTriplet = [mags1[s], grads2[s], grads3[s]]
		sensorLabel = str(metaData.chanlocs[mags1[s]].labels)[:-1]
		print(s,sensorLabel, sensorTriplet)
		pylab.subplot(11,10,s+1)
		pylab.plot(magMultFactor*allWordERPs[sensorTriplet[0],:].T)
		pylab.plot(allWordERPs[sensorTriplet[1:],:].T)
		pylab.title(sensorLabel)
		pylab.xlim(ticks[0],ticks[-1])
		pylab.xticks([])
		pylab.yticks([])
		pylab.axvline(x=25, color='grey')
		pylab.axvline(x=37, color='grey')
		pylab.axvline(x=75, color='grey')
	
	pylab.xticks([0,25,50,75,100,125],['','0','0.2','0.4','0.6','0.8'])
	pylab.subplot(11,10,s+2)
	pylab.plot(magMultFactor*allWordERPs[sensorTriplet[0],:].T)
	pylab.plot(allWordERPs[sensorTriplet[1:],:].T)
	pylab.xlim(ticks[0],ticks[-1])
	pylab.xticks([])
	pylab.yticks([])
	pylab.legend(['mag','grad2','grad3'])
	pylab.axvline(x=25, color='grey')
	pylab.axvline(x=37, color='grey')
	pylab.axvline(x=75, color='grey')

	# show ERPs for conditions, and differences
	selERPs = allRelevantERPs[ERPchannels,:]
	tmp = epochedSignalData[wordProps[propOfInterest] == catsOfInterest[0],:,:];
	selERPsA = numpy.mean(tmp[:,ERPchannels,:],axis=0) # get error message if do these two slice operations in one go
	tmp = epochedSignalData[wordProps[propOfInterest] == catsOfInterest[1],:,:];
	selERPsB = numpy.mean(tmp[:,ERPchannels,:],axis=0) # get error message if do these two slice operations in one go
	selERPsDiff = selERPsA-selERPsB

	pylab.figure()
	pylab.subplot(2,2,1)
	pylab.plot(selERPs.T)
	pylab.xticks(ticks,tickLabels)
	pylab.xlim(ticks[0],ticks[-1])
	pylab.title('all')
	pylab.subplot(2,2,2)
	pylab.plot(selERPsDiff.T)
	pylab.xticks(ticks,tickLabels)
	pylab.xlim(ticks[0],ticks[-1])
	pylab.title('diff')
	pylab.subplot(2,2,3)
	pylab.plot(selERPsA.T)
	pylab.xticks(ticks,tickLabels)
	pylab.xlim(ticks[0],ticks[-1])
	pylab.title(catsOfInterest[0])
	pylab.subplot(2,2,4)
	pylab.plot(selERPsB.T)
	pylab.xticks(ticks,tickLabels)
	pylab.xlim(ticks[0],ticks[-1])
	pylab.title(catsOfInterest[1])
	pylab.legend(channelLabels)

	pylab.show()


# TIC TOC functionality
tictoctimer = 0
def tic():
	global tictoctimer
	tictoctimer = time.time()
def toc(tag = ''):
	d = time.time()-tictoctimer
	print d
	if d >= 24*60*60:
		print '%s elapsed %dd %dh %dm %.3fs' % (tag, int(d/60.0/60/24), int(d/60.0/60), int(d/60), d%60)
	elif d >= 60*60:
		print '%s elapsed %dh %dm %.3fs' % (tag, int(d/60.0/60), int(d/60), d%60)
	elif d >=60:
		print '%s elapsed %dm %.3fs' % (tag, int(d/60.0), d%60)
	else:
		print '%s elapsed %.3fs' % (tag, d)
	return d

