import scipy
import pickle
import numpy
import sklearn.linear_model
import matplotlib     #needed to avoid having to run an X-server just to output to a png
matplotlib.use('Agg') #needed to avoid having to run an X-server just to output to a png
import pylab
import re

def adjustR2(R2, numFeatures, numSamples):
	#1/0
	#return R2
	return R2-(1-R2)*(float(numFeatures)/(numSamples-numFeatures-1))


def mynormalise(A):
	A = scipy.stats.zscore(A)
	A[numpy.isnan(A)] = 0
	return A

wordPropsFile = '/home/corpora/original/english/meg_hod_corpus/projects/heartOfDarkness/wordTimesAnnots/hod_JoeTimes_LoadsaFeaturesV3.tab' # best yet! have automatic Stanford tagging, several word length and freq measures, and also the 2 and 3 token back-grams
wordProps = scipy.genfromtxt(wordPropsFile,delimiter='\t',names=True,dtype="i4,f4,f4,S50,S50,i2,i2,i2,S10,f4,f4,f4,f4,f4,f4,f4,f4,f4,f4,f4")

epochedGroupMeanAreasensorsMeanSensors = pickle.load(open('/home/corpora/original/english/meg_hod_corpus/usr/data/meg/heartOfDarkness/meanPartsMeanChannelsLeftTempPNPgradsUPMC4_TSSS_0.1-8Hz_@125.pkl'))
#dimensions of original input file: (14595 epochs, 375 channels)
#epochedGroupMeanAreasensorsMeanSensors = pickle.load(open('/usr1/meg/audioBookHofD/groupedDatasets/meanPartsMeanChannelsLeftTempPNPgradsUPMC4_TSSS_0.1-8Hz_@125.pkl'))
#epochedGroupMeanAreasensorsMeanSensors = pickle.load(open('/usr1/meg/audioBookHofD/groupedDatasets/meanPartsMeanChannelsLeftTempNPNgradsUPMC4_TSSS_0.1-8Hz_@125.pkl'))
#epochedGroupMeanAreasensorsMeanSensors = pickle.load(open('/usr1/meg/audioBookHofD/groupedDatasets/meanPartsMeanChannelsLeftAntTempMagsUPMC4_TSSS_0.1-8Hz_@125.pkl'))
#epochedGroupMeanAreasensorsMeanSensors = pickle.load(open('/usr1/meg/audioBookHofD/groupedDatasets/meanPartsMeanLeftTempMagsGradsUPMC4_TSSS_0.1-8Hz_@125.pkl'))

wordTrials = numpy.array([p != '' for p in wordProps['stanfPOS']])

#epochTimeSelection = range(750) #range(125,625) # -0.5 to +1.5s

epochStart = -1; # 1s before onset
epochEnd = 2; # 2s after onset
samplingRate = 125
epochSamples = epochedGroupMeanAreasensorsMeanSensors.shape[1]
epochLength = epochEnd-epochStart;
zeroSample = (abs(epochStart)/float(epochLength)*epochSamples)


wordEpochs = epochedGroupMeanAreasensorsMeanSensors[wordTrials]

wordFeatures = wordProps[wordTrials]

# TEMP, make task smaller
wordEpochs = wordEpochs[:]


#wordFeatures = 
#wordFeatures[:100]['surprisal2back_COCA']

#regParam = [0.001,0.01, 0.1, 1, 10, 1e+2, 1e+3]
regParam = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 1e+2, 2e+2, 5e+2, 1e+3, 2e+3, 5e+3]


#epochHistory=0

features = [
# 'random',

# 'track',
# 'sentenceSerial', # will be perfectly correlated with one from previous epoch; see below, first one gets re-introduced
# 'runSerial',
# 'storySerial', # is linear combination of track and runserial
# 'stanfPOS',
 
# 'duration',
# 'lenLett',
# 'lenPhonCMU',
# 'lenSyllCMU',
 'logFreq_ANC',
 
# 'tokenPrimed',
# 'tokenPosPrimed',

 'surprisal2back_COCA',
# 'bigramLogProbBack_COCA',
# 'trigramLogProbBack_COCA',
 
# 'bigramEntropy_COCA_previous', # will be perfectly correlated with one from previous epoch; 
 'bigramEntropy_COCA_here',
]

labelMap = {
 'logFreq_ANC': 'freq',
 'surprisal2back_COCA': 'surprisal',
 'bigramEntropy_COCA_here': 'entropy',
 'sentenceSerial': 'position',
}

legendLabels = features


explanatoryFeatures = numpy.zeros((wordFeatures.shape)) # dummy
#explanatoryFeatures = numpy.array([])
for feature in features:
	if feature == 'duration':
		explanatoryFeatures = numpy.vstack((explanatoryFeatures, wordFeatures['offTime']-wordFeatures['onTime']))
	elif feature == 'random':
		explanatoryFeatures = numpy.vstack((explanatoryFeatures, numpy.random.random((wordFeatures.shape))))
	else:
		explanatoryFeatures = numpy.vstack((explanatoryFeatures, wordFeatures[feature]))
#print '\tlearning semantic dimension',dim,'non-zeros, sum',len(trainY)-sum(trainY==0),sum(trainY), time.ctime()
	#explanatoryFeatures[-1,:] = (explanatoryFeatures[-1,:] - numpy.mean(explanatoryFeatures[-1,:]))/numpy.std(explanatoryFeatures[-1,:])
explanatoryFeatures = explanatoryFeatures[1:].T # strip zeros out again
#explanatoryFeatures = explanatoryFeatures.T # strip zeros out again
#features.insert(0,'dummyConstant')

def commonPlotProps():
	#pylab.plot((0,epochSamples),(0,0),'k--')
	#pylab.ylim((-2.5e-13,2.5e-13)) #((-5e-14,5e-14)) # better way just to get the ones it choose itself?
	#pylab.plot((zeroSample,zeroSample),(0,0.01),'k--')
	pylab.xticks(numpy.linspace(0,epochSamples,7),epochStart+(numpy.linspace(0,epochSamples,7)/samplingRate))
	pylab.xlabel('time (s) relative to auditory onset') #+refEvent)
	pylab.xlim((62,313))
	pylab.show()
	pylab.axhline(0, color='k', linestyle='--')
	pylab.axvline(125, color='k', linestyle='--')





#if epochsBack > 0:
#	historyFeatures = numpy.zeros((explanatoryFeatures.shape[0], explanatoryFeatures.shape[1]*(epochsBack+1)))

for epochHistory in [3]: #range(10):
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

	# put in sentence serial - can't leave in history, cos is too highly correlated...
	explanatoryFeatures = numpy.vstack((explanatoryFeatures.T, wordFeatures['sentenceSerial'])).T
	features.append('sentenceSerial')
	legendLabels.append('sentenceSerial')

	#explanatoryFeatures = mynormalise(explanatoryFeatures)	

	#pylab.figure(); pylab.imshow(explanatoryFeatures,interpolation='nearest', aspect='auto'); pylab.show()


	#1/0

	for t in range(epochSamples):
		#print 'fitting at timepoint',t
		# NOTES # tried a load of different versions, and straight linear regression does as well as any of them, measured in terms of R^2
		#lm = sklearn.linear_model.LinearRegression(fit_intercept=True, normalize=True)
		#lm = sklearn.linear_model.RidgeCV(fit_intercept=True, normalize=True, alphas=regParam) #, 10000, 100000])
		lm = sklearn.linear_model.LassoLars(alpha=0.0001) #(alpha=1.0, fit_intercept=True, verbose=False, normalize=True, precompute='auto', max_iter=500, eps=2.2204460492503131e-16, copy_X=True)


		#lm = sklearn.linear_model.Ridge(fit_intercept=True, normalize=True)
		#lm = sklearn.linear_model.RidgeCV(fit_intercept=True, normalize=True, alphas=[1000000000]) #, 10000, 100000]) # with one test, and 9 explanatory variables, found that got alpha down to .01 for the times when we have a big R^2
		#lm = sklearn.linear_model.RidgeCV(fit_intercept=True, normalize=True, alphas=[1]) #, 10000, 100000])
		#lm = sklearn.linear_model.RidgeCV(fit_intercept=True, normalize=True, alphas=[1e-6,1e-4,1e-2,1e+0,1e+2, 1e+4, 1e+6]) #, 10000, 100000])
		#lm = sklearn.linear_model.ElasticNetCV(0.5) # rho (l1/l2 balance) parameter range suggested by doc pages
		#lm = sklearn.linear_model.ElasticNetCV([.1, .5, .7, .9, .95, .99, 1]) # rho (l1/l2 balance) parameter range suggested by doc pages
	# found that ended up taking rho of 1, and zero betas, so trying more L2 biased rho (documentation seems to be contradictory about l1/l2 aka lasso/ridge)
		trainX = mynormalise(explanatoryFeatures)
		trainY = mynormalise(wordEpochs[:,t])
		#trainX = explanatoryFeatures
		#trainY = wordEpochs[:,t]
		trainedLM = lm.fit(trainX,trainY)
		modelParameters.append(lm)
		#guessY = lm.predict(testX)
		#guessTestSemantics[:,dim] = guessY
		modelTrainingFit.append(adjustR2(lm.score(trainX,trainY), trainX.shape[1], trainX.shape[0]))
		#modelTestCorrelation.append(numpy.corrcoef(guessTestSemantics[:,dim],realTestSemantics[:,dim])[1,0])
		#print '\t\tdone, explaining R^2 of',modelTrainingFit[-1],'reg param',trainedLM.alpha_,'betas',modelParameters[-1].coef_ 	
		#print '\t\tdone, explaining R^2 of',modelTrainingFit[-1],'betas',modelParameters[-1].coef_, lm.alpha_ #'chose reg param of',lm.best_alpha # lm.alphas_[0] # lm.best_alpha for ridge; lm.alphas_[0] for lassolars
		#guessTestSemantics[testTrial,dim] = lm.predict(testX[testTrial,:])

	betaMatrix = numpy.array([p.coef_ for p in modelParameters])
	neatLabels = [l.replace(re.match(r'[^-]+',l).group(0), labelMap[re.match(r'[^-]+',l).group(0)]) for l in legendLabels if re.match(r'[^-]+',l).group(0) in labelMap]
	legendLabels = numpy.array(legendLabels)
	#numFeaturesDisplay = len(legendLabels)
	neatLabels = numpy.array(neatLabels)


	f = pylab.figure()
	s = pylab.subplot(2,2,1)
	pylab.title('R-squared '+str(trainedLM))
	pylab.plot(modelTrainingFit, linewidth=2)
	commonPlotProps()
	s = pylab.subplot(2,2,2)
	#pylab.plot(betaMatrix, '-', linewidth=2)
	pylab.plot(betaMatrix[:,:7], '-', linewidth=2)
	pylab.plot(betaMatrix[:,7:], '--', linewidth=2)
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
	pylab.plot(numpy.mean(epochedGroupMeanAreasensorsMeanSensors,axis=0), linewidth=2)
	pylab.title('ERF')
	commonPlotProps()

	pylab.savefig('meg_testfig.png')
	print 'history %d, mean model fit over -0.5s to +1.0s: %.5f, max is %.5f' % (epochHistory, numpy.mean(modelTrainingFit[62:250]), numpy.max(modelTrainingFit[62:250]))

	#interestingFeatures = numpy.argsort(numpy.sum(numpy.abs(betaMatrix),axis=0))[::-1][:7]
	#interestingBetas = betaMatrix[:,interestingFeatures]
	#interestingLabels = legendLabels[interestingFeatures]
	#interestingExpFeatures = explanatoryFeatures[:,interestingFeatures]

	#f = pylab.figure()
	#s = pylab.subplot(2,2,1)
	#pylab.title('R-squared '+str(trainedLM))
	#pylab.plot(modelTrainingFit)
	#commonPlotProps()
	#s = pylab.subplot(2,2,2)
	#pylab.plot(interestingBetas)
	#pylab.legend(interestingLabels)
	#pylab.title('betas for %d most interesting normed variables' % len(interestingFeatures))
	#commonPlotProps()
	#s = pylab.subplot(2,2,3)
	#pylab.title('correlations between explanatory variables')
	#pylab.imshow(numpy.corrcoef(interestingExpFeatures.T),interpolation='nearest') # leave out the dummy one
	#pylab.yticks(range(len(interestingLabels)-1),interestingLabels[1:])
	#pylab.ylim((-0.5,len(interestingLabels)-1-0.5))
	#pylab.xticks(range(len(interestingLabels)-1),interestingLabels[1:], rotation=90)
	#pylab.xlim((-0.5,len(interestingLabels)-1-0.5))
	#pylab.colorbar()
	#s = pylab.subplot(2,2,4)
	#pylab.plot(numpy.mean(epochedGroupMeanAreasensorsMeanSensors,axis=0))
	#pylab.title('ERF')
	#commonPlotProps()



