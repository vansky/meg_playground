"""Prototype routine to extract spectral estimate data (frequency domain amplitude and phase) at a specified of frequency resolutions, from epoched EEG data (epochs x channels x timepoints). Uses an FFT with a Hamming window, applied to each epoch - and so the epochs used should include some padding.
To be integrated into PyMVPA later.
brian.murphy@unitn.it, 23rd June 2010.
"""
	
	
# IMPORT LIBRARIES
# python numerical stuff
import scipy.fftpack as Sfft
import scipy.signal as Ssig
import numpy as N
#import pylab as P
# logging and status messages
import logging as L
import time
import sys # to flush print for processing progress indicator
# for interpreting text parameters
import re



# UTILITY FUNCTIONS
# print progress indicator on same line (overprinting)
def inPlacePrint(message):
	"""utility function to print progress indicator on same line (overprinting). Takes single printable argument."""
	print '\r',message,
	sys.stdout.flush()
	
# PROTO REVERSE MAPPING METHODS
def unmapFeatures(features, trialsSlice, channelsSlice, featuresPerChannel):
	"""given features in trials x features array, plus required trials and channels, returns them in form trials x channels x features"""
	#featuresPerChannel = features.shape[1]/numChannels
	channelsFeatureSlice = [range(c*featuresPerChannel,(c+1)*featuresPerChannel) for c in channelsSlice]
	return features[trialsSlice][:,channelsFeatureSlice] # [trialsSlice,channelsFeatureSlice] doesn't work, don't know why

# MAIN METHOD TO EXTRACT FEATURES
def mapFeatures(epochedSignalData, samplingRate, freqRes=0, windowShape='hann', featureType='both', fftLength=0):
	"""Main function takes epochedSignalData, and specifications for windowed FFT, and returns vector of values (machine learning features) for each trial (aka epoch; aka sample in PyMVPAish). By default chooses length of epoch as FFT width.
		Input:
			epochedSignalData - EEG data epochs x channels x timepoints
			freqRes - length of fft, affects frequencies to analyse, up to half of sampling rate. Defaults to length of epoch
			samplingRate - the usual (number of EEG samples per second, in Hz - no effect on analysis, just for intepreting scale of FFT output)
			windowShape - kind of window to apply to signal before FFT - defaults to 'hann'. Other possibilities: 'flat', 'hamming' (non-zero ends), 'blackman' (leaner tails), 'bartlet' (triangle) see http://docs.scipy.org/doc/scipy/reference/signal.html#window-functions for full list
	"""

	epochNumTimepoints = epochedSignalData.shape[2] # number of timepoints (samples in EEGish) in each epoch

	# work out frequency bands to extract spectral features
	if freqRes==0:
		freqRes=round(epochNumTimepoints/2);
	if fftLength==0:
		fftLength = freqRes*2;
#	if fftLength < epochNumTimepoints:
#		fftLength = epochNumTimepoints
	maxFreq = float(samplingRate)/2
	bandWidth = float(maxFreq)/freqRes
	spectralFrequencies = N.linspace(bandWidth/2,maxFreq-bandWidth/2,freqRes)
	#spectralFrequencies = Sfft.fftshift(Sfft.fftfreq(fftLength,float(1)/samplingRate))[fftLength/2+1:]
	L.info('will extract features for '+str(freqRes)+ ' equally spaced frequencies-bands of width '+str(bandWidth)+'Hz up to '+str(maxFreq)+'Hz, with an FFT window of length '+str(fftLength)+' and this windowing function: '+windowShape)
	L.debug(spectralFrequencies)
	
	# PRE-ALOCATE STORAGE FOR FEATURES
	numTrials = epochedSignalData.shape[0]
	numChannels = epochedSignalData.shape[1]
	L.info('have %d trials of %d channel data' % (numTrials, numChannels))
	if(featureType.lower()=='both') or (re.findall('phase',featureType.lower()) and re.findall('amp',featureType.lower())):
		numFeatures = numChannels*freqRes*2 # remember, there are two features per fft point - amplitude and phase
		doFreq = True
		doPhase = True
		L.info('will extract %d phase and amplitude features per trial [option %s] (progress indicator below shows trials)' % (numFeatures, featureType))
	elif re.findall('phase',featureType.lower()):
		numFeatures = numChannels*freqRes
		doFreq = False
		doPhase = True
		L.info('will extract %d phase features per trial [option %s] (progress indicator below shows trials)' % (numFeatures, featureType))
	else:
		numFeatures = numChannels*freqRes
		doFreq = True
		doPhase = False
		L.info('will extract %d amplitude features per trial [option %s] (progress indicator below shows trials)' % (numFeatures, featureType))
	mappedTrialFeatures = N.zeros((numTrials, numFeatures))
	
	# STEP THROUGH TRIALS AND CHANNELS, GENERATING FEATURES FOR EACH
	startTime = time.time() # to give estimate of time remaining
	for trial in range(numTrials): # for each trial
		trialFeatureVector = N.array([]) # will have ordering ((firstChannel (fftAmp fftPhase) ... (lastChannel))
		for channel in range(numChannels): # ... for each channel
			windowVector = eval('Ssig.'+windowShape+'(epochNumTimepoints)')
			windowedSignalEpoch = epochedSignalData[trial,channel,:]*windowVector # extract the relevant part of the raw signal for this trial on this channel
			fftDecomposition = Sfft.fftshift(Sfft.fft(windowedSignalEpoch, fftLength))[fftLength/2+1:]
			# extract amp and phase, and smooth to desired frequency resolution
			if doFreq:
				fftAmp = Ssig.resample(abs(fftDecomposition),freqRes,window='flat')/fftLength # scaled so invariant to fft length - not sure what units are
				trialFeatureVector = N.append(trialFeatureVector,[fftAmp])
			if doPhase:
				fftPhase = Ssig.resample(N.angle(fftDecomposition),freqRes,window='flat') # in radians - not sure will tell us anything unless very stable 
				trialFeatureVector = N.append(trialFeatureVector,[fftPhase])
			inPlacePrint("... [progress %d%%] %ds elapsed, est. %ds remaining:\t channel %d of %d\ttrial %d of %d" % (round(100*(trial+1)/float(numTrials)), time.time()-startTime, (1-(trial+1)/float(numTrials))*(time.time()-startTime)/((trial+1)/float(numTrials)), channel+1, numChannels, trial+1, numTrials)) # progress indicator
		mappedTrialFeatures[trial,:] = trialFeatureVector
	print "\n" # line return after progress indicator finished
		
	return (mappedTrialFeatures, spectralFrequencies)

# DEMO IF RUN AS MAIN SCRIPT
if __name__ == '__main__':
	import protoReadExtractEEGlab
	L.basicConfig(level=L.DEBUG) 
	eegFile = 'FedExpAImagesOct08p6x2v2-it-c-2_resamp120_filt1-50_labelled_ICAed_handPruned.set'
	L.info('Demo on this file: '+eegFile)
	(rawSignalData, metaData, trials, epochedSignalData, epochSliceTimepoints) = protoReadExtractEEGlab.load(eegFile, ['S%3d' % i for i in range(1,5)], 1, 0)
	(features, trendResolutions, resolutionIntervals, resolutionLabels) = mapFeatures(epochedSignalData, 100)
	
## TO DO ##
#
# no parameter checking for things that are stupid (e.g. resolution higher than length of epoch), or would make it misfunction (e.g. negative resolutions)
# add visualisation for the demo
#
# [nuffink else]

