"""Prototype routine to read extract epochs from continuous data, and return in epoched form.
To be integrated into PyMVPA later.
brian.murphy@unitn.it, 9th Jan 2011.
"""
# IMPORT LIBRARIES
# python numerical stuff
#import scipy.io as Sio
import numpy as N
#import pylab as P
# logging and status messages
import logging as L
# other utility classes
import os.path # parse out parts of file-system paths



# MAIN FUNCTION TO LOAD EEGLAB DATASET FILE, AND RETURN EPOCHED
def epochify(rawSignalData, samplingRate, trialTimeSeconds, epochEndSec=1, epochStartSec=0):
	"""Main function loads EEGlab dataset, and returns tuple of epoched data together with metadata (descriptive data).
		Input:
			rawSignalData - continuous EEG data channels x timepoints
			samplingRate - 
			trialTimeSeconds - list of reference time points (e.g. stimulus onset) for each trial (in secs)
			epochEndSec - the end point (in seconds) for each epoch, relative to the latency of the trigger. Defaults to 1s
			epochStartSec - ditto for the start point (in seconds) for each epoch. Negative values will make the epoch start before the trigger. Defaults to 0s
		Output: tuple of the following
			trials - which trials were interesting, with trigger values and point in time; trials x (trigger, latency)
			epochedSignalData - EEG data epochs x channels x timepoints
			epochSliceSamples - array of arrays, each containing the indices for the a single epoch - will split a rawSignal by epoch
	"""

	# CHECK INPUT
	if epochEndSec <= epochStartSec:
		print 'ERROR: epoch ends: ',epochEndSec,', before it starts:',epochStartSec
		return ()

	# IDENTIFY TRIALS OF INTEREST, AND EXTRACT CORRESPONDING EPOCHS
	trialTimeSamples = []; # latencies in timepoints (aka samples) to make record array of trials
	epochSliceSamples = []; # embedded lists containing indices for each epoch
	epochStartSamples = round(epochStartSec*samplingRate) # number of timepoints (aka samples) before trigger to include in epoch
	epochEndSamples = round(epochEndSec*samplingRate) # ... ditto after the trigger
	for timeSeconds in trialTimeSeconds:
		refSample = N.round(timeSeconds*samplingRate) 	
		trialTimeSamples.append(refSample)
		epochSliceSamples.append(range(int(round(refSample+epochStartSamples)),int(round(refSample+epochEndSamples))))
		#print 'at time secs/samps',timeSeconds,refSample,'taking sample range',int(round(refSample+epochStartSamples)),int(round(refSample+epochEndSamples))
	# create record array of trial times (in seconds and samples)
	trials = N.core.records.fromarrays([trialTimeSeconds, trialTimeSamples],names='seconds,samples')
	# create epoch-based view on data, using epochs of particular type
	epochedSignalData = rawSignalData[:,epochSliceSamples].swapaxes(0,1) # epochs x channels x timepoints
	# print a random epoch to validate
	L.info('Extracted %d trials of interest, with epochs running from %-.2fs to %+.2fs relative to triggers' % (len(trials), epochStartSec, epochEndSec))
	
	
	return (trials, epochedSignalData, epochSliceSamples)
	
# DEMO IF RUN AS MAIN SCRIPT
if __name__ == '__main__':
	L.basicConfig(level=L.DEBUG) 
	eegFile = 'FedExpAImagesOct08p6x2v2-it-c-2_resamp120_filt1-50_labelled_ICAed_handPruned.set'
	L.info('Demo on this file: '+eegFile)
	(rawSignalData, metaData, trials, epochedSignalData, epochSliceSamples) = load(eegFile, ['S%3d' % i for i in range(1,128)], 1, 0)
	P.plot(epochedSignalData[range(3),59].transpose()) # plot first three epochs on Oz
	P.show()
	
## TO DO ##
#
# no parameter checking - e.g. files that don't exist, triggers that don't exist, unreasonable epoch bounds
# [nuffink]

# EEG structure from EEGlab in Matlab:
'''
>> EEG

EEG = 

             setname: [1x77 char]
            filename: [1x80 char]
            filepath: '/mnt/data/brianEEG/eeglabData/'
             subject: ''
               group: ''
           condition: ''
             session: []
            comments: 'Original file: FedExpAImagesOct08p6x2v2_it_c_2.eeg'
              nbchan: 64
              trials: 1
                pnts: 470063
               srate: 120
                xmin: 0
                xmax: 3.9172e+03
               times: []
                data: [64x470063 single]
              icaact: [62x470063 single]
             icawinv: [64x62 double]
           icasphere: [64x64 double]
          icaweights: [62x64 double]
         icachansind: [1x64 double]
            chanlocs: [1x64 struct]
          urchanlocs: []
            chaninfo: [1x1 struct]
                 ref: 'common'
               event: [1x1049 struct]
             urevent: [1x1049 struct]
    eventdescription: {[]  []  []  []  []  []  []}
               epoch: []
    epochdescription: {}
              reject: [1x1 struct]
               stats: [1x1 struct]
            specdata: []
          specicaact: []
          splinefile: ''
       icasplinefile: ''
              dipfit: []
             history: [1x1742 char]
               saved: 'yes'
                 etc: []
             spedata: []

>> 
'''
