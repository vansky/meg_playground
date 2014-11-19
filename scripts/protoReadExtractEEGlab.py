"""Prototype routine to read in EEG dataset in EEGlab format, and return in epoched form.
To be integrated into PyMVPA later. Updated for PyMVPA v0.6 (dev)
brian.murphy@unitn.it, 6th April 2011.
"""
### DEV NOTES ###
#
#	- At the moment, I'm using the mat_struct datatype to load the EEGlab structure into. According to online docs (for scilab.io.loadmat), this option is being deprecated and may be removed, replaced with the Numpy Record Arrays. I've tried using record arrays, and they won't work for me. Instead of a field (attibute) based access (e.g. EEG.srate = 120), I'm returned an undifferentiated tuple, containing each substructure - so the data is there, but you can't find it by it's field name. I guess this is the reason - it seems Numpy Record Arrays are intended for strictly sized arrays of R rows by F fields (ie they are not ragged arrays). If that is correct, then they are not an appropriate storage for this structure (e.g. EEG.setname contains a single string, EEG.data contains a huge numeric array, and EEG.chanlocs contains a series of structures - though in this last case a Record Array would be good for this subelement). Again, if that is right, we should write to the Scipy people and point out that many matlab structures cannot be represented in record arrays.
#	- I'm storing the EEG metadata, and signal data separately. I then delete (using del()) the raw signal and ICA signal data from the original EEG object. I haven't checked if this actually has any effect on memory usage (it should almost half it, since the ICA data is as big as the raw data). It might be necessary to do some garbage collection
#	- sometimes EEGLAB stores the actual data as separate files (.dat, .fdt, .icafdt at least), with .set only holding the metadata. In this case the eLDS.data (EEG.data) field holds the file name containing data. According to source code of EEGlab routines (pop_saveset, which calls floatwrite, which calls fwrite), data is stored as 4-byte floats, little-endian. FDT files have the data in sample order (sample1 (channel1 channel2 ... channelN) ... sampleN). DAT files are in channel order (channel1 (sample1 sample2 ... sampleN) ... channelN). At first I tried using the python struct package (struct.unpack('<f',x)) after reading the whole binary file in with open(filename, 'rb'), but this works on single elements at a time, so would take longer (the looping over each f.read(4) element, and unpacking I guess). Instead I used scipy.io.numpyio.fread that reads and unpacks the whole file in one command. Then this was deprecated and now I use the much simpler numpy.fromfile.
#
#################


# IMPORT LIBRARIES
# python numerical stuff
import scipy.io as Sio
import numpy as N
# logging and status messages
import logging as L
# other utility classes
import os.path # parse out parts of file-system paths 



# MAIN FUNCTION TO LOAD EEGLAB DATASET FILE, AND RETURN EPOCHED
def load(eegFile, triggersOfInterest, epochEndSec=1, epochStartSec=0):
	"""Main function loads EEGlab dataset, and returns tuple of epoched data together with metadata (descriptive data).
		Input:
			eegFile - full path to the EEGlab .set file
			triggersOfInterest - list of strings and/or integers corresponding to triggers, as found in the EEGlab EEG.event(i).type field
			epochEndSec - the end point (in seconds) for each epoch, relative to the latency of the trigger. Defaults to 1s
			epochStartSec - ditto for the start point (in seconds) for each epoch. Negative values will make the epoch start before the trigger. Defaults to 0s
		Output: tuple of the following
			rawSignalData - EEG data channels x timepoints
			metaData - all non-data content of the original EEGlab EEG object in matlab, as a Scipy mat_struct object (see foot of code for example of EEG structure)
			trials - which trials were interesting, with trigger values and point in time; trials x (trigger, latency)
			epochedSignalData - EEG data epochs x channels x timepoints
			epochSliceTimepoints - array of arrays, each containing the indices for the a single epoch - will split a rawSignal by epoch
	"""
	# IMPORT EEGLAB DATASET (is a matlab structure file)
	L.info('loading EEGlab file: '+eegFile)
	eLDS = Sio.loadmat(eegFile, appendmat=False, chars_as_strings=True, squeeze_me=True, struct_as_record=False)['EEG'] # don't apppend "mat" to file name; convert any char arrays to strings; squeeze all pointless matlab single dimensions out; load structure as scipy mat_struct type
	# note that mat_struct will be removed in future versions of SciPy. When I tried using struct_as_record=True, I got the structure as a tuple, without being able (as far as I could see) to refer to fields by their name, even though the Numpy Record Array docs suggested this should work... like EEG['srate'] or maybe EEG.srate
	#1/0
	L.debug(str(dir(eLDS))+'\n type: '+str(type(eLDS)))
	L.info('This EEGlab data set has %d channels, and is %0.1f s long at a sampling rate of %0.0f Hz', eLDS.nbchan, round(eLDS.pnts/eLDS.srate), eLDS.srate)
	L.info('set name is: '+str(eLDS.setname))

	# STORE SIGNAL DATA, AND METADATA
	if isinstance(eLDS.data, N.ndarray) and len(eLDS.data.shape) > 0: # data stored directly in .set file
		rawSignalData = eLDS.data
	else:
		if not isinstance(eLDS.data, basestring): # or isinstance(eLDS.data.item(), basestring): # haven't checked if works with normal data in single .set file!
		#if isinstance(eLDS.data.item(), basestring):
			eLDS.data = eLDS.data.item()
		L.info('data stored in separate file, loading: '+eLDS.data)
		f = open(os.path.join(os.path.dirname(eegFile),eLDS.data),'rb') # open the data file as a binary (data file name plus any path from original EEG set file)
		rawSignalData = N.fromfile(f,"<f4") # read all samples, as 4-byte floats, little-endian - see EEGLab's pop_saveset and fwrite to see how they save pure data files [used to use deprecated Sio.numpyio.fread(f,eLDS.nbchan*eLDS.pnts,'f','f',0)]
		if eLDS.data.lower().endswith('fdt'): # sample order
			rawSignalData = rawSignalData.reshape(eLDS.pnts,eLDS.nbchan).transpose()
		elif eLDS.data.lower().endswith('dat'): # channel order
			rawSignalData = rawSignalData.reshape(eLDS.nbchan,eLDS.pnts)
	#del(eLDS.data) ### check that this does actually save memory - i.e. that metaData is actually smaller than original structure 
	metaData = eLDS

	# IDENTIFY TRIALS OF INTEREST, AND EXTRACT CORRESPONDING EPOCHS
	trialTriggers = []; # list of triggers
	trialLatencies = []; # ... and their latencies in timepoints (aka samples) to make record array of trials
	epochSliceTimepoints = []; # embedded lists containing indices for each epoch
	epochStartTimepoints = round(epochStartSec*eLDS.srate) # number of timepoints (aka samples) before trigger to include in epoch
	epochEndTimepoints = round(epochEndSec*eLDS.srate) # ... ditto after the trigger
	for event in eLDS.event: 	
	# how to see stucture of one trigger: [(a, eval("eLDS.event[0].%s" % a)) for a in dir(eLDS.event[0]) if not a.startswith('_')]
		if event.type in triggersOfInterest:
			trialTriggers.append(event.type)
			trialLatencies.append(event.latency)
			epochSliceTimepoints.append(range(int(round(event.latency+epochStartTimepoints)),int(round(event.latency+epochEndTimepoints))))
	# create record array of trial IDs and their position in samples of the recording, allows field and entry-wise referencing: trials.timepoint, trials[5], trials[5].trigger, trials.trigger[5]
	trials = N.core.records.fromarrays([trialTriggers, trialLatencies],names='trigger,timepoint')
	L.info('Extracted %d trials of interest, with epochs running from %-.2fs to %+.2fs relative to triggers' % (len(trials), epochStartSec, epochEndSec))
	# create epoch-based view on data, using epochs of particular type
	epochedSignalData = rawSignalData[:,epochSliceTimepoints].swapaxes(0,1) # epochs x channels x timepoints
	# print a random epoch to validate
	
	
	return (rawSignalData, metaData, trials, epochedSignalData, epochSliceTimepoints)
	
# DEMO IF RUN AS MAIN SCRIPT
if __name__ == '__main__':
	L.basicConfig(level=L.DEBUG) 
	eegFile = 'FedExpAImagesOct08p6x2v2-it-c-2_resamp120_filt1-50_labelled_ICAed_handPruned.set'
	L.info('Demo on this file: '+eegFile)
	(rawSignalData, metaData, trials, epochedSignalData, epochSliceTimepoints) = load(eegFile, ['S%3d' % i for i in range(1,128)], 1, 0)
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
