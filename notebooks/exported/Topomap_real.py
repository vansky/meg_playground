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
%autosave 0

# <codecell>

#basic imports

%pylab inline
import time
import pickle
import cPickle
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

import mne

# <codecell>

#custom imports
%cd ../scripts

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
%cd ../MEG_data

#*# MARTY #*# choose a file - I found participant V to be pretty good, and 0.01 to 50Hz filter is pretty conservative #*#
(megFileTag1, megFile1) = ('V_TSSS_0.01-50Hz_@125', 'v_hod_allRuns_tsss_audiobookPrepro_stPad1_lp50_resamp125_frac10ICAed.set')#_hp0.010000.set')
(megFileTag2, megFile2) = ('A_TSSS_0.01-50Hz_@125', 'aud_hofd_a_allRuns_tsss_audiobookPrepro_stPad1_lp50_resamp125_frac10ICAed_hp0.010000.set')
(megFileTag3, megFile3) = ('C_TSSS_0.01-50Hz_@125', 'aud_hofd_c_allRuns_tsss_audiobookPrepro_stPad1_lp50_resamp125_frac10ICAed_hp0.010000.set')

## put your on properties in here, as a .tab file of similar format (tab delimited, and field names in a first comment line - should be easy to do in excel...)
#to get the V7.tab:
#  python ../scripts/buildtab.py hod_JoeTimes_LoadsaFeaturesV3.tab hod.wsj02to21-comparativized-gcg15-1671-4sm.fullberk.parsed.gcgbadwords > hod_JoeTimes_LoadsaFeaturesV4.tab
#  python ../scripts/addsentid.py hod_JoeTimes_LoadsaFeaturesV4.tab > hod_JoeTimes_LoadsaFeaturesV5.tab
#  python ../scripts/addparserops.py hod_JoeTimes_LoadsaFeaturesV5.tab hod.wsj02to21-comparativized-gcg15-1671-4sm.fullberk.parsed.gcgparseops > hod_JoeTimes_LoadsaFeaturesV6.tab
#  python ../scripts/addtotsurp.py hod_JoeTimes_LoadsaFeaturesV6.tab hod.totsurp > hod_JoeTimes_LoadsaFeaturesV7.tab
tokenPropsFile = 'hod_JoeTimes_LoadsaFeaturesV7.tab' # best yet! have automatic Stanford tagging, several word length and freq measures, and also the 2 and 3 token back-grams

# WHICH CHANNELS TO LOOK AT AS ERFS
#*# MARTY #*# decide which channels to use - channels of interest are the first few you can look at in an ERF, and then from them you can choose one at a time with "channelToAnalyse" for the actual regression analysis #*#
channelLabels = ['MEG0111', 'MEG0121', 'MEG0131', 'MEG0211', 'MEG0212', 'MEG0213', 'MEG0341']
#?# this way of doing things was a slightly clumsy work-around, cos I didn't have enough memory to epoch all 306 channels at one time

# LOAD WORD PROPS
#*# MARTY #*# change dtype to suit the files in your .tab file #*#
tokenProps = scipy.genfromtxt(tokenPropsFile,
                              delimiter='\t',names=True,
                              dtype="i4,f4,f4,S50,S50,i2,i2,i2,S10,f4,f4,f4,f4,f4,f4,f4,f4,f4,f4,f4,i1,>i4,S4,f8")
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
(contSignalData, metadata, trackTrials, tokenProps, audioSignal, samplingRate, numChannels) = loadBookMEGWithAudio(megFile1, tokenPropsPickle, triggersOfInterest, epochEnd, epochStart, icaComps=False)
del contSignalData
del trackTrials

# <codecell>

print(metadata.chanlocs[0]._fieldnames)

# <codecell>

#Get blank sensor map
%cd ../stats/
resultType = 'wide'
resultsFile = 'signifresults.multifactor.cpk' # % (resultType)

rsq = cPickle.load(open(resultsFile))

# EXTRACT SENSOR LOCATIONS FROM METADATA                                                                                                                       
sensorLocations = {}
# don't have the data at hand, but here it would be good to first extract the X/Y co-ordinates of all                                                          
for sensor in metadata.chanlocs:
    # not sure if the structure is like this - will have to check in metadata returned by the python script
    sensorLocations[sensor.labels] = (-sensor.Y, sensor.X) 
    #sensorLocations[sensor.labels] = (sensor.X, sensor.Y)

# DO A TOPOPLOT FOR EACH BAND AND STAT (three beside eachother, one for each type of sensor)                                                                   
for stat in ['p']:#, 'max']:
    for band in ['alpha']: # add more bands if you want                                                                                           
        #sensorPositions = [] # in horizontal plane, looking downwards                                                                                  
        #sensorValues = [] # r^2 or whatever other value we want to do map of                                                                           

        pylab.figure()
        for sensorType in ['3']:
            sensorPositions = [] # in horizontal plane, looking downwards                                                                                  
            sensorValues = [] # r^2 or whatever other value we want to do map of
            sensorLabels = []
            # sensors that end in 1 are magnetometers, those ending in 2 or 3 are gradiometers of different orientations                                                                                                                                                              
            for sensorLabel in rsq[stat].keys(): # 'ave' or 'max'? not sure
                if sensorLabel.endswith(sensorType):
                    sensorValues = numpy.append(sensorValues,rsq[stat][sensorLabel][band]['syndepth']) # e.g. rsq['ave']['MEG2643']['alpha']                                
                    sensorPositions = numpy.append(sensorPositions,sensorLocations[sensorLabel]) # this is already an X-Y tuple
                    sensorLabels = numpy.append(sensorLabels,sensorLabel)
                sensorValues = numpy.array(sensorValues)
                sensorPositions = numpy.array(sensorPositions)
                sensorLabels = numpy.array(sensorLabels)
            #subplotCounter += 1
            #pylab.subplot(1,3,subplotCounter)
            #print(sensorValues.shape, sensorPositions.reshape(-1,2).shape)
            if stat == 'p':
                for i,s in enumerate(sensorValues):
                    if not np.isfinite(s):
                        sensorValues[i] = 1.0
            im, cn = mne.viz.plot_topomap(sensorValues, sensorPositions.reshape(-1,2),
                                          names=sensorLabels,show_names=True,
                                          vmin=0.0,vmax=0.05)
            #im, cn = mne.viz.plot_topomap(sensorValues, sensorPositions.reshape(-1,2))
            pylab.colorbar(im,fraction=0.046, pad=0.04)
            pylab.title('Sensor map')
            #pylab.legend(loc='upper right')
        pylab.savefig('graphics/large-post-meg_%s-%s_%s.png' % (resultType,stat,band))
        pylab.show()

# <codecell>

#Get blank sensor map
%cd ../stats/
resultType = 'wide'
resultsFile = 'signifresults.multifactor.dev.bigscale.cpk' # % (resultType)

rsq = cPickle.load(open(resultsFile))

# EXTRACT SENSOR LOCATIONS FROM METADATA                                                                                                                       
sensorLocations = {}
# don't have the data at hand, but here it would be good to first extract the X/Y co-ordinates of all                                                          
for sensor in metadata.chanlocs:
    # not sure if the structure is like this - will have to check in metadata returned by the python script
    sensorLocations[sensor.labels] = (-sensor.Y, sensor.X) 
    #sensorLocations[sensor.labels] = (sensor.X, sensor.Y)

# DO A TOPOPLOT FOR EACH BAND AND STAT (three beside eachother, one for each type of sensor)                                                                   
for stat in ['p']:#, 'max']:
    for factor in ['sentenceSerial','syndepth','bigramLogProbBack_COCA','logFreq_ANC']: # add more bands if you want                                                                                           
        #sensorPositions = [] # in horizontal plane, looking downwards                                                                                  
        #sensorValues = [] # r^2 or whatever other value we want to do map of                                                                           

        pylab.figure()
        subplotCounter = 0
        for sensorType in ['1','2','3']:
            sensorPositions = [] # in horizontal plane, looking downwards                                                                                  
            sensorValues = [] # r^2 or whatever other value we want to do map of
            sensorLabels = []
            # sensors that end in 1 are magnetometers, those ending in 2 or 3 are gradiometers of different orientations                                                                                                                                                              
            for sensorLabel in rsq[stat].keys(): # 'ave' or 'max'? not sure
                if sensorLabel.endswith(sensorType):
                    sensorValues = numpy.append(sensorValues,rsq[stat][sensorLabel]['alpha'][factor]) # e.g. rsq['ave']['MEG2643']['alpha']                                
                    sensorPositions = numpy.append(sensorPositions,sensorLocations[sensorLabel]) # this is already an X-Y tuple
                    sensorLabels = numpy.append(sensorLabels,sensorLabel)
                sensorValues = numpy.array(sensorValues)
                sensorPositions = numpy.array(sensorPositions)
                sensorLabels = numpy.array(sensorLabels)
            subplotCounter += 1
            pylab.subplot(1,3,subplotCounter)
            #print(sensorValues.shape, sensorPositions.reshape(-1,2).shape)
            if stat == 'p':
                for i,s in enumerate(sensorValues):
                    if not np.isfinite(s):
                        sensorValues[i] = 1.0
            #im, cn = mne.viz.plot_topomap(sensorValues, sensorPositions.reshape(-1,2),
            #                              names=sensorLabels,show_names=True,
            #                              vmin=0.0,vmax=0.05)
            im, cn = mne.viz.plot_topomap(sensorValues, sensorPositions.reshape(-1,2),
                                          vmin=0.0,vmax=0.05)
            pylab.colorbar(im,fraction=0.046, pad=0.04)
            pylab.title('%s map' % (factor))
            #pylab.legend(loc='upper right')
        pylab.savefig('graphics/meg_%s-%s_%s.png' % (stat,'alpha',factor))
        pylab.show()

# <codecell>

#Get blank sensor map
%cd ../stats/
resultType = 'wide'
resultsFile = 'signifresults.multifactor.dev.bigscale.cpk' # % (resultType)

rsq = cPickle.load(open(resultsFile))

# EXTRACT SENSOR LOCATIONS FROM METADATA                                                                                                                       
sensorLocations = {}
# don't have the data at hand, but here it would be good to first extract the X/Y co-ordinates of all                                                          
for sensor in metadata.chanlocs:
    # not sure if the structure is like this - will have to check in metadata returned by the python script
    sensorLocations[sensor.labels] = (-sensor.Y, sensor.X) 
    #sensorLocations[sensor.labels] = (sensor.X, sensor.Y)

# DO A TOPOPLOT FOR EACH BAND AND STAT (three beside eachother, one for each type of sensor)                                                                   
for stat in ['r2']:#, 'max']:
    for factor in ['syndepth']: # add more bands if you want                                                                                           
        #sensorPositions = [] # in horizontal plane, looking downwards                                                                                  
        #sensorValues = [] # r^2 or whatever other value we want to do map of                                                                           

        pylab.figure()
        subplotCounter = 0
        for sensorType in ['1','2','3']:
            sensorPositions = [] # in horizontal plane, looking downwards                                                                                  
            sensorValues = [] # r^2 or whatever other value we want to do map of
            sensorLabels = []
            # sensors that end in 1 are magnetometers, those ending in 2 or 3 are gradiometers of different orientations                                                                                                                                                              
            for sensorLabel in rsq[stat].keys(): # 'ave' or 'max'? not sure
                if sensorLabel.endswith(sensorType):
                    sensorValues = numpy.append(sensorValues,rsq[stat][sensorLabel]['alpha']) # e.g. rsq['ave']['MEG2643']['alpha']                                
                    sensorPositions = numpy.append(sensorPositions,sensorLocations[sensorLabel]) # this is already an X-Y tuple
                    sensorLabels = numpy.append(sensorLabels,sensorLabel)
                sensorValues = numpy.array(sensorValues)
                sensorPositions = numpy.array(sensorPositions)
                sensorLabels = numpy.array(sensorLabels)
            subplotCounter += 1
            pylab.subplot(1,3,subplotCounter)
            #print(sensorValues.shape, sensorPositions.reshape(-1,2).shape)
            if stat == 'p':
                for i,s in enumerate(sensorValues):
                    if not np.isfinite(s):
                        sensorValues[i] = 1.0
            #im, cn = mne.viz.plot_topomap(sensorValues, sensorPositions.reshape(-1,2),
            #                              names=sensorLabels,show_names=True,
            #                              vmin=0.0,vmax=0.05)
            im, cn = mne.viz.plot_topomap(sensorValues, sensorPositions.reshape(-1,2),
                                          vmin=0.0)
            pylab.colorbar(im,fraction=0.046, pad=0.04)
            pylab.title('%s map' % (factor))
            #pylab.legend(loc='upper right')
        pylab.savefig('graphics/meg_%s-%s_conf.png' % (stat,'alpha'))
        pylab.show()

# <codecell>

#Get blank sensor map
%cd ../MEG_data/
resultType = 'wide'
resultsFile = 'fitresults.%s.cpk'  % (resultType)

rsq = cPickle.load(open(resultsFile))

# EXTRACT SENSOR LOCATIONS FROM METADATA                                                                                                                       
sensorLocations = {}
# don't have the data at hand, but here it would be good to first extract the X/Y co-ordinates of all                                                          
for sensor in metadata.chanlocs:
    # not sure if the structure is like this - will have to check in metadata returned by the python script
    sensorLocations[sensor.labels] = (-sensor.Y, sensor.X) 
    #sensorLocations[sensor.labels] = (sensor.X, sensor.Y)

# DO A TOPOPLOT FOR EACH BAND AND STAT (three beside eachother, one for each type of sensor)                                                                   
for stat in ['ave']:#, 'max']:
    for band in ['alpha']: # add more bands if you want                                                                                           
        #sensorPositions = [] # in horizontal plane, looking downwards                                                                                  
        #sensorValues = [] # r^2 or whatever other value we want to do map of                                                                           

        pylab.figure()
        for sensorType in ['3']:
            sensorPositions = [] # in horizontal plane, looking downwards                                                                                  
            sensorValues = [] # r^2 or whatever other value we want to do map of
            sensorLabels = []
            # sensors that end in 1 are magnetometers, those ending in 2 or 3 are gradiometers of different orientations                                                                                                                                                              
            for sensorLabel in rsq[stat].keys(): # 'ave' or 'max'? not sure
                if sensorLabel.endswith(sensorType):
                    sensorValues = numpy.append(sensorValues,rsq[stat][sensorLabel][band]) # e.g. rsq['ave']['MEG2643']['alpha']                                
                    sensorPositions = numpy.append(sensorPositions,sensorLocations[sensorLabel]) # this is already an X-Y tuple
                    sensorLabels = numpy.append(sensorLabels,sensorLabel)
                sensorValues = numpy.array(sensorValues)
                sensorPositions = numpy.array(sensorPositions)
                sensorLabels = numpy.array(sensorLabels)
            #subplotCounter += 1
            #pylab.subplot(1,3,subplotCounter)
            #print(sensorValues.shape, sensorPositions.reshape(-1,2).shape)
            im, cn = mne.viz.plot_topomap(sensorValues, sensorPositions.reshape(-1,2),
                                          names=sensorLabels,show_names=True,
                                          vmin=None)
            #im, cn = mne.viz.plot_topomap(sensorValues, sensorPositions.reshape(-1,2))
            pylab.colorbar(im,fraction=0.046, pad=0.04)
            pylab.title('Sensor map')
            #pylab.legend(loc='upper right')
        pylab.savefig('graphics/large-post-meg_%s-%s_%s.png' % (resultType,stat,band))
        pylab.show()

# <codecell>

#Get blank sensor map
%cd ../MEG_data/
resultType = 'wide'
resultsFile = 'fitresults.%s.cpk'  % (resultType)

rsq = cPickle.load(open(resultsFile))

# EXTRACT SENSOR LOCATIONS FROM METADATA                                                                                                                       
sensorLocations = {}
# don't have the data at hand, but here it would be good to first extract the X/Y co-ordinates of all                                                          
for sensor in metadata.chanlocs:
    # not sure if the structure is like this - will have to check in metadata returned by the python script
    sensorLocations[sensor.labels] = (-sensor.Y, sensor.X) 
    #sensorLocations[sensor.labels] = (sensor.X, sensor.Y)

# DO A TOPOPLOT FOR EACH BAND AND STAT (three beside eachother, one for each type of sensor)                                                                   
for stat in ['ave']:#, 'max']:
    for band in ['alpha']: # add more bands if you want                                                                                           
        #sensorPositions = [] # in horizontal plane, looking downwards                                                                                  
        #sensorValues = [] # r^2 or whatever other value we want to do map of                                                                           

        pylab.figure()
        for sensorType in ['2']:
            sensorPositions = [] # in horizontal plane, looking downwards                                                                                  
            sensorValues = [] # r^2 or whatever other value we want to do map of
            sensorLabels = []
            # sensors that end in 1 are magnetometers, those ending in 2 or 3 are gradiometers of different orientations                                                                                                                                                              
            for sensorLabel in rsq[stat].keys(): # 'ave' or 'max'? not sure
                if sensorLabel.endswith(sensorType):
                    sensorValues = numpy.append(sensorValues,rsq[stat][sensorLabel][band]) # e.g. rsq['ave']['MEG2643']['alpha']                                
                    sensorPositions = numpy.append(sensorPositions,sensorLocations[sensorLabel]) # this is already an X-Y tuple
                    sensorLabels = numpy.append(sensorLabels,sensorLabel)
                sensorValues = numpy.array(sensorValues)
                sensorPositions = numpy.array(sensorPositions)
                sensorLabels = numpy.array(sensorLabels)
            #subplotCounter += 1
            #pylab.subplot(1,3,subplotCounter)
            #print(sensorValues.shape, sensorPositions.reshape(-1,2).shape)
            im, cn = mne.viz.plot_topomap(sensorValues, sensorPositions.reshape(-1,2),
                                          names=sensorLabels,show_names=True,
                                          vmin=0.0)
            #im, cn = mne.viz.plot_topomap(sensorValues, sensorPositions.reshape(-1,2))
            pylab.colorbar(im,fraction=0.046, pad=0.04)
            pylab.title('Sensor map')
            #pylab.legend(loc='upper right')
        pylab.savefig('graphics/large-post-meg_%s-%s_%s.png' % (resultType,stat,band))
        pylab.show()

# <codecell>

%cd ../MEG_data/
resultType = 'wide'
resultsFile = 'fitresults.%s.cpk' % (resultType)

rsq = cPickle.load(open(resultsFile))

# EXTRACT SENSOR LOCATIONS FROM METADATA                                                                                                                       
sensorLocations = {}
# don't have the data at hand, but here it would be good to first extract the X/Y co-ordinates of all                                                          
for sensor in metadata.chanlocs:
    # not sure if the structure is like this - will have to check in metadata returned by the python script
    #sensorLocations[sensor.labels] = (sensor.X, sensor.Y) 
    sensorLocations[sensor.labels] = (-sensor.Y, sensor.X)

# DO A TOPOPLOT FOR EACH BAND AND STAT (three beside eachother, one for each type of sensor)                                                                   
for stat in ['ave']:#, 'max']:
    for band in ['alpha']:#,'beta1','beta2','gamma',]: # add more bands if you want                                                                                           
        #sensorPositions = [] # in horizontal plane, looking downwards                                                                                  
        #sensorValues = [] # r^2 or whatever other value we want to do map of                                                                           

        pylab.figure()
        subplotCounter = 0
        for sensorType in ['1', '2', '3']: 
            sensorPositions = [] # in horizontal plane, looking downwards                                                                                  
            sensorValues = [] # r^2 or whatever other value we want to do map of
            # sensors that end in 1 are magnetometers, those ending in 2 or 3 are gradiometers of different orientations                                                                                                                                                              
            for sensorLabel in rsq[stat].keys(): # 'ave' or 'max'? not sure
                if sensorLabel.endswith(sensorType):
                    sensorValues = numpy.append(sensorValues,rsq[stat][sensorLabel][band]) # e.g. rsq['ave']['MEG2643']['alpha']                                
                    sensorPositions = numpy.append(sensorPositions,sensorLocations[sensorLabel]) # this is already an X-Y tuple
                sensorValues = numpy.array(sensorValues)
                sensorPositions = numpy.array(sensorPositions)
            subplotCounter += 1
            pylab.subplot(1,3,subplotCounter)
            #print(sensorValues.shape, sensorPositions.reshape(-1,2).shape)
            #mne.viz.plot_topomap(sensorValues, sensorPositions.reshape(-1,2),names=rsq[stat].keys(),show_names=True)
            im, cn = mne.viz.plot_topomap(sensorValues, sensorPositions.reshape(-1,2))
            pylab.colorbar(im,fraction=0.046, pad=0.04)
            if sensorType == '2':
                pylab.title(stat+' '+band+' band power R^2 with embedding depth from '+resultsFile)
            #pylab.legend(loc='upper right')
        pylab.savefig('graphics/meg_%s-%s_%s.png' % (resultType,stat,band))
        pylab.show()

# <codecell>

%cd ../stats/

resultType = 'wide'
resultsFile = 'signifresults.multifactor.cpk'

rsq = cPickle.load(open(resultsFile))

# EXTRACT SENSOR LOCATIONS FROM METADATA                                                                                                                       
sensorLocations = {}
# don't have the data at hand, but here it would be good to first extract the X/Y co-ordinates of all                                                          
for sensor in metadata.chanlocs:
    # not sure if the structure is like this - will have to check in metadata returned by the python script
    sensorLocations[sensor.labels] = (-sensor.Y, sensor.X)    
    
# DO A TOPOPLOT FOR EACH BAND AND STAT (three beside eachother, one for each type of sensor)                                                                   
for stat in ['r2']:#, 'max']:
    for band in ['theta','alpha','beta1','beta2']:#,'gamma',]: # add more bands if you want                                                                                           
        #sensorPositions = [] # in horizontal plane, looking downwards                                                                                  
        #sensorValues = [] # r^2 or whatever other value we want to do map of                                                                           

        pylab.figure()
        subplotCounter = 0
        #for sensorType in ['1', '2', '3']:
        for sensorType in ['1','2', '3']: 
            # sensors that end in 1 are magnetometers, those ending in 2 or 3 are gradiometers of different orientations
            sensorPositions = [] # in horizontal plane, looking downwards                                                                                  
            sensorValues = [] # r^2 or whatever other value we want to do map of
                                                                                                                                                              
            for sensorLabel in rsq[stat].keys(): # 'ave' or 'max'? not sure
                if sensorLabel.endswith(sensorType):
                    sensorValues = numpy.append(sensorValues,rsq[stat][sensorLabel][band]) # e.g. rsq['ave']['MEG2643']['alpha']                                
                    sensorPositions = numpy.append(sensorPositions,sensorLocations[sensorLabel]) # this is already an X-Y tuple
                sensorValues = numpy.array(sensorValues)
                sensorPositions = numpy.array(sensorPositions)
            subplotCounter += 1
            pylab.subplot(1,3,subplotCounter)
            #print(sensorValues.shape, sensorPositions.reshape(-1,2).shape)
            im, cn = mne.viz.plot_topomap(sensorValues, sensorPositions.reshape(-1,2))
            #im, cn = mne.viz.plot_topomap(sensorValues, sensorPositions.reshape(-1,2),names=rsq[stat].keys(),show_names=True)
            pylab.colorbar(im,fraction=0.046, pad=0.04)
            if sensorType == '2':
                pylab.title(stat+' '+band+' band power R^2 with embedding depth from '+resultsFile)
        pylab.savefig('graphics/meg_pandas_%s-%s_%s.png' % (resultType,stat,band))
        pylab.show()

# <codecell>


