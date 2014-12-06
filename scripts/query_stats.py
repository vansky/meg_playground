#provides rough cli to quickly query stats output from meg_stats_runner.py

import cPickle

DEV = True #output stats from dev data
CLUSTER = True #output stats from cluster analysis
SHORT = False #output stats from Pz-only analysis
if CLUSTER:
  if DEV:
    resultsFile = 'signifresults.multifactor.dev.cluster.cpk'
  else:
    resultsFile = 'signifresults.multifactor.test.cluster.cpk'
else:
  if SHORT:
    if DEV:
      resultsFile = 'sigresults.dev.short.cpk'
    else:
      resultsFile = 'sigresults.test.short.cpk'
  else:
    raise # we don't currently support quick analysis of a full sensor-based annalysis over all channels (too much data); use notebooks/Topomap_real for visualization capabilities
  
rsq = cPickle.load(open(resultsFile))

print rsq.keys() #print queryable properties

if CLUSTER:
  #output cluster coordinates
  for cluster in rsq['clustersize']:
    print cluster, rsq['clustersize'][cluster]

for stat in ['p']:
  for band in ['alpha']:
    print 'Significance in', band
    if not CLUSTER:
      for sensor in rsq[stat]: #['0113','0123','0112','0122','1542','1532','1543','1533']:
        #print 'MEG'+sensor, 'p =',rsq[stat]['MEG'+sensor][band]
        print 'Sensor:', sensor, 'R2:', rsq['r2'][sensor][band]
        print rsq[stat][sensor][band]
    else:
      #mainly we care about the bottom 6 clusters in a (3,3) cluster setup; the front of the helmet has more air since subj heads rest towards the back
      for sensor in [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1)]:
      #for sensor in [(1,0)]: check back central cluster in a (x,3) setup
        #print 'MEG'+sensor, 'p =',rsq[stat]['MEG'+sensor][band]
        print 'Cluster:', sensor, 'R2:', rsq['r2'][sensor][band]
        print rsq[stat][sensor][band]
