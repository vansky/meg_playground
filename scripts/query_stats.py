import cPickle

DEV = False
CLUSTER = False
SHORT = True
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
    pass
  
rsq = cPickle.load(open(resultsFile))

print rsq.keys()

if CLUSTER:
  for cluster in rsq['clustersize']:
    print cluster, rsq['clustersize'][cluster]

for stat in ['p']:
  for band in ['alpha']:
    print 'Significance in', band
    if True:
      for sensor in rsq[stat]: #['0113','0123','0112','0122','1542','1532','1543','1533']:
        #print 'MEG'+sensor, 'p =',rsq[stat]['MEG'+sensor][band]
        print 'Cluster:', sensor, 'R2:', rsq['r2'][sensor][band]
        print rsq[stat][sensor][band]
    else:
      for sensor in [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1)]: #rsq[stat]: #['0113','0123','0112','0122','1542','1532','1543','1533']:
        #print 'MEG'+sensor, 'p =',rsq[stat]['MEG'+sensor][band]
        print 'Cluster:', sensor, 'R2:', rsq['r2'][sensor][band]
        print rsq[stat][sensor][band]
