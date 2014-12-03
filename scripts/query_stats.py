import cPickle

resultsFile = 'signifresults.multifactor.dev.cluster.cpk'

rsq = cPickle.load(open(resultsFile))

print rsq.keys()

for cluster in rsq['clustersize']:
  print cluster, rsq['clustersize'][cluster]

for stat in ['p']:
  for band in ['alpha','theta']:
    print 'Significance in', band
    for sensor in rsq[stat]: #['0113','0123','0112','0122','1542','1532','1543','1533']:
      #print 'MEG'+sensor, 'p =',rsq[stat]['MEG'+sensor][band]
      print sensor, 'p =',rsq[stat][sensor][band]
