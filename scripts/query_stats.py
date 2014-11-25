import cPickle

resultsFile = 'signifresults.cpk'

rsq = cPickle.load(open(resultsFile))

for stat in ['ave']:
  for band in ['alpha','theta']:
    print 'Significance in', band
    for sensor in ['0113','0123','0112','0122','1542','1532','1543','1533']:
      print 'MEG'+sensor, 'p =',rsq[stat]['MEG'+sensor][band]
