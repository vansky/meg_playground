# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import cPickle
import pandas

# <codecell>

%cd ../stats
#resultsFile = 'signifresults.multifactor.cpk'
resultsFile = 'signifresults.cpk'

rsq = cPickle.load(open(resultsFile))

# <codecell>

#rsq['p', 'r2'][chanlabel]['theta', 'beta2', 'beta1', 'alpha']
print rsq.keys()

print type(rsq['r2']['MEG1211']['theta'])

print rsq['p']['MEG1211']['theta']

# <codecell>

#for band in ['alpha','theta']:
#  print 'Significance in', band
  #for sensor in ['0113','0123','0112','0122','1542','1532','1543','1533']:
#    print 'MEG'+sensor, 'p =',rsq['p']['MEG'+sensor][band]['syndepth']

# <codecell>

%cd ../stats
resultsFile = 'signifresults.multifactor.dev.bigscale.cpk'

rsq = cPickle.load(open(resultsFile))

for band in ['alpha']:
  print 'Significance in', band
  #for sensor in ['0113','0123','0112','0122','1542','1532','1543','1533']:
  for sensor in ['0313','0113','0143','1543','1533','1713']:
    print 'MEG'+sensor,'\t', rsq['r2']['MEG'+sensor]['alpha']
    for i in rsq['p']['MEG'+sensor]['alpha'].index:
        if i != 'Intercept':
            if i != 'bigramLogProbBack_COCA':
                print i, '\t\t',rsq['p']['MEG'+sensor]['alpha'][i]
            else:
                print i, '\t',rsq['p']['MEG'+sensor]['alpha'][i]
        #print rsq['p']['MEG'+sensor][band]#['syndepth']

# <codecell>


