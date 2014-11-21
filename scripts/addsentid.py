# buildtab.py tableFile
# adds a sentid to the feature table file

import re
import sys


with open(sys.argv[1], 'r') as f:
    tableFile = f.readlines()

# now add the sentids to the table
sentid = 0
for tline in tableFile:
    stline = tline.strip().split()
    
    if '<SENT-END>' == stline[3]:
        # we're at the end of a sentence
        sentid += 1
    if stline[0][0] == '#':
        #in a comment line
        stline.append('sentid')
        sys.stdout.write('\t'.join(stline)+'\n')
        continue
    elif len(stline) < 6:
        #in a non-parsed line
        sys.stdout.write(tline[:-1]+'\t\n')
        continue
    stline.append(str(sentid))
    sys.stdout.write('\t'.join(stline)+'\n')
