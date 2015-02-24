# addtotsurp.py tableFile newcolFile > newtable
# newcolFile has two columns: totsurp word

import re
import sys

DEBUG = False
HEADERNAME = 'totsurp'

def americanize(word):
    #the parser left things with British spelling, while the MEG data is epoched to American spelling
    amerdict = {'rumours':'rumors', 'centre':'center','travelling':'traveling','valour':'valor','clamour':'clamor','honourable':'honorable','colour':'color','honour':'honor','dishonour':'dishonor','discoloured':'discolored','travelled':'traveled','favourite':'favorite','favour':'favor','neighbours':'neighbors','neighbour':'neighbor','honoured':'honored','coloured':'colored'}
    if word in amerdict:
        return(amerdict[word])
    return(word)

with open(sys.argv[1], 'r') as f:
    tableFile = f.readlines()
with open(sys.argv[2], 'r') as f:
    newcolFile = f.readlines()
newcol = []
SKIP = 0
# first, connect all hyphenated words (summing surprisal of pieces)
for i,l in enumerate(newcolFile):
    if SKIP:
        SKIP -= 1
        continue
    word = l.strip().split()
    if '-' == word[1]:
        newcol[-1][1] = newcol[-1][1] + word[1] + newcolFile[i+1].strip().split()[1] #build up hyphenated word
        newcol[-1][0] = str(float(newcol[-1][0]) + float(word[0])) #sum surprisal over all hyphenated components
        SKIP = 1
        continue
    #ensure similar segmentation to the table
    elif "Hm'm" == word[1]:
        pos = l.strip().split()[0]
        newcol.append([pos,'Hm'])
        newcol.append([pos,"'m"])
        continue
    elif "cannot" == word[1]:
        pos = l.strip().split()[0]
        newcol.append([pos,'can'])
        newcol.append([pos,'not'])
        continue
    elif "I've" == word[1]:
        pos = l.strip().split()[0]
        newcol.append([pos,'I'])
        newcol.append([pos,"'ve"])
        continue
    elif "you've" == word[1]:
        pos = l.strip().split()[0]
        newcol.append([pos,'you'])
        newcol.append([pos,"'ve"])
        continue

    newcol.append(l.strip().split())

# now add the surprisal to the table
wordix = 0
falsebadparses = [1]
badparseix = 0
inBAD = False
badchars = re.compile('[\W_]+')
for tline in tableFile:
    stline = tline.strip().split()
    
    if DEBUG and inBAD:
        sys.stderr.write(str(stline)+'\n')
    if inBAD and '<SENT-END>' == stline[3]:
        #skip badly parsed table rows until the end of that sentence
        # there are cases where the parser outputs data from the middle of the sentence (mis-segments sentence),
        # but then the surprisal would be off anyway
        inBAD = False
    if stline[0][0] == '#':
        #in a comment line
        stline.append(HEADERNAME)
        sys.stdout.write('\t'.join(stline)+'\n')
        continue
    elif len(stline) < 6:
        #in a non-parsed line
        sys.stdout.write(tline[:-1]+'\t\n')
        continue
    elif inBAD:
        #in a badparse, so we don't know the surprisal
        stline.append('-1')
        sys.stdout.write('\t'.join(stline)+'\n')
        continue
    elif stline[4] == "'":
        #take care of the occasional lone possessive apostrophe
        stline.append(newcol[wordix-1][0])
        sys.stdout.write('\t'.join(stline)+'\n')
        continue
    if newcol[wordix][1] == 'BADPARSE':
        badparseix += 1
        wordix += 1
        if badparseix not in falsebadparses:
            inBAD = True
            stline.append('-1')
            sys.stdout.write('\t'.join(stline)+'\n')
            continue
    if DEBUG:
        sys.stderr.write(str(wordix)+'/'+str(len(newcol))+': '+str(newcol[wordix])+' ?= ')
        sys.stderr.write(str(stline[4])+'\n')
    while americanize(badchars.sub('',newcol[wordix][1].lower())) != badchars.sub('',stline[4].lower()):
        wordix += 1 #go until we find the right word
        if DEBUG:
            sys.stderr.write(str(wordix)+'/'+str(len(newcol))+': '+str(newcol[wordix])+' ?= ')
            sys.stderr.write(str(stline[4])+'\n')
    stline.append(newcol[wordix][0])
    sys.stdout.write('\t'.join(stline)+'\n')
    wordix += 1
    while badchars.sub('',newcol[wordix][1].lower()) == '':
        #go until we find a new real word
        if DEBUG:
            sys.stderr.write('skipping: '+newcol[wordix][1]+'\n')
        wordix += 1
