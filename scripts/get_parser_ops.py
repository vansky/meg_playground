#cat input.linetrees | python get_parser_ops.py > output
#use a list as a stack to get highest spanning category for each word
# input must be in linetrees format
# output is a list of line-delimited word/category pairs

import sys

cats = []
output  = []

for line in sys.stdin.readlines():
  line = line.strip()
  if line == '()':
    output.append('BADPARSE BADPARSE')
    continue
  mycat = ''
  myword = ''
  build_cat = False
  for char in line:
    if char == '(':
      build_cat = True
      mycat = ''
      myword = ''
    elif char == ' ':
      if build_cat:
        #add cat to stack
        build_cat = False
        cats.append(mycat)
      else:
        #add word/cat pair to output
        output.append(mycat + ' ' + myword)
    elif char == ')':
      #grab the most recent incomplete category as the current category
      mycat = cats.pop()
    else:
      if build_cat:
        #build up the category
        mycat = mycat + char
      else:
        #build up the word
        myword = myword + char
  #add final word/cat pair to output
  output.append(mycat + ' ' + myword)
sys.stdout.write('\n'.join(output)+'\n')
