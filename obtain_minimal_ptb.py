import nltk
from nltk.corpus import treebank


print('Download PTB data')
nltk.download('treebank')

print('Read PTB files')
files = [i for i in treebank.fileids()]
trees = []
for f in files:
  for tree in treebank.parsed_sents(f):
    trees.append(tree)
print('Read trees: ', len(trees))

print('write train files (all but the last 500 sentences)')
trainFile = open('data/PennTreebank/ptb-3414', 'w')
for tree in trees[:3414]:
    trainFile.write(str(tree))
    trainFile.write('\n')
trainFile.close()

evalFileName = 'data/PennTreebank/ptb-500'
print('write eval files (the last 500 sentences) to ', evalFileName)
evalFile = open('data/PennTreebank/ptb-500', 'w')
# write test files (the rest)
for tree in trees[3414:]:
    evalFile.write(str(tree))
    evalFile.write('\n')
evalFile.close()

