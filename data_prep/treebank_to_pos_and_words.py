import argparse
from nltk.tree import Tree
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tb', help='treebank file')
    parser.add_argument('-o', help='output tsv file. One word per line, new sentences marked by an empty line')
    parsedargs = parser.parse_args()

    with open(parsedargs.tb, 'r') as f:
        print('read tree strings')
        treestrings = f.read().splitlines()
    
    print('write to ', parsedargs.o)
    with open(parsedargs.o, 'w') as f:
        for k,treestr in enumerate(treestrings):
            tree = Tree.fromstring(treestr)
            pos = tree.pos()
            for w,p in pos:
                f.write(w + '\t' + p + '\n')
            f.write('\n')
            if k % 200 == 0:
                print(k, end="\r", flush=True)


if __name__=='__main__':
    main()


