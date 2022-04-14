import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help="tsv file produced by the experiment script")
    parser.add_argument('-o', help="txt file with line-by-line as output")
    parsedargs = parser.parse_args()

    with open(parsedargs.i,'r') as f:
        lines = f.read().splitlines()

    outlines = [] 
    sent = [] 
    prev_line_sent_ix = lines[1].split('_')[0]

    for line in lines[1:]: 
        this_line_sent_ix = line.split('_')[0] 
        if this_line_sent_ix != prev_line_sent_ix: 
            outlines.append(sent) 
            sent = []  
        sent.append(line.split('\t')[-1]) 
        prev_line_sent_ix = this_line_sent_ix
    outlines.append(sent)

    with open(parsedargs.o, 'w') as f:
        for l in outlines:
            f.write(' '.join(l) +'\n')


if __name__=='__main__':
    main()

