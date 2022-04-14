from treetoolbox import find_node, load_ptb, np_regex, vp_regex, tr_leaf_regex, find_end_indices, find_xps_above_i, address_is_xp, find_label, find_tracing_alignment, find_trace_ix, preprocess, lowest_phrase_above_leaf_i
# during test: from data_prep.treetoolbox
import sys
import argparse
import re

punct_regex = re.compile(r"[^\w][^\w]?")

def findBIESTag(i, tree, i_tok, with_phrase_label=False):
    if punct_regex.match(i_tok):
        return 'PCT'
    phrase_label, phrase_node, ga_of_phrase_node = lowest_phrase_above_leaf_i(i, tree, return_target_ga=True)
    ga_of_leaf = tree.treeposition_spanning_leaves(i,i+1)
    ga_phrase_to_leaf = ga_of_leaf[len(ga_of_phrase_node):]
    is_beginning = ga_phrase_to_leaf[0] == 0 and (len(set(ga_phrase_to_leaf))==1)
    is_end = True
    node = phrase_node
    for k in ga_phrase_to_leaf:
        if len(node)-1>k:
            is_end=False
            break
        else:
            node = node[k]
        

    if is_beginning and is_end:
        tag = 'S'
    elif is_beginning:
        tag = 'B'
    elif is_end:
        tag = 'E'
    else:
        tag = 'I'
    if with_phrase_label:
        if phrase_label.startswith('NP') and len(phrase_label.split('-')) > 1:
            tag+='-'+'-'.join(phrase_label.split('-')[:2])
        else:
            tag+='-'+phrase_label.split('-')[0]
    return tag

def biesLabels(sent, tree_notr, with_phrase_labels=False):
    # outfile_only_pos.write('### ' + str(k) + '\n')
    leaves_notr = tree_notr.leaves()

    # preproc_alignment preproc_sent -> leaves_notr
    preproc_alignment, preproc_sent = preprocess(tree_notr.leaves(), only_parentheses=True)
        
    text_toks = []
    bies_labels = []

    for i_preproc in range(len(preproc_sent)):
        # find out if alignment[i] has an index r in tree_tr
        i = preproc_alignment[i_preproc]

        i_tok = preproc_sent[i_preproc]

                    
        # find phrase above token i
        label = findBIESTag(i, tree_notr, i_tok, with_phrase_label=with_phrase_labels)

        text_toks.append(i_tok)
        bies_labels.append(label)
    return preproc_sent,bies_labels

if __name__=='__main__':
    """
    Usage: 
    python3 span_prediction_format -ptb_tr <file_pattern> -ptb_notr <file_pattern> -text_toks <filename> -bies_labels <filename>

    Input Options: 
    - PTB with and without traces (both are needed)

    Output Options (each file has same number of lines): 
    - text_toks txt file with one sentence per line (input file for computing activations)
    - bies_labels txt file where the k-th label in the l-th line represents the beginning/inside/end/only label in the PTB of the corresponding tokens in text_toks.txt

    other options
    - cutoff x an integer x such that the scripts stops after processing x sentences
    -max_sent_length an integer x for the maximum sentence length (suggested: 20)
    
    The script asserts that all output files have the same number of lines, and that the output files text_toks and bies_labels have the same number of elements per line
    """

    ###############
    # PREP
    ###############

    # print(sys.argv)

    parser = argparse.ArgumentParser()
    parser.add_argument('-ptb_notr')
    parser.add_argument('-text_toks') 
    parser.add_argument('-bies_labels') 
    parser.add_argument('-cutoff')
    parser.add_argument('-max_sent_length')
    parser.add_argument('-with_phrase_labels', action='store_true')
    parsedargs = parser.parse_args()

    ignored_sents = []
    ignore_list = []

    cutoff = 2000000000 # never going to have such a big treebank
    if parsedargs.cutoff is not None:
        cutoff = int(parsedargs.cutoff)
    max_sent_length = 200000000
    if parsedargs.max_sent_length is not None:
        max_sent_length = int(parsedargs.max_sent_length)
    
    ptb_notrace = load_ptb(parsedargs.ptb_notr, corpus_root=r"data/PennTreebank")
    text_toks_file = open(parsedargs.text_toks, 'w')
    bies_labels_file = open(parsedargs.bies_labels, 'w')

    binary = False
    label_counts = dict()

    print(len(ptb_notrace.sents()))

    output_sents = set()
    ###############
    # Conversion 
    ###############
    for k, (sent, tree_notr) in enumerate(zip(ptb_notrace.sents(), ptb_notrace.parsed_sents())):
        if k - len(ignored_sents) > cutoff:
            continue
        # some sentences are not covered yet
        if k in ignore_list or len(sent) > max_sent_length: #31:
            ignored_sents.append(k)
            continue

        with_phrase_labels = parsedargs.with_phrase_labels
        preproc_sent, bies_labels = biesLabels(sent, tree_notr, with_phrase_labels=with_phrase_labels)

 
        assert len(preproc_sent) == len(bies_labels)

        # Do not store the same sentence twice!
        if str(preproc_sent) in output_sents:
            continue
        output_sents.add(str(preproc_sent))

        for l in bies_labels:
            if l not in label_counts:
                label_counts[l] = 0
            label_counts[l] = label_counts[l] + 1
        bies_labels_file.write(' '.join(bies_labels) + '\n')
        text_toks_file.write(' '.join(preproc_sent) + '\n')


        if k % 100 == 0:
            print(k, end="\r", flush=True)


    text_toks_file.close()
    bies_labels_file.close()

    print('Finished. ignored sentences: ', ignored_sents)
    print('Label distribution: ')
    label_counts = dict(sorted(label_counts.items(), key=lambda item: item[1], reverse=True))
    total = sum(label_counts.values())
    print(label_counts)
    for l,c in label_counts.items():
        print(l,'\t', c, '\t', str(float(c/total)))
