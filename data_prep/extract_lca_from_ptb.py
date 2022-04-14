from treetoolbox import find_node, load_ptb, np_regex, vp_regex, tr_leaf_regex, find_end_indices, find_xps_above_i, address_is_xp, find_label, find_tracing_alignment, find_trace_ix, preprocess, lowest_phrase_above_leaf_i
# during test: from data_prep.treetoolbox
import sys
import argparse
import re
import random

punct_regex = re.compile(r"[^\w][^\w]?")

def specialConditionNPEmbed(tree, i, j, emb_depth=None):
    """special condition that i and j are leaf indices in the tree, and that they are a token pair such that
    (i) the LCA is an NP and 
    (ii) the right token has more NPs above it than the left (meaning the right token is deeper embedded than the left)
    """
    xps_above_i = len(find_xps_above_i(i, tree, np_regex))
    xps_above_j = len(find_xps_above_i(j, tree, np_regex))
    if emb_depth is None:
        return xps_above_i < xps_above_j
    else:
        # print(xps_above_j - xps_above_i)
        return xps_above_j - xps_above_i == emb_depth

def phrasePropertiesForAdjacentTokenPairsInPTB(k, sent, tree):
    preproc_alignment, preproc_sent = preprocess(tree.leaves(), only_parentheses=True)
    rel_toks = []
    lca_labels = []
    max_span_labels = []
    shared_levels = []
    unary_labels = []
    shared_only_root = []
    if tree.label() == 'VROOT' and len(tree) == 1:
        tree = tree[0]

    for i_preproc in range(len(preproc_sent)-1):
        i = preproc_alignment[i_preproc]
        i_tok = preproc_sent[i_preproc]

        j_preproc = i_preproc + 1
        j = preproc_alignment[j_preproc]
        j_tok = preproc_sent[j_preproc]
        lowest_common_ancestor = tree.treeposition_spanning_leaves(i,j+1)
        if len(lowest_common_ancestor) == 0:
            shared_only_root.append(len(shared_levels))
        # else:
        shared = len(lowest_common_ancestor)+1
        shared_levels.append(shared)

        label = find_label(tree, lowest_common_ancestor).split('-')[0]

        # check for unary leaves
        i_tok_address = tree.leaf_treeposition(i)
        unary_label = 'XX'
        above_pos = tree[i_tok_address[:-2]]
        if (len(above_pos) == 1):
            unary_label = above_pos.label().split('-')[0]
            # print('-----------')
            # tree.pretty_print()
            # print(tree.leaves()[i], i)
            # print(label, lowest_common_ancestor, i_tok_address) 

        lca_labels.append(label)
        rel_toks.append('_'.join([str(k), str(i_preproc), str(j_preproc)]))
        unary_labels.append(unary_label)
        max_span_labels.append('0')

    shared_levels_rel = []
    for l,s in enumerate(shared_levels):
        if l == 0:
            shared_levels_rel.append(str(s)) 
            last_s = s
        else:
            shared_levels_rel.append(str(s - last_s))
            last_s = s
    for r in shared_only_root:
        shared_levels_rel[r] = "ROOT"
    # print(k, tree.leaves())
    # print(shared_levels)
    # print(shared_levels_rel)
    # print()
    return preproc_sent, rel_toks, lca_labels, shared_levels_rel, unary_labels


def phrasePropertiesForTokenPairsInPTB(k, sent, tree_notr, specialCondition='', emb_depth=None):
    # outfile_only_pos.write('### ' + str(k) + '\n')
    leaves_notr = tree_notr.leaves()
    sent_str = ' '.join(sent)
    # set_str_cleaned = preprocess 
    xps_in_sent = []

    # outfile_only_pos.write(f"### {sent_str}\n")
    # preproc_alignment preproc_sent -> leaves_notr
    preproc_alignment, preproc_sent = preprocess(tree_notr.leaves(), only_parentheses=True)
    # outfile_only_pos.write(f"### {' '.join(preproc_sent)}\n")
        
    rel_toks = []
    lca_labels = []
    max_span_labels = []

    for i_preproc in range(len(preproc_sent)):
        # find out if alignment[i] has an index r in tree_tr
        i = preproc_alignment[i_preproc]

        i_tok = preproc_sent[i_preproc]
        if punct_regex.match(i_tok):
            continue
        if not(specialCondition=='NP_embed'):
            # find phrase above token i

            label, node = lowest_phrase_above_leaf_i(i, tree_notr)
            label=label.split('-')[0]
            # outfile_only_pos.write('\t'.join([str(i_preproc), str(i_preproc), label]) + '\n') #, preproc_sent[i_preproc], preproc_sent[i_preproc]])+'\n')

            max_span_label = '1' if len(node.leaves()) == 1 else '0'
            max_span_labels.append(max_span_label)

            rel_toks.append('_'.join([str(k), str(i_preproc), str(i_preproc)]))
            lca_labels.append(label)
            # CONTINUE HERE: 
            # Write method to find out if there is an index above leaves_tr[i_tr]
            # if so, find the origin and find the lowest possible ancestors?
            # coindex_treepos = find_trace_ix(tree_tr, i_tr)
        for j_preproc in range(i+1,len(preproc_sent)):
                # find out if alignment[j] has an index s in tree_tr
            j = preproc_alignment[j_preproc]

            j_tok = preproc_sent[j_preproc]
            if punct_regex.match(j_tok):
                continue
                # default case: no traces
            lowest_common_ancestor = tree_notr.treeposition_spanning_leaves(i,j+1)
            label = find_label(tree_notr, lowest_common_ancestor).split('-')[0]
                # outfile_only_pos.write('\t'.join([str(i_preproc), str(j_preproc), label]) + '\n') # , preproc_sent[i_preproc], preproc_sent[j_preproc]])+'\n')

            lca_node = find_node(tree_notr, lowest_common_ancestor)
            max_span_label = '0'
            if len(lca_node.leaves()) == 1 + j - i:
                max_span_label = '1'
            else:
                i_tok_matches = lca_node.leaves()[0] == i_tok or (punct_regex.match(lca_node.leaves()[0]) and lca_node.leaves()[1] == i_tok)
                j_tok_matches = lca_node.leaves()[-1] == j_tok or (punct_regex.match(lca_node.leaves()[-1]) and lca_node.leaves()[-2] == j_tok)
                if i_tok_matches and j_tok_matches:
                    max_span_label = '1'
            
            
            if not(specialCondition=='NP_embed'):
                lca_labels.append(label)
                rel_toks.append('_'.join([str(k), str(i_preproc), str(j_preproc)]))
                max_span_labels.append(max_span_label)
            elif label=='NP' and specialConditionNPEmbed(tree_notr, i, j, emb_depth=emb_depth):
                lca_labels.append(label)
                rel_toks.append('_'.join([str(k), str(i_preproc), str(j_preproc)]))
                max_span_labels.append(max_span_label)

    return preproc_sent,rel_toks,lca_labels,max_span_labels

def findAndSampleMostDeeplyEmbeddedNPsInPTB(k, sent, tree_notr):
    # print(k)
    preproc_alignment, preproc_sent = preprocess(tree_notr.leaves(), only_parentheses=True)
    preproc_alignment_inv = {v:k for k,v in preproc_alignment.items()} # inv: from sent to preproc_sent
    leaf_to_nps_above = {i:find_xps_above_i(i, tree_notr, np_regex) for i in range(len(tree_notr.leaves()))}
    
    max_depth = max([len(l) for l in leaf_to_nps_above.values()])
    deepest_embedded_leaves = [i for i,v in leaf_to_nps_above.items() if len(v) == max_depth]
    lca = ''
    try_again = 0
    while lca != 'NP' and try_again<10:
        deep_token_i = random.choice(deepest_embedded_leaves)
        deep_token = preproc_sent[preproc_alignment_inv[deep_token_i]]
        while punct_regex.match(deep_token):
            deep_token_i = random.choice(deepest_embedded_leaves)
            deep_token = preproc_sent[preproc_alignment_inv[deep_token_i]]

        highest_np_above_deep_token = leaf_to_nps_above[deep_token_i][0]
        possible_high_token_is = []

        for j in range(1,10):
            if len(possible_high_token_is)==0:
                possible_high_token_is = [i for i,nps in leaf_to_nps_above.items() if len(nps)==j and highest_np_above_deep_token in nps and i < deep_token_i]
                possible_high_token_is_with_fitting_feats = []
                for i in possible_high_token_is:
                    high_token = preproc_sent[preproc_alignment_inv[i]]
                    if not(punct_regex.match(high_token)):
                        lowest_common_ancestor = tree_notr.treeposition_spanning_leaves(i,deep_token_i+1)
                        lca_label = find_label(tree_notr, lowest_common_ancestor).split('-')[0]
                        if lca_label=='NP':                    
                            possible_high_token_is_with_fitting_feats.append(i)
                possible_high_token_is = possible_high_token_is_with_fitting_feats

        if len(possible_high_token_is) > 0:
            high_token_i = random.choice(possible_high_token_is)
            high_token = preproc_sent[preproc_alignment_inv[high_token_i]]

            lowest_common_ancestor = tree_notr.treeposition_spanning_leaves(high_token_i,deep_token_i+1)
            lca = find_label(tree_notr, lowest_common_ancestor).split('-')[0]
            #  print(high_token_i, high_token, deep_token_i, deep_token)
            # print(lca)
            # fill return values
            rel_toks= ['_'.join([str(k), str(preproc_alignment_inv[high_token_i]), str(preproc_alignment_inv[deep_token_i])])]
            lca_labels = [lca]
            return preproc_sent, rel_toks, lca_labels
        else:
            try_again = try_again + 1
    return None, None, None

if __name__=='__main__':
    """
    Usage: 
    python3 span_prediction_format -ptb_tr <file_pattern> -ptb_notr <file_pattern> -text_toks <filename> -rel_toks <filename> -rel_labels <filename>

    Input Options: 
    - PTB with and without traces (both are needed)

    Output Options (each file has same number of lines): 
    -text_toks txt file with one sentence per line (input file for computing activations)
    -rel_toks txt file with tokens i_j for each i, j within the sentence length
    -rel_labels txt file where the k-th label in the l-th line represents the lowest common ancestor label in the PTB of the corresponding tokens in rel_toks.txt
    -max_span_const optional file. If present, binary labels are printed to the file that indicate whether or not the token pair is the first and last element of a constituent or not

    other options
    -cutoff x an integer x such that the scripts stops after processing x sentences
    -max_sent_length an integer x for the maximum sentence length (suggested: 20)
    -np_embed special condition for sampling only output labels such that 
    The script asserts that all output files have the same number of lines, and that the output files rel_toks and rel_labels have the same number of elements per line
    -next create only output for adjacent tokens, instead of all possible token pairs. 
    -shared_levels is used together with next. It indicates for a pair of adjacent tokens the variation in the number of shared tree levels between adjacent tokens (i.e. how much deeper in the tree is the LCA of the current token pair, compared to the last token pair?)
    -unary is used together with next. It indicates the output file with labels of unary leaf chains above a token.
    """

    ###############
    # PREP
    ###############

    # print(sys.argv)

    parser = argparse.ArgumentParser()
    parser.add_argument('-ptb_notr')
    parser.add_argument('-text_toks') 
    parser.add_argument('-rel_toks')
    parser.add_argument('-rel_labels') 
    parser.add_argument('-max_span_const')
    parser.add_argument('-cutoff')
    parser.add_argument('-max_sent_length')
    parser.add_argument('-np_embed', action='store_true', default=False)
    parser.add_argument('-next', action='store_true', default=False)
    parser.add_argument('-shared_levels', default=None)
    parser.add_argument('-unary', default=None)
    parser.add_argument('-tree_out', default=None, help="optional file where all trees are printed that are processed (and not skipped)")
    parsedargs = parser.parse_args()

    ignored_sents = []
    ignore_list = []

    cutoff = 2000000000 # never going to have such a big treebank
    if parsedargs.cutoff is not None:
        cutoff = int(parsedargs.cutoff)
    max_sent_length = 200000000
    if parsedargs.max_sent_length is not None:
        max_sent_length = int(parsedargs.max_sent_length)
    
    if not('/' in parsedargs.ptb_notr):
        ptb_notrace = load_ptb(parsedargs.ptb_notr, corpus_root=r"data/PennTreebank")
    else:
        corpus_root = '/'.join(parsedargs.ptb_notr.split('/')[:-1])
        filename=parsedargs.ptb_notr.split('/')[-1:]
        ptb_notrace = load_ptb(filename, corpus_root=corpus_root)

    text_toks_file = open(parsedargs.text_toks, 'w')
    rel_toks_file = open(parsedargs.rel_toks, 'w')
    rel_labels_file = open(parsedargs.rel_labels, 'w')
    if parsedargs.max_span_const is not None:
        max_span_file = open(parsedargs.max_span_const, 'w')
    if parsedargs.shared_levels is not None:
        shared_levels_file = open(parsedargs.shared_levels, 'w')
    if parsedargs.unary is not None:
        unary_file = open(parsedargs.unary, 'w')
    if parsedargs.tree_out is not None:
        tree_out_file = open(parsedargs.tree_out, 'w') 
    binary = False
    print(len(ptb_notrace.sents()))
    specialCond = 'NP_embed' if parsedargs.np_embed else False 
    if specialCond:
        emb_depth=int(parsedargs.np_embed)
    else:
        emb_depth=None

    
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

        if parsedargs.np_embed:
            preproc_sent, rel_toks, lca_labels = findAndSampleMostDeeplyEmbeddedNPsInPTB(k-len(ignored_sents), sent, tree_notr)
            if preproc_sent is None and rel_toks is None and lca_labels is None:
                ignored_sents.append(k)
                continue
            assert len(rel_toks) == len(lca_labels)
            rel_toks_file.write(' '.join(rel_toks) + '\n')
            rel_labels_file.write(' '.join(lca_labels) + '\n')
            text_toks_file.write(' '.join(preproc_sent) + '\n')      
        else:
            next = parsedargs.next
            if next:
                if len(tree_notr.leaves())<3:
                    ignored_sents.append(k)
                    continue
                preproc_sent, rel_toks, lca_labels, shared_levels_rel, unary_labels = phrasePropertiesForAdjacentTokenPairsInPTB(k-len(ignored_sents), sent, tree_notr)
            else:
                preproc_sent, rel_toks, lca_labels, max_span_labels = phrasePropertiesForTokenPairsInPTB(k-len(ignored_sents), sent, tree_notr, specialCondition=specialCond, emb_depth=emb_depth)
       
                assert len(rel_toks) == len(lca_labels) == len(max_span_labels)
            rel_toks_file.write(' '.join(rel_toks) + '\n')
            rel_labels_file.write(' '.join(lca_labels) + '\n')
            text_toks_file.write(' '.join(preproc_sent) + '\n')
            if parsedargs.shared_levels is not None:
                shared_levels_file.write(' '.join(shared_levels_rel) + '\n')
            if parsedargs.unary is not None:
                unary_file.write(' '.join(unary_labels) + '\n')
            if parsedargs.max_span_const is not None:
                max_span_file.write(' '.join(max_span_labels) + '\n')
            if parsedargs.tree_out is not None:
                if len(tree_notr) == 1 and tree_notr.label() == 'VROOT':
                    tree_notr = tree_notr[0]
                tree_out_file.write(str(tree_notr).replace('\n','').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')+'\n')

        # outfile_only_pos.write('###\n')
        if k % 100 == 0:# and outfile_only_pos.name != '<stdout>':
            print(k, end="\r", flush=True)

    text_toks_file.close()
    rel_toks_file.close()
    rel_labels_file.close()
    if parsedargs.max_span_const is not None:
        max_span_file.close()
    if parsedargs.shared_levels is not None:
        shared_levels_file.close()
    if parsedargs.unary is not None:
        unary_file.close()
    if parsedargs.tree_out is not None:
        tree_out_file.close()
    print('finished. ignored sentences: ', ignored_sents)

