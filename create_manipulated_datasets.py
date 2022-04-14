from conllu import parse, parse_tree
import pandas as pd
from nltk.tree import Tree
import argparse
from data_prep import treetoolbox
import os.path
import random
import spacy

def find_patterns(dep_tree, patterns=[]):
    form = dep_tree.token['form']
    lemma = dep_tree.token['lemma']
    deprel = dep_tree.token['deprel']
    upos = dep_tree.token['upos']
    deprels_to_children = []
    for c in dep_tree.children:
        deprels_to_children.append(c.token['deprel'])
    patterns.append((form, lemma, deprel, upos, str(deprels_to_children)))
    for c in dep_tree.children:
        patterns = find_patterns(c, patterns=patterns)
    return patterns


def find_subtree_with_token_ix(dep_tree, i):
    """given a dep tree and an index, find the subtree with that index"""
    if dep_tree.token['id'] == i:
        return dep_tree
    # TODO continue here: STH like this
    for c in dep_tree.children:
        in_children = find_subtree_with_token_ix(c,i) 
        if in_children is not None:
            return in_children
    return None

def find_deprels_to_children(dep_tree, i):
    """dep_tree: a TokenTree dependency tree
    i: the index of the token of which we want the deprels to the children
    """
    relevant_subtree = find_subtree_with_token_ix(dep_tree, i)
    children_list = []
    for c in relevant_subtree.children:
        children_list.append(c.token['deprel'])
    return children_list

def find_deprels_to_children_spacy(dep_sent, ix):
    tok = dep_sent[ix]
    deps_children = [c.dep_ for c in tok.children]
    return deps_children

if __name__=='__main__':

    
    parser = argparse.ArgumentParser()
    parser.add_argument('-const_in')
    parser.add_argument('-dep_in')
    parser.add_argument('-const_out')
    parser.add_argument('-text_out')
    parser.add_argument('-text_out_orig')
    parser.add_argument('-ratio')
    parser.add_argument('-stop_out_after')
    parsedargs = parser.parse_args()

    print('read constituency PTB')
    path_to_ptb_split = parsedargs.const_in.split('/')
    corp_root = r"{}".format('/'.join(path_to_ptb_split[:-1]))
    ptb = treetoolbox.load_ptb(path_to_ptb_split[-1:], corpus_root=corp_root)
    const_trees = ptb.parsed_sents()

    if parsedargs.dep_in is not None and os.path.isfile(parsedargs.dep_in):
        print('read dep resources')
        with open(parsedargs.dep_in, 'r') as f:
            raw_data = f.readlines()
            raw_data = ''.join(raw_data)
        dep_trees = parse_tree(raw_data) # Trees
        dep_sents = parse(raw_data)      # Sentences

        # make a test to be sure that the data is properly aligned
        k = 0
        hit = 0
        print('make alignment test')
        for const_tree, dep_tree, dep_sent in zip(const_trees, dep_trees, dep_sents):
            k = k + 1
            dep_str = ''
            for dep_tok in dep_sent:
                dep_str += str(dep_tok) + ' '
            const_str = ' '.join(const_tree.leaves())
            dep_str = dep_str.strip()
            const_str = const_str.replace('\/','/').replace('LRB','(').replace('RRB',')').replace('\*','*')
            dep_str = dep_str.replace('-LCB-','LCB').replace('-RCB-','RCB')
            if not(const_str == dep_str):
                print(const_str)
                print(dep_str)
                print(k)
                hit = hit + 1
                break
        print(hit, ' problems')
        if hit==0:
            print('this means that your constituency and dependency files are properly aligned.')
        else:
            print('this means that your constituency and dependency files are not properly aligned.')
            exit(1)

        print('find patterns')
        # a pattern is a tuple (form, lemma, upos, deprel_to_head, list(deprels_to_children))
        patterns_nested_list = []
        k = 0
        for tree in dep_trees: 
            new_patterns = find_patterns(tree,patterns=[])
            patterns_nested_list.append(new_patterns)
            # print(len(new_patterns))
            k = k + 1
        patterns = [item for sublist in patterns_nested_list for item in sublist]
        patterns_df = pd.DataFrame(patterns, columns=['form', 'lemma', 'deprel', 'upos', 'deprels_to_children'])

        # remove some unwanted patterns
        forms_to_rm = ['ai', 'de', 'del', 'en', 'la', 'le', 'na', 's', 'v.', 'Versicherung']
        forms_to_rm += [e for e in set(patterns_df.form) if e.isupper()]
        forms_to_rm = set(forms_to_rm)
        rm_idxs = []
        for form in forms_to_rm:
            rm_idxs += patterns_df[patterns_df['form'] == form].index.tolist()  
        
        patterns_df = patterns_df.drop(rm_idxs)

        # prepare a pattern dictionary from the dataframe -> faster
        patterns_dict = dict() # upos -> deprel -> deprels_to_children -> list(forms)
        for ix,(form,lemma,deprel,upos,deprels_to_children) in patterns_df.iterrows():
            if upos not in patterns_dict:
                patterns_dict[upos] = dict()
            if deprel not in patterns_dict[upos]:
                patterns_dict[upos][deprel] = dict()
            if deprels_to_children not in patterns_dict[upos][deprel]:
                patterns_dict[upos][deprel][deprels_to_children] = []
            patterns_dict[upos][deprel][deprels_to_children].append(form)
    else:
        print('load spacy model')
        nlp = spacy.load('en_core_web_lg')
        print('create dependency parses for sentences')
        sent_strs = [' '.join(tree.leaves()) for tree in const_trees]
        dep_trees = list(nlp.pipe(sent_strs,disable='ner'))
        print('create pattern')
        patterns_dict = dict()
        dep_sents = []
        for sent_doc in dep_trees:
            dep_sents.append([{'upos':tok.pos_,'deprel':tok.dep_,'form':tok.text,'id':i} for i,tok in enumerate(sent_doc)])
            for tok in sent_doc: 
                pos = tok.pos_
                dep = tok.dep_
                children = str([c.dep_ for c in tok.children])
                if pos not in patterns_dict:
                    patterns_dict[pos] = dict()
                if dep not in patterns_dict[pos]:
                    patterns_dict[pos][dep] = dict()
                if children not in patterns_dict[pos][dep]:
                    patterns_dict[pos][dep][children] = []
                text = tok.text
                if text not in patterns_dict[pos][dep][children]:
                    patterns_dict[pos][dep][children].append(text)

    print('create new dataset')
    new_const_trees = []
    k = 0
    replace_ratio = float(parsedargs.ratio) # percentage_of_replaced_tokens
    print('total trees to convert: ', max(len(const_trees), replace_ratio))
    # loop over sentences
    if parsedargs.stop_out_after is not None:
        stop_after = int(parsedargs.stop_out_after)
    else:
        stop_after = 2**24
    for dep_sent, dep_tree, const_tree in zip(dep_sents, dep_trees, const_trees): 
        if k >= stop_after:
            break
        if k % 2000 == 0:
            print(k,end=", ", flush=True)
        k = k + 1
        new_const_tree = const_tree.copy()
        sent_len = len(dep_sent)
        replaced_tokens = []    
        num_punct_tokens = len([tok for tok in dep_sent if tok['deprel']=='punct'])
        goal_to_replace = (len(dep_sent) - num_punct_tokens) * replace_ratio

        since_last_replacement=0
        while since_last_replacement<10 and len(replaced_tokens) < goal_to_replace:
            since_last_replacement=since_last_replacement+1
            try_token = random.choice(range(sent_len))
            token = dep_sent[try_token]
            if try_token in replaced_tokens or token['form'] in forms_to_rm:
                continue
            if token['deprel']=='punct' or token['form'].startswith("'"):
                continue
            if not(isinstance(dep_tree,spacy.tokens.doc.Doc)):
                deprels_to_children = str(find_deprels_to_children(dep_tree, token['id']))
            else:
                deprels_to_children = str(find_deprels_to_children_spacy(dep_tree, token['id']))
            alternatives = patterns_dict[token['upos']][token['deprel']][deprels_to_children]
            if len(alternatives) == 1:
                continue
            new_word = random.choice(alternatives)
            while new_word.startswith("'"):
                new_word = random.choice(alternatives)        
            # change the constituent tree (This seems to be the least complicated way, works reasonably fast :/)
            new_const_tree_str = str(new_const_tree).replace(' '+token['form'] + ')', ' '+new_word+')')
            new_const_tree = Tree.fromstring(new_const_tree_str)
            since_last_replacement=0
            replaced_tokens.append(try_token)
        new_const_trees.append(new_const_tree)

    print('\nwrite output')
    outfile_trees = open(parsedargs.const_out,'w')
    outfile_text = open(parsedargs.text_out, 'w')
    origtextout = parsedargs.text_out_orig is not None
    if origtextout:
        outfile_text_orig = open(parsedargs.text_out_orig, 'w')
    for new_tree, orig_tree in zip(new_const_trees, const_trees):
        print_str = str(new_tree).replace('\n','').replace('  ',' ').replace('  ',' ').replace('  ', '')
        outfile_trees.write(print_str + '\n')
        outfile_text.write(' '.join(new_tree.leaves()) + '\n')
        if origtextout:
            outfile_text_orig.write(' '.join(orig_tree.leaves()) + '\n')
    outfile_trees.close()
    outfile_text.close()
    if origtextout:
        outfile_text_orig.close()
