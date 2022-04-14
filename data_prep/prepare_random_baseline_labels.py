from ast import parse
from os import pardir
import numpy as np
import argparse
import re

def compute_word_freqs(sentences, compute_pairs=False):
    # compute train token and token pair frequencies
    word_freqs = dict()
    word_pair_freqs = dict()
    for line in sentences:
        sent_split = line.strip().split()
        for i,token in enumerate(sent_split):
            # word level
            if token not in word_freqs:
                word_freqs[token] = 0
            word_freqs[token] = word_freqs[token] + 1
            if compute_pairs==False:
                continue
            # word pairs
            for snd in sent_split[i:]:
                pair = ' '.join([token, snd])
                if pair not in word_pair_freqs:
                    word_pair_freqs[pair] = 0
                word_pair_freqs[pair] = word_pair_freqs[pair] + 1
    if compute_pairs:
        return word_freqs, word_pair_freqs
    else:
        return word_freqs

def compute_relative_freqs(abs_freqs):
    # compute relative label frequencies
    total = sum(abs_freqs.values())
    rel_freqs = dict()
    for label, abs_freq in abs_freqs.items():
        rel_freqs[label] = abs_freq / total
    np.testing.assert_almost_equal(sum(rel_freqs.values()),1.)
    return rel_freqs

punct_regex = re.compile(r"[^\w][^\w]?")

def writeRandomLabelDistributionToFile(rel_freqs, text, filename, words_to_labels=dict()):
    """
    assigns a random label to each token and writes the resulting token labels to file
    return the dictionary of token types and label
    """
    ps = sorted(rel_freqs.values(), reverse=True)
    num_classes = len(rel_freqs)
    outfile = open(filename, 'w')
    for line in text:
        toks = line.strip().split()
        new_labels_for_line = []
        #print('length of text line: ', len(toks))
        #print('sum(range(len(toks)))',sum(range(len(toks))))
        for i in range(len(toks)):
            tok_i = toks[i]
            if tok_i not in words_to_labels:
                words_to_labels[tok_i] = np.random.choice(range(num_classes), p=ps)
            new_labels_for_line.append(str(words_to_labels[tok_i]))
        outfile.write(' '.join(new_labels_for_line)+'\n')
    outfile.close()
    return words_to_labels

def writeRandomLabelDistributionToFileForTokenPairs(rel_freqs, text, filename, word_pairs_to_labels=dict()):
    """
    assigns a random label to each token pair and writes the resulting toekn
    pair labels to file
    return the dictionary of token type pairs and label
    """
    ps = sorted(rel_freqs.values(), reverse=True)
    num_classes = len(rel_freqs)
    outfile = open(filename, 'w')
    for line in text:
        toks = line.strip().split()
        new_labels_for_line = []
        #print('length of text line: ', len(toks))
        #print('sum(range(len(toks)))',sum(range(len(toks))))
        for i in range(len(toks)):
            tok_i = toks[i]
            if punct_regex.match(tok_i):
                continue
            for j in range(i, len(toks)):
                tok_j = toks[j]
                if punct_regex.match(tok_j):
                    continue
                pair = ' '.join([tok_i, tok_j])
                if pair not in word_pairs_to_labels:
                    word_pairs_to_labels[pair] = np.random.choice(range(num_classes), p=ps)
                new_labels_for_line.append(str(word_pairs_to_labels[pair]))
        #print('number of generated labels: ', len(new_labels_for_line))
        outfile.write(' '.join(new_labels_for_line)+'\n')
    outfile.close()
    return word_pairs_to_labels

if __name__=='__main__':
    """
    Prepare a random baseline: 
    -text_train, -text_dev two text files
    -labels_train_in A file assigning to each token pair in text_train a label, e.g. the lowest common ancestor in a syntax tree
    -labels_train_out, -labels_dev_out Write random labels to these files. The labels follow the same distribution as the original labels in -labels_train_in, but are assigned randomly for each word type pair
    -mode "pair" or "token" depending on wether the output labels should be respective to token pairs (e.g. for LCA prediction) or baseline based on individual tokens (e.g. BIO or POS tagging)

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-text_train')
    parser.add_argument('-text_dev')
    parser.add_argument('-labels_train_in')
    parser.add_argument('-labels_train_out')
    parser.add_argument('-labels_dev_out')
    parser.add_argument('-mode')

    parsedargs = vars(parser.parse_args())

    # read in original train text and compute token, token pair frequencies
    with open(parsedargs['text_train'], 'r') as ptb_train_text:
        ptb_train = ptb_train_text.readlines()
    word_freqs, word_pair_freqs = compute_word_freqs(ptb_train, compute_pairs=True)
    print('number of word types :       ', len(word_freqs))
    print('number of word tokens:      ', sum(word_freqs.values()))
    print('number of word pair types: ', len(word_pair_freqs))
    print('number of word pairs:     ', sum(word_pair_freqs.values()))
    print('unique words:                ', len([x for x in list(word_freqs.values()) if x==1]))
    print('unique word pairs:         ', len([x for x in list(word_pair_freqs.values()) if x==1]))

    # read in original labels and compute label frequencies
    with open(parsedargs['labels_train_in'], 'r') as ptb_train_labels_file:
        ptb_train_labels = ptb_train_labels_file.readlines()
    label_freqs = compute_word_freqs(ptb_train_labels)
    print('number of labels: ', len(label_freqs))
    label_rel_freqs = compute_relative_freqs(label_freqs)
    print('relative frequencies of the labels: ')
    print(label_rel_freqs)

    # write new train label distribution
    if parsedargs['mode'] == 'pair':
        word_pairs_to_random_labels = writeRandomLabelDistributionToFileForTokenPairs(label_rel_freqs, ptb_train, parsedargs['labels_train_out'])

        # read in dev text and generate new labels
        with open(parsedargs['text_dev'], 'r') as dev_text_file:
            ptb_dev = dev_text_file.readlines()
        writeRandomLabelDistributionToFileForTokenPairs(label_rel_freqs, ptb_dev, parsedargs['labels_dev_out'], word_pairs_to_labels=word_pairs_to_random_labels)
    else:
        words_to_random_labels = writeRandomLabelDistributionToFile(label_rel_freqs, ptb_train, parsedargs['labels_train_out'])
        with open(parsedargs['text_dev'], 'r') as dev_text_file:
            ptb_dev = dev_text_file.readlines()
        writeRandomLabelDistributionToFile(label_rel_freqs, ptb_dev, parsedargs['labels_dev_out'], words_to_labels = words_to_random_labels)




