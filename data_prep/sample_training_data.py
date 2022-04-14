import argparse
import random
import pandas as pd




def sample_data_std(art_tok_sents, label_sents, orig_rel_freqs, target_rel_freqs, art_toks_sampled, labels_sampled, sampled_to_orig_ratio, sampled_freqs):
    for toks, labels_in_sent in zip(art_tok_sents, label_sents):
        out_toks = []
        out_labels = []
        for tok, label in zip(toks.split(), labels_in_sent.split()):
            print_with_prob = sampled_to_orig_ratio * target_rel_freqs[label] / orig_rel_freqs[label]
            print_with_prob
            if print_with_prob >= 1.:
                print_tok = True
            else:
                print_tok = random.random() < print_with_prob
            if print_tok:
                out_toks.append(tok)
                out_labels.append(label)
                sampled_freqs[label] = sampled_freqs[label] + 1
        if len(out_toks) > 0:
            art_toks_sampled.write(' '.join(out_toks) + '\n')
            labels_sampled.write(' '.join(out_labels) + '\n')

    art_toks_sampled.close()
    labels_sampled.close()


def sample_data_two_outfiles(art_tok_sents, label_sents, orig_rel_freqs, target_rel_freqs, art_toks_sampled, labels_sampled, sampled_to_orig_ratio, sampled_freqs, snd_sents, snd_file_sampled):
    for toks, labels_in_sent, snd_labels in zip(art_tok_sents, label_sents, snd_sents):
        out_toks = []
        out_labels = []
        out_snd_labels = []
        for tok, label, snd_label in zip(toks.split(), labels_in_sent.split(), snd_labels.split()):
            print_with_prob = sampled_to_orig_ratio * target_rel_freqs[label] / orig_rel_freqs[label]
            print_with_prob
            if print_with_prob >= 1.:
                print_tok = True
            else:
                print_tok = random.random() < print_with_prob
            if print_tok:
                out_toks.append(tok)
                out_labels.append(label)
                out_snd_labels.append(snd_label)
                sampled_freqs[label] = sampled_freqs[label] + 1
        if len(out_toks) > 0:
            art_toks_sampled.write(' '.join(out_toks) + '\n')
            labels_sampled.write(' '.join(out_labels) + '\n')
            snd_file_sampled.write(' '.join(out_snd_labels) + '\n')

    art_toks_sampled.close()
    labels_sampled.close()
    snd_file_sampled.close()


if __name__=='__main__':
    """
    -rel_toks_i input file with artificial tokens
    -labels_i input file with labels
    -rel_toks_sampled output file with sampled artificial tokens
    -labels_sampled corresponding labels output file
    -target_size int for the target number of tokens (try e.g. 100000)
    -second_i <optional> second input file whose probabilities are not considered, but which is also sampled in the same way
    -second_o <optional> second output file with sampled values from second_i file
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-rel_toks_i')
    parser.add_argument('-labels_i')
    parser.add_argument('-rel_toks_sampled')
    parser.add_argument('-labels_sampled')
    parser.add_argument('-target_size')
    parser.add_argument('-second_i')
    parser.add_argument('-second_o')
    parsedargs = parser.parse_args()

    # read data
    print('open files ')
    art_toks_in = open(parsedargs.rel_toks_i, 'r')
    labels_file_in = open(parsedargs.labels_i, 'r')
    if parsedargs.second_i is not None:
        snd_file_in = open(parsedargs.second_i, 'r')

    print('load files ')
    art_tok_sents = [line.strip() for line in art_toks_in]
    print('loaded ', parsedargs.rel_toks_i)
    label_sents = [line.strip() for line in labels_file_in]
    print('loaded ', parsedargs.labels_i)
    if parsedargs.second_i is not None:
        snd_sents = [line.strip() for line in snd_file_in]
        snd_file_in.close()


    art_toks_in.close()
    labels_file_in.close()

    # compute label frequencies: 
    label_freqs = dict()
    for lab_sent in label_sents:
        labels = lab_sent.split()
        for label in labels:
            if label not in label_freqs:
                label_freqs[label] = 0
            label_freqs[label] = label_freqs[label] + 1

    # and relative frequencies in the original data
    dataset_target_size = int(parsedargs.target_size)
    orig_rel_freqs = dict()
    total = sum(label_freqs.values())
    for label, freq in label_freqs.items():
        rel_freq = freq / total
        orig_rel_freqs[label] = rel_freq    

    # dict of relative frequencies target in the sampled data
    # weighted between uniform distribution and original frequencies in traing data
    p = 0.5
    target_rel_freqs = {k: p*(1 / len(orig_rel_freqs))+(1.-p)*(orig_rel_freqs[k]) for k in orig_rel_freqs.keys() }

    art_toks_sampled = open(parsedargs.rel_toks_sampled, 'w')
    labels_sampled = open(parsedargs.labels_sampled, 'w')
    if parsedargs.second_i is not None:
        snd_file_sampled = open(parsedargs.second_o, 'w')

    sampled_to_orig_ratio = dataset_target_size / sum(label_freqs.values())

    sampled_freqs = {k:0 for k in label_freqs.keys()}

    print('prepared frequencies, now sample data')
    if parsedargs.second_i is None:
        sample_data_std(art_tok_sents, label_sents, orig_rel_freqs, target_rel_freqs, art_toks_sampled, labels_sampled, sampled_to_orig_ratio, sampled_freqs)
    else:
        sample_data_two_outfiles(art_tok_sents, label_sents, orig_rel_freqs, target_rel_freqs, art_toks_sampled, labels_sampled, sampled_to_orig_ratio, sampled_freqs, snd_sents, snd_file_sampled)

    sample_sum = sum(sampled_freqs.values())
    sampled_rel_freqs = {k:v/sample_sum for k,v in sampled_freqs.items()}
    results_dict = {'label_freqs_original': label_freqs, 
    'label_rel_freqs_original': orig_rel_freqs,
    'targetted_rel_freqs': target_rel_freqs,
    'sampled_freqs': sampled_freqs,
    'sampled_rel_freqs': sampled_rel_freqs}


    print('Done.\nLabel distribution in the original data: ')
    print(label_freqs)
    print('Relative frequencies in the original data: ')
    print(orig_rel_freqs)
    print('target dataset size: ', dataset_target_size)
    print('tokens in the sampled dataset: ', sum(sampled_freqs.values()))
    print('Note that the actuual number of tokens in the sampled dataset could be smaller than the target dataset size')
    print('if for some label, there are less items in the dataset than (target_relative_freq_of_label * total_tokens_in_dataset) ')    

    print('frequencies in the sampled dataset: ')
    print(sampled_freqs)
    print(pd.DataFrame(results_dict))
    print(results_dict)
