import json, h5py
import numpy as np
import argparse

def average_activation_dicts(i_act, j_act):
    new_tok = str(i_act['token']) + '_' + str(j_act['token'])
    new_layers = []
    for i_layer, j_layer in zip(i_act['layers'], j_act['layers']):
        assert i_layer['index'] == j_layer['index']
        new_layer = {'index': i_layer['index']}
        assert len(i_layer['values']) == len(j_layer['values'])
        new_vals = []
        for i_val, j_val in zip(i_layer['values'], j_layer['values']):
            new_vals.append((i_val + j_val) / 2.)
        new_layer['values'] = new_vals
        new_layers.append(new_layer)
    result = {'token': new_tok, 'layers': new_layers}
    return result

def extract_from_json(infilename, outfilename):
    f = open('resources/Data_david/dev_sentences.json',)
    json_lines = f.readlines()
    f.close()
    out_f = open('resources/Data_david/avg.json', 'w')

    sent_activations = []
    for line in json_lines:
        sent_activations.append(json.loads(line))

    print('done loading, now compute avgs...')
    # averaged_activations = []
    for k, sent_act in enumerate(sent_activations):
        avg_act_sent = {'linex_index': sent_act['linex_index']}
        avg_feats = []
        max_j = len(sent_act['features'])
        for i,tok_dict in enumerate(sent_act['features']):
            for j in range(i,max_j):
                i_act = sent_act['features'][i]
                j_act = sent_act['features'][i]
                avg_act = average_activation_dicts(i_act, j_act)
                avg_feats.append(avg_act)
    
        avg_act_sent['features'] = avg_feats
        # averaged_activations.append(avg_act_sent)

        avg_json_str = json.dumps(avg_act_sent)
        out_f.write(avg_json_str)
        out_f.write('\n')
        # print(k)
        out_f.close()

def signedabsmax(a): 
    """a is a pair of (possibly negative floats
    
    Return the element from a that has the higher absolute value
    """
    abs_a = np.abs(a) 
    if abs_a[0] > abs_a[1]: 
        return a[0] 
    else: 
        return a[1]

def combine_activations_np(sentarray, mode='avg', round_to=-1):
    res_sent_length = sum(range(sentarray.shape[1]+1))
    if mode in ['avg', 'max']:
        combined_arr = np.empty((sentarray.shape[0], res_sent_length, sentarray.shape[2]))
    if mode=='concat':
        combined_arr = np.empty((sentarray.shape[0], res_sent_length, sentarray.shape[2]*2))


    # print(avg_arr.shape)
    # print(sentarray.shape)
    k = 0
    for i in range(sentarray.shape[1]):
        for j in range(i, sentarray.shape[1]):
            i_repr = sentarray[:,i]
            j_repr = sentarray[:,j]
            if mode=='avg':
                avg_i_j = np.mean([i_repr, j_repr], axis=0)
                if round_to > -1:
                    avg_i_j = np.round(avg_i_j, decimals=round_to)
            
                combined_arr[:,k] = avg_i_j
            elif mode=='concat': 
                concat_i_j = np.concatenate([i_repr, j_repr], axis=1).astype('float16')
                if round_to > -1:
                    concat_i_j = np.round(concat_i_j, decimals=round_to)
                combined_arr[:, k] = concat_i_j
            else:
                stacked = np.stack([i_repr,j_repr])
                combined_arr[:,k] = np.apply_along_axis(signedabsmax, 0, stacked)
            k = k+1 
    return combined_arr 

def combine_activations_sampled(sentarray, sent, mode='avg', round_to=-1):
    sent_split = sent.split()
    res_sent_length = len(sent_split)
    if mode in ['avg', 'max','left', 'right']:
        combined_arr = np.empty((sentarray.shape[0], res_sent_length, sentarray.shape[2]), dtype='float16')
    if mode=='concat':
        combined_arr = np.empty((sentarray.shape[0], res_sent_length, sentarray.shape[2]*2), dtype='float16')

    k = 0
    for tok in sent_split:
        [_, i, j] = tok.split('_')
        i_repr = sentarray[:,int(i)]
        j_repr = sentarray[:,int(j)]
        if mode=='avg':
            avg_i_j = np.mean([i_repr, j_repr], axis=0)
            if round_to > -1:
                avg_i_j = np.round(avg_i_j, decimals=round_to)
            
            combined_arr[:,k] = avg_i_j
        elif mode=='concat': 
            concat_i_j = np.concatenate([i_repr, j_repr], axis=1)
            if round_to > -1:
                concat_i_j = np.round(concat_i_j, decimals=round_to)
            combined_arr[:, k] = concat_i_j
        elif mode=='left':
            combined_arr[:,k] = i_repr
        elif mode=='right':
            combined_arr[:,k] = j_repr        
        else: # signed abs max
            stacked = np.stack([i_repr,j_repr])
            combined_arr[:,k] = np.apply_along_axis(signedabsmax, 0, stacked)
        k = k + 1
    return combined_arr


def extract_from_hdf5(infilename, outfilename, reltoksfilename, mode='avg', round_to=-1):

    f_in = h5py.File(infilename, 'r')
    f_out = h5py.File(outfilename, 'w')
    rel_toks_f = open(reltoksfilename, 'r')
    rel_toks_sents = rel_toks_f.readlines()
    rel_toks_f.close()
    sents_to_ix = dict()

    for k, (ix, dataset) in enumerate(f_in.items()):
        if ix == 'sentence_to_index':
            continue
        else:
            # sentence, create new numpy ndarray with averaged reprs
            avg_data = combine_activations_np(dataset, mode=mode, round_to=round_to)
            f_out.create_dataset(ix, data=avg_data)
            sents_to_ix[rel_toks_sents[k].strip()] = str(k)
            print(k, end="\r", flush=True)
            k = k + 1


    sentence_index_dataset = f_out.create_dataset("sentence_to_index", (1,), dtype=h5py.special_dtype(vlen=str))
    sentence_index_dataset[0] = json.dumps(sents_to_ix)
    f_in.close()
    f_out.close()

def extract_from_hdf5_sampled(infilename, outfilename, reltoksfilename, mode='avg', round_to=-1):
    f_in = h5py.File(infilename, 'r')
    f_out = h5py.File(outfilename, 'w')
    rel_toks_f = open(reltoksfilename, 'r')
    rel_toks_sents = [line.strip() for line in rel_toks_f]
    rel_toks_f.close()
    sents_to_ix = dict()

    for sent in rel_toks_sents:
        k = sent.split('_')[0]
        # sentence, create new numpy ndarray with averaged reprs
        # try:
        out_data = combine_activations_sampled(f_in[str(k)], sent, mode=mode, round_to=round_to)
        # except:
        #    print('sth went wrong, try again')
        #    out_data = combine_activations_sampled(f_in[k], sent, mode=mode, round_to=round_to)
        f_out.create_dataset(str(k), data=out_data)
        sents_to_ix[sent.strip()] = str(k)
        print(k, end="\r", flush=True)
        
    sentence_index_dataset = f_out.create_dataset("sentence_to_index", (1,), dtype=h5py.special_dtype(vlen=str))
    sentence_index_dataset[0] = json.dumps(sents_to_ix)
    f_in.close()
    f_out.close()    




if __name__=='__main__':
    """
    Take a file with LM activations and create a new file with averaged representation:
    For a sentence with length n, a new artificial sentence is created with length sum(0,...,n)
    The first token in that new sentence is the average of tokens 0 and 0 in the original, the second the average
    of tokens 0 and 1, and so on (for the whole sentence)

    -rel_toks txt file is used for the sentence_to_index dictionary  
    -m <avg or concat or max or left or right> optional: specify if the activations should be averaged or concatenated or if the max should be taken (using absolute values). Default is average. Only works for hdf5, not json
    -round <int> optional: round activation values to so many decimals (default: do not round at all). Only works for hdf5, not json
    -sampled: use this option if representations should only be combined for a sampled portion of the data. Only works for hdf5, not json

    usage: 
    python3 average_activations.py -i <infile> -o <outfile> -format <json or hdf5> -rel_toks <rel_toks.txt file> -sampled

    
    """


    parser = argparse.ArgumentParser()
    parser.add_argument('-i')
    parser.add_argument('-o')
    parser.add_argument('-format')
    parser.add_argument('-rel_toks')
    parser.add_argument('-m')
    parser.add_argument('-round')
    parser.add_argument('-sampled', action='store_true')
    parsedargs = parser.parse_args()
    
    mode = 'avg'
    if parsedargs.m is not None:
        mode = parsedargs.m
        if mode not in ['avg', 'concat', 'max', 'left', 'right']:
            print('wrong mode: one of [avg, concat, max, left, right]')
    round_to = -1
    if parsedargs.round is not None:
        round_to = int(parsedargs.round)




    if parsedargs.format == 'json':
        extract_from_json(parsedargs.i, parsedargs.o)
    else:
        if parsedargs.sampled:
            extract_from_hdf5_sampled(parsedargs.i, parsedargs.o, parsedargs.rel_toks, mode=mode, round_to=round_to)
        else:
            extract_from_hdf5(parsedargs.i, parsedargs.o, parsedargs.rel_toks, mode=mode, round_to=round_to)
    


    # extract_from_json()
    # print('done, now convert to json')
    #avg_json_strs = []
    #for line in averaged_activations:
    #    avg_json_strs.append(json.dumps(line))
    
    #print('done, now print')
    #with open('resources/Data_david/avg.json', 'w') as f:
    #    for line in avg_json_strs:
    #        f.write(line + '\n')


