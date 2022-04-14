# David: I made this as a script version of the notebook "syntax_probing_end_to_end_example" to make it easier to colect the results from multiple experiments

import sys
sys.path.append('NeuroX/')

from neurox.interpretation import linear_probe, utils, ablation
from neurox.data import loader as data_loader
import argparse
import json
import numpy as np
import pandas as pd
import os
import pickle
import re
import h5py
import torch
import matplotlib.pyplot as plt
import time
import csv

def findMinimalNeurons(tensors, train_tokens, dev_tokens, L1=0.0001, L2=0.0001, MODEL='aux_model.pt', idx2label=dict(), JSON='mappings.json', results=dict()):

    print("Building model with: L1: " + str(L1) + " L2: " + str(L2))
    root_model = linear_probe.train_logistic_regression_probe(
        tensors['X'], tensors['y'], lambda_l1=L1, lambda_l2=L2, num_epochs=10, batch_size=128)
    train_accuracies = linear_probe.evaluate_probe(
        root_model, tensors['X'], tensors['y'], idx2label, source_tokens=train_tokens['source'])
    dev_accuracies = linear_probe.evaluate_probe(
        root_model, tensors['X_dev'], tensors['y_dev'], idx2label, source_tokens=dev_tokens['source'])
    accuracyElastic = dev_accuracies['__OVERALL__']
    accuracies = {'all': {'train': train_accuracies['__OVERALL__'],
                          'dev': dev_accuracies['__OVERALL__']}}

    torch.save(root_model, MODEL)
    with open(JSON, 'w') as fp:
        json.dump(tensors['mappings'], fp)
    
    # for the out-of-class idx
    l2i2 = label2idx.copy()
    l2i2.pop(config['task_specific_tag'])
    ordering, cutoffs = linear_probe.get_neuron_ordering(
        root_model, l2i2, search_stride=1000)
    results['neuron_ordering'] = ordering
    for percentage in [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]:
        top_neurons = ordering[:int(percentage*len(ordering))]
        print("Selecting %d top neurons" % (len(top_neurons)))
        X_filtered = ablation.filter_activations_keep_neurons(
            tensors['X'], top_neurons)
        X_dev_filtered = ablation.filter_activations_keep_neurons(
            tensors['X_dev'], top_neurons)
        model_temp = linear_probe.train_logistic_regression_probe(
            X_filtered, tensors['y'], lambda_l1=L1, lambda_l2=L2, num_epochs=10, batch_size=config['batch_size'])
        results[str(percentage)+'_top_train'] = linear_probe.evaluate_probe(
            model_temp, X_filtered, tensors['y'], idx2label, source_tokens=train_tokens['source'])
        results[str(percentage)+'_top_dev'] = linear_probe.evaluate_probe(
            model_temp, X_dev_filtered, tensors['y_dev'], idx2label, source_tokens=dev_tokens['source'])
        accuracyElasticTop = results[str(percentage)+'_top_dev']['__OVERALL__']
        del X_filtered, X_dev_filtered

        random_neurons = np.random.choice(
            ordering, size=len(top_neurons), replace=False)
        print("Selecting %d random neurons" % (len(random_neurons)))
        X_filtered = ablation.filter_activations_keep_neurons(
            tensors['X'], random_neurons)
        X_dev_filtered = ablation.filter_activations_keep_neurons(
            tensors['X_dev'], random_neurons)
        model_temp = linear_probe.train_logistic_regression_probe(
            X_filtered, tensors['y'], lambda_l1=L1, lambda_l2=L2, num_epochs=10, batch_size=config['batch_size'])
        results[str(percentage)+'_random_train'] = linear_probe.evaluate_probe(
            model_temp, X_filtered, tensors['y'], idx2label, source_tokens=train_tokens['source'])
        results[str(percentage)+'_random_dev'] = linear_probe.evaluate_probe(
            model_temp, X_dev_filtered, tensors['y_dev'], idx2label, source_tokens=dev_tokens['source'])
        accuracyElasticRandom = results[str(percentage)+'_random_dev']['__OVERALL__']
        del X_filtered, X_dev_filtered
        
        bottom_neurons = ordering[-int(percentage*len(ordering)):]
        print("Selecting %d bottom neurons" % (len(bottom_neurons)))
        X_filtered = ablation.filter_activations_keep_neurons(
            tensors['X'], bottom_neurons)
        X_dev_filtered = ablation.filter_activations_keep_neurons(
            tensors['X_dev'], bottom_neurons)
        model_temp = linear_probe.train_logistic_regression_probe(
            X_filtered, tensors['y'], lambda_l1=L1, lambda_l2=L2, num_epochs=10, batch_size=config['batch_size'])
        results[str(percentage)+'_bot_train'] = linear_probe.evaluate_probe(
            model_temp, X_filtered, tensors['y'], idx2label, source_tokens=train_tokens['source'])
        results[str(percentage)+'_bot_dev'] = linear_probe.evaluate_probe(
            model_temp, X_dev_filtered, tensors['y_dev'], idx2label, source_tokens=dev_tokens['source'])
        accuracyElasticBottom = results[str(percentage)+'_bot_dev']['__OVERALL__']
        del X_filtered, X_dev_filtered

        print("Accuracy with selected: " + str(percentage*100) +
              str("% = ") + str(len(top_neurons)) + str(" neurons"))
        print("Overall Accuracy for this Lambda Set: " + str(accuracyElastic))
        print("Top Neurons Accuracy: " + str(accuracyElasticTop))
        print("Random Neuron Accuracy: " + str(accuracyElasticRandom))
        print("Bottom Neuron Accuracy: " + str(accuracyElasticBottom))
        accuracies[str(percentage*100)+"% = "+str(len(top_neurons))+' top'] = {
            'train': results[str(percentage)+'_top_train']['__OVERALL__'],
            'dev': accuracyElasticTop
        }
        accuracies[str(percentage*100)+"% = "+str(len(top_neurons))+' random'] = {
            'train': results[str(percentage)+'_random_train']['__OVERALL__'],
            'dev': accuracyElasticRandom
        }
        accuracies[str(percentage*100)+"% = "+str(len(top_neurons))+' bot'] = {
            'train': results[str(percentage)+'_bot_train']['__OVERALL__'],
            'dev': accuracyElasticBottom
        }
    accuracies_df = pd.DataFrame(accuracies)
    return accuracies_df, root_model, ordering, results


def leftRightBaseline(config, train_tokens, dev_tokens, tensors, idx2label, results, duplicate=False):
    print("try baseline: only left or only right token")
    exp_name_left = 'bl_'
    exp_name_right = 'bl_'
    if duplicate:
        exp_name_left += 'left_left_'
        exp_name_right += 'right_right_'
    else:
        exp_name_left += 'left_'
        exp_name_right += 'right_'
    
    leftneurons = []
    for i in range(int(config['num_neurons_per_layer']/2)):
        leftneurons = leftneurons + \
            [i+(l*config['num_neurons_per_layer'])
                for l in range(config['num_layers'])]
    leftneurons = sorted(leftneurons)
    rightneurons = sorted([x for x in range(
        config['num_layers']*config['num_neurons_per_layer']) if x not in leftneurons])
    assert len(leftneurons) == len(rightneurons)
    X_left = ablation.filter_activations_keep_neurons(tensors['X'], leftneurons)
    X_dev_left = ablation.filter_activations_keep_neurons(
        tensors['X_dev'], leftneurons)
    X_right = ablation.filter_activations_keep_neurons(tensors['X'], rightneurons)
    X_dev_right = ablation.filter_activations_keep_neurons(
        tensors['X_dev'], rightneurons)

    if duplicate:
        x_old_num_neurons = X_left.shape[1]
        X_left = np.concatenate([X_left, X_left], axis=1)
        assert x_old_num_neurons*2 == X_left.shape[1]
        X_right = np.concatenate([X_right, X_right], axis=1)
        X_dev_left = np.concatenate([X_dev_left, X_dev_left], axis=1)
        X_dev_right = np.concatenate([X_dev_right, X_dev_right], axis=1)

    print('train and evaluate models on left token only')
    model_left = linear_probe.train_logistic_regression_probe(
        X_left, tensors['y'], lambda_l1=config['L1'], lambda_l2=config['L2'], num_epochs=10, batch_size=config['batch_size'])
    results[exp_name_left+'train'] = linear_probe.evaluate_probe(model_left, X_left, tensors['y'], idx2label, source_tokens=train_tokens['source'])
    results[exp_name_left+'dev'] = linear_probe.evaluate_probe(model_left, X_dev_left, tensors['y_dev'], idx2label, source_tokens=dev_tokens['source'])
    print('train and evaluate models on right token only')

    model_right = linear_probe.train_logistic_regression_probe(
        X_right, tensors['y'], lambda_l1=config['L1'], lambda_l2=config['L2'], num_epochs=10, batch_size=config['batch_size'])
    results[exp_name_right+'train'] = linear_probe.evaluate_probe(model_right, X_right, tensors['y'], idx2label, source_tokens=train_tokens['source'])
    results[exp_name_right+'dev'] = linear_probe.evaluate_probe(model_right, X_dev_right, tensors['y_dev'], idx2label, source_tokens=dev_tokens['source'])
    
    del X_left, X_dev_left, X_right, X_dev_right
    one_token_results_df = pd.DataFrame({'train left': results[exp_name_left+'train'], 'dev left': results[exp_name_left+'dev'],
                                         'train right': results[exp_name_right+'train'], 'dev right': results[exp_name_right+'dev']})
    print('results on left and right only: ')
    print(one_token_results_df)
    print(one_token_results_df.to_latex())

    return results


def printImportantNeuronsByLayer(config, ordering):
    percentage = 0.05
    top_neurons = ordering[:int(percentage*len(ordering))]
    layerWidth = config['num_neurons_per_layer']
    layers = [0] * config['num_layers']
    layerIndex = list(range(0, config['num_layers']))
    for i in top_neurons:
        layerNum = int(i/layerWidth)
        layers[layerNum] = layers[layerNum] + 1

    plt.bar(layerIndex, layers)
    plt.title('Important neurons per layer')
    plt.draw()
    time.sleep(2)
    plt.savefig(config['out_dir']+"/important_neurons_per_layer.pdf")


def layerWiseProbing(config, tensors, train_tokens, dev_tokens, results=dict()):

    layerAccuracy_train = [0] * config['num_layers']
    layerAccuracy_dev = [0] * config['num_layers']
    layerIndex = list(range(config['num_layers']))

    for layer in range(config['num_layers']):

        print("Selecting from Layer " + str(layer))
        filter_layers = "f" + str(layer)
        print(config['num_neurons_per_layer'], config['num_layers'])
        X_filtered = ablation.filter_activations_by_layers(tensors['X'], [layer], config['num_layers'])
        X_dev_filtered = ablation.filter_activations_by_layers(tensors['X_dev'], [layer], config['num_layers'])

        label2idx, idx2label, src2idx, idx2src = tensors['mappings']

        print("Building model...")

        model_filtered = linear_probe.train_logistic_regression_probe(
            X_filtered, tensors['y'], lambda_l1=config['L1'], lambda_l2=config['L2'], num_epochs=10, batch_size=config['batch_size'])
        overall_train_accuracies = linear_probe.evaluate_probe(
            model_filtered, X_filtered, tensors['y'], idx2label, source_tokens=train_tokens['source'])
        overall_dev_accuracies, predictions = linear_probe.evaluate_probe(
            model_filtered,  X_dev_filtered, tensors['y_dev'], idx2label, source_tokens=dev_tokens['source'], return_predictions=True)
        layerAccuracy_train[layer-1] = overall_train_accuracies['__OVERALL__']
        layerAccuracy_dev[layer-1] = overall_dev_accuracies['__OVERALL__']
        results['layer_'+str(layer-1)+'_train_acc'] = overall_train_accuracies
        results['layer_'+str(layer-1)+'_dev_acc'] = overall_dev_accuracies
        results['layer_'+str(layer-1)+'_dev_conf_matrix'] = createConfMatrix(predictions)
        if len(predictions[0][0].split('_')) == 3:
            results['layer_'+str(layer-1)+'_dev_acc_per_length'] = analyzeCorrectnessPerLength(predictions)
        del X_filtered, X_dev_filtered

    plt.bar(layerIndex, layerAccuracy_train)
    plt.title('Accuracy per layer: train')
    plt.draw()
    time.sleep(2)
    plt.savefig(config['out_dir']+"/acc_per_layer_train.pdf")
    plt.bar(layerIndex, layerAccuracy_dev)
    plt.title('Accuracy per layer: dev')
    plt.draw()
    time.sleep(2)
    plt.savefig(config['out_dir']+"/acc_per_layer_dev.pdf")

    # spread of neurons per property

    percentage = 0.05
    top_neurons_global = ordering[:int(percentage*len(ordering))]
    fac = 1
    top_neurons_local = []
    
    l2i2 = label2idx.copy()
    l2i2.pop(config['task_specific_tag'])
    while (len(top_neurons_local) < len(top_neurons_global)):
        top_neurons_local, classWise = linear_probe.get_top_neurons(
            model, percentage*fac, l2i2)
        fac = fac + 0.1

    properties = []
    numNeurons = []

    for k, v in classWise.items():
        #print (k, v)
        properties.append(k)
        numNeurons.append(len(v))
    plt.figure(figsize=(30, 5))
    plt.bar(properties, numNeurons)
    plt.title('spread of neurons per property')
    # plt.show()
    plt.draw()
    time.sleep(2)
    plt.savefig(config['out_dir']+"/spread_of_neurons_per_property.pdf")

    # Spread per layer per property
    top_neurons_global = ordering[:int(percentage*len(ordering))]
    fac = 1
    top_neurons_local = []

    while (len(top_neurons_local) < len(top_neurons_global)):
        top_neurons_local, classWise = linear_probe.get_top_neurons(
            model, percentage*fac, l2i2)
        fac = fac + 0.1

    properties = []
    numNeurons = []

    layerWidth = config['num_neurons_per_layer']
    winner = list(range(0, config['num_layers']))

    if len(classWise) < 35:
        for k, v in classWise.items():
            layers = [0] * config['num_layers']
            for i in v:
                layerNum = int(i/layerWidth)
                layers[layerNum] = layers[layerNum] + 1
    
            print(k)
            plt.figure(figsize=(25, 5))
            plt.bar(winner, layers)
            plt.title(str(k))
            plt.draw()
            time.sleep(2)
            plt.savefig(config['out_dir']+"/spread_per_prop_" + str(k) + ".pdf")

    return results

def initTeXOutFile(filename):
    initString = "\\documentclass{article}\n\\usepackage[utf8]{inputenc}\n\\usepackage[margin=1in]{geometry}\n"
    initString += "\\title{Constituency Structure in Neural Language Models}\n"
    initString += "\\begin{document}\n\\maketitle\n"

    tex_out_file = open(filename, 'w')

    tex_out_file.write(initString)

    return tex_out_file

def createConfMatrix(predictions, save_to_file=None):
    """returns the confusion matrix of the preditcions and optionally saves to a file
    if save_to_file is a path to a file"""
    # predictions is then a list of tuples (src_token, True/False, true_label, pred_label)
    pred_to_actual_to_count = dict()
    for _, _, true_label, pred_label in predictions:
        if pred_label not in pred_to_actual_to_count:
            pred_to_actual_to_count[pred_label] = dict()
        if true_label not in pred_to_actual_to_count[pred_label]:
            pred_to_actual_to_count[pred_label][true_label] = 0
        pred_to_actual_to_count[pred_label][true_label] = pred_to_actual_to_count[pred_label][true_label] + 1
    res_df = pd.DataFrame(pred_to_actual_to_count).fillna(0)
    res_df = res_df.sort_index(axis=1)
    res_df = res_df.sort_index(axis=0)
    if save_to_file is not None:
        res_df.to_pickle(save_to_file)
    return res_df

def writePredsToFile(predictions, filename):
    with open(filename, 'w') as outfile:
        wr = csv.writer(outfile, delimiter='\t')
        wr.writerow(['token', 'correct', 'true_label', 'pred_label'])
        wr.writerows(predictions)

def analyzeCorrectnessPerLength(predictions, save_to_file=None):
    len_to_true = dict()
    len_to_false = dict()
    for token, correct, _, _ in predictions:
        [_, i, j]= token.split('_')
        span = int(j)-int(i)
        if correct:
            if span not in len_to_true:
                len_to_true[span] = 0
            len_to_true[span] = len_to_true[span] + 1
        else:
            if span not in len_to_false:
                len_to_false[span] = 0
            len_to_false[span] = len_to_false[span] + 1            
    result = {"T": len_to_true, "F": len_to_false}
    res_df = pd.DataFrame(result).fillna(0)
    res_df = res_df.sort_index(axis=1)
    res_df = res_df.sort_index(axis=0)
    if save_to_file is not None:
        res_df.to_pickle(save_to_file)
    return res_df

if __name__ == '__main__':

    # Vorarbeit
    parser = argparse.ArgumentParser()
    parser.add_argument('-out_dir')
    parser.add_argument('-train_tokens')
    parser.add_argument('-train_labels')
    parser.add_argument('-train_activations')
    parser.add_argument('-dev_tokens')
    parser.add_argument('-dev_labels')
    parser.add_argument('-dev_activations')
    parser.add_argument('-no_detailed_analysis', action='store_true')
    parser.add_argument('-lr_baseline', action='store_true', default=False)
    parser.add_argument('-layer_selection', default=None)
    parsedargs = parser.parse_args()
    config = vars(parsedargs)
    if 'no_detailed_analysis' not in config:
        config['no_detailed_analysis'] = False

    # Create required directories
    os.makedirs(config['out_dir'], exist_ok=True)

    results = dict()
    result_tex_out_file = initTeXOutFile(config['out_dir']+'results_tables.tex')
    # Training setup
    config['num_epochs'] = 10
    config['batch_size'] = 128

    config['L1'] = 0.0001
    config['L2'] = 0.0001

    config['max_sent_length'] = 10000
    config['is_brnn'] = False
    config['task_specific_tag'] = 'NN'

    representations = h5py.File(config['train_activations'], "r")
    for i in range(200):
        if str(i) in representations.keys():
            sentRepresentation = torch.FloatTensor(representations[str(i)])
            break
    config['num_neurons_per_layer'] = sentRepresentation.shape[2]

    # Loading data
    print("Loading activations...")
    train_activations, config['num_layers'] = data_loader.load_activations(
        config['train_activations'], config['num_neurons_per_layer'], is_brnn=config['is_brnn'], layerSelection=config['layer_selection'])
    dev_activations, _ = data_loader.load_activations(
        config['dev_activations'], config['num_neurons_per_layer'], is_brnn=config['is_brnn'], layerSelection=config['layer_selection'])
    print("Number of train sentences: %d" % (len(train_activations)))
    print("Number of dev sentences: %d" % (len(dev_activations)))

    print('loading tokens and labels')
    train_tokens = data_loader.load_data(
        config['train_tokens'], config['train_labels'], train_activations, config['max_sent_length'])
    dev_tokens = data_loader.load_data(
        config['dev_tokens'], config['dev_labels'], dev_activations, config['max_sent_length'])
    NUM_TOKENS = sum([len(t) for t in train_tokens['target']])

    print('Number of total train tokens: %d' % (NUM_TOKENS))
    NUM_SOURCE_TOKENS = sum([len(t) for t in train_tokens['source']])
    print('Number of source words: %d' % (NUM_SOURCE_TOKENS))
    NUM_NEURONS = train_activations[0].shape[1]
    print('Number of neurons: %d' % (NUM_NEURONS))
    print("Number of layers: %d" % (config['num_layers']))

    # create tensors
    print("Creating train tensors...")
    tensors = dict()
    tensors['X'], tensors['y'], tensors['mappings'] = utils.create_tensors(
        train_tokens, train_activations, config['task_specific_tag'], dtype='float16')
    # tensors['X'] = tensors['X'].half()
    print(tensors['X'].shape, tensors['X'].dtype)
    print(tensors['y'].shape, tensors['y'].dtype)
    print("Creating dev tensors...")
    tensors['X_dev'], tensors['y_dev'], tensors['mappings'] = utils.create_tensors(
        dev_tokens, dev_activations, config['task_specific_tag'], mappings = tensors['mappings'], dtype='float16')
    if config['no_detailed_analysis']:
        del train_activations, dev_activations

    # tensors['X_dev'] = tensors['X_dev'].half()
    label2idx, idx2label, src2idx, idx2src = tensors['mappings']

    if 'lr_baseline' in config and config['lr_baseline']:
        results = leftRightBaseline(config, train_tokens, dev_tokens, tensors, idx2label, results, duplicate=False)
        if 'large' not in config['out_dir']:
            results = leftRightBaseline(config, train_tokens, dev_tokens, tensors, idx2label, results, duplicate=True)

    print("Building model...")
    model = linear_probe.train_logistic_regression_probe(tensors['X'], tensors['y'], lambda_l1=0,
                                     lambda_l2=0, num_epochs=config['num_epochs'], batch_size=config['batch_size'])
    overall_train_accs, general_train_preds = linear_probe.evaluate_probe(model, tensors['X'], tensors['y'], idx2label, return_predictions=True)
    results['general_acc_train'] = overall_train_accs
    overall_dev_accs, general_dev_preds = linear_probe.evaluate_probe(model, tensors['X_dev'], tensors['y_dev'], idx2label, return_predictions=True, source_tokens=dev_tokens['source'])
    results['general_acc_dev'] = overall_dev_accs
    results['general_dev_conf_matrix'] = createConfMatrix(general_dev_preds) 
    # store individual predictions in file 
    writePredsToFile(general_dev_preds, config['out_dir']+'pred_labels_test.tsv')
    writePredsToFile(general_train_preds, config['out_dir']+'pred_labels_train.tsv')
    if len(general_dev_preds[0][0].split('_')) == 3:
        results['general_dev_per_length'] = analyzeCorrectnessPerLength(general_dev_preds)
    overall_accuracies = pd.DataFrame({'train': results['general_acc_train'],
                                       'dev': results['general_acc_dev']}).fillna(0)

    print('=============')
    print('Overall accuracies: ')
    print(overall_accuracies)
    print('=============')
    result_tex_out_file.write("\\section{Overall accuracies}\n")
    result_tex_out_file.write(overall_accuracies.to_latex())
    result_tex_out_file.write("\n\n")
    if not(config['no_detailed_analysis']):
        # run experiments with minimal neurons
        
        minimal_accuracies, model, ordering, results = findMinimalNeurons(tensors, train_tokens, dev_tokens, idx2label=idx2label, MODEL=config['out_dir']+'classifier_for_min_neurons.pt', JSON=config['out_dir'] +'mappings.json', results=results)
        print('results of experiments with minimal neurons: ')
        print(minimal_accuracies)
        result_tex_out_file.write("\\section{Minimal accuracies}\n")
        result_tex_out_file.write(minimal_accuracies.to_latex())
        # show neurons from which layers the algo thinks are important
        
        printImportantNeuronsByLayer(config, ordering)
        
        # layer wise probe accuracy
        results = layerWiseProbing(config, tensors, train_tokens, dev_tokens, results=results)


    results['config'] = config
    with open(config['out_dir']+'/results.p', 'wb') as handle:
        pickle.dump(results, handle)

    result_tex_out_file.close()
