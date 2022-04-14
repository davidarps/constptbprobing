#!/bin/bash

model=$1
traincutoff=$2
testcutoff=$3
layersel="second"
datadir="parsing-as-pretraining/exp_trees/"
modeldir=$datadir"/"$model"/"
mkdir $datadir
mkdir $modeldir

echo "using layer selection for experiments: every x-th layer: "$layersel


echo "STEP: extract labels training set"
python3 data_prep/extract_lca_from_ptb.py -ptb_notr data/PennTreebank/ptb-train_orig.notrace -text_toks $datadir/train_text.txt -rel_toks $datadir/train_rel_toks.txt -rel_labels $datadir/train_rel_labels.txt -next -shared_levels $datadir/train_shared_levels.txt -unary $datadir/train_unaries.txt -tree_out $datadir/train_gold_trees.txt -cutoff $traincutoff


sleep 2

echo "STEP: extract labels test set"
python3 data_prep/extract_lca_from_ptb.py -ptb_notr data/PennTreebank/ptb-19-21_orig.notrace -text_toks $datadir/test_text.txt -rel_toks $datadir/test_rel_toks.txt -rel_labels $datadir/test_rel_labels.txt -next -shared_levels $datadir/test_shared_levels.txt -unary $datadir/test_unaries.txt -tree_out $datadir/test_gold_trees.txt -cutoff $testcutoff

sleep 2

echo "STEP: extract POS tags and words file for eval later"
python3 data_prep/treebank_to_pos_and_words.py -tb $datadir/train_gold_trees.txt -o $datadir/train_pos_and_words.txt
python3 data_prep/treebank_to_pos_and_words.py -tb $datadir/test_gold_trees.txt -o $datadir/test_pos_and_words.txt

echo "STEP: replace some quotes in the text files"
./data_prep/replace-quotes-XLNet.sh $datadir/train_text.txt
./data_prep/replace-quotes-XLNet.sh $datadir/test_text.txt

echo "STEP: extract representations from "$model
python3 NeuroX/neurox/data/extraction/transformers_extractor.py --aggregation average $model $datadir/train_text.txt $modeldir/train_activations.hdf5
python3 NeuroX/neurox/data/extraction/transformers_extractor.py --aggregation average $model $datadir/test_text.txt $modeldir/test_activations.hdf5

echo "STEP: concat activations"
python3 data_prep/combine_activations.py -i $modeldir/train_activations.hdf5 -o $modeldir/train_concat_activations.hdf5 -rel_toks $datadir/train_rel_toks.txt -m concat -sampled
python3 data_prep/combine_activations.py -i $modeldir/test_activations.hdf5 -o $modeldir/test_concat_activations.hdf5 -rel_toks $datadir/test_rel_toks.txt -m concat -sampled

echo "STEP: run experiments"
python3 syntax_probing_experiments.py \
    -out_dir $modeldir"/concat_lca/" \
    -train_tokens $datadir/train_rel_toks.txt \
    -train_labels $datadir/train_rel_labels.txt \
    -dev_tokens $datadir/test_rel_toks.txt \
    -dev_labels $datadir/test_rel_labels.txt \
    -train_activations $modeldir/train_concat_activations.hdf5 \
    -dev_activations $modeldir/test_concat_activations.hdf5 \
    -no_detailed_analysis \
    -layer_selection $layersel

sleep 2

echo "STEP: Lev experiments"
python3 syntax_probing_experiments.py \
    -out_dir $modeldir"/concat_lev/" \
    -train_tokens $datadir/train_rel_toks.txt \
    -train_labels $datadir/train_shared_levels.txt \
    -dev_tokens $datadir/test_rel_toks.txt \
    -dev_labels $datadir/test_shared_levels.txt \
    -train_activations $modeldir/train_concat_activations.hdf5 \
    -dev_activations $modeldir/test_concat_activations.hdf5 \
    -no_detailed_analysis \
    -layer_selection $layersel
    # -lr_baseline

sleep 2

echo "STEP: Unaries experiments"
python3 syntax_probing_experiments.py \
    -out_dir $modeldir"/concat_unary/" \
    -train_tokens $datadir/train_rel_toks.txt \
    -train_labels $datadir/train_unaries.txt \
    -dev_tokens $datadir/test_rel_toks.txt \
    -dev_labels $datadir/test_unaries.txt \
    -train_activations $modeldir/train_concat_activations.hdf5 \
    -dev_activations $modeldir/test_concat_activations.hdf5 \
    -no_detailed_analysis \
    -layer_selection $layersel


echo "STEP: Postprocessing, reformat the predictions"
python3 data_prep/tsv_preds_to_line_by_line.py -i $modeldir/concat_lca/pred_labels_test.tsv -o $modeldir/concat_lca/pred_labels_test.txt
python3 data_prep/tsv_preds_to_line_by_line.py -i $modeldir/concat_lev/pred_labels_test.tsv -o $modeldir/concat_lev/pred_labels_test.txt
python3 data_prep/tsv_preds_to_line_by_line.py -i $modeldir/concat_unary/pred_labels_test.tsv -o $modeldir/concat_unary/pred_labels_test.txt

echo "STEP: Done"



