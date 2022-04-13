

# Constituency structure in language model representations

## Installation

Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

Create a conda environment with the dependencies required for running the code, and switch to the environment by running the following commands in a bash terminal:

```
$ conda env create -f environment.yml
$ conda activate syntax-prob-env
```

## Data preparation

For running the experiments, you need a full version of the [Penn Treebank (PTB)](https://catalog.ldc.upenn.edu/LDC99T42) Wall Street Journal sections (sections 0-24). Preparation steps for replicating the experiments without the full dataset are listed below.

Put a copy of your PTB files in the directory `data/PennTreebank/`. 
Merge the files of the different PTB sections into files in `data/PennTreebank`. 
The first argument of the merging script is the path to the directory with the different sections of the PTB WSJ sections (0-24). In this case, it is `data/PennTreebank/package/treebank_3/parsed/mrg/wsj/`.

```
./merge_ptb_secs.sh data/PennTreebank/package/treebank_3/parsed/mrg/wsj/ 0 18 data/PennTreebank/ptb-0-18
./merge_ptb_secs.sh data/PennTreebank/package/treebank_3/parsed/mrg/wsj/ 19 21 data/PennTreebank/ptb-19-21
./merge_ptb_secs.sh data/PennTreebank/package/treebank_3/parsed/mrg/wsj/ 22 24 data/PennTreebank/ptb-22-24
./merge_ptb_secs.sh data/PennTreebank/package/treebank_3/parsed/mrg/wsj/ 19 24 data/PennTreebank/ptb-19-24
```

Create versions of the PTB file in which trees do not contain traces.

```
treetools transform data/PennTreebank/ptb-0-18 data/PennTreebank/ptb-train_orig.notrace --src-format brackets --dest-format brackets --trans ptb_delete_traces
treetools transform data/PennTreebank/ptb-19-21 data/PennTreebank/ptb-19-21_orig.notrace --src-format brackets --dest-format brackets --trans ptb_delete_traces
treetools transform data/PennTreebank/ptb-22-24 data/PennTreebank/ptb-22-24_orig.notrace --src-format brackets --dest-format brackets --trans ptb_delete_traces
treetools transform data/PennTreebank/ptb-19-24 data/PennTreebank/ptb-test_orig.notrace --src-format brackets --dest-format brackets --trans ptb_delete_traces
```

Create versions of the dataset with replaced tokens. This requires that you have the full dependency-parsed PTB (sections 0-24, ordered) in two files `data/PennTreebank/depptb-0-18.conllu` and `data/PennTreebank/depptb-19-24.conllu`. If you do not have these files, the following script will create dependency parses on the fly (usign spaCy). Note that errors in the dependency parses might produce errors in the manipulated datasets.

```
python3 create_manipulated_datasets.py -const_in data/PennTreebank/ptb-train_orig.notrace -dep_in data/PennTreebank/depptb-0-18.conllu -const_out data/PennTreebank/ptb-train_033.notrace -ratio 0.33 -text_out data/PennTreebank/ptb-train_033_text.txt -text_out_orig data/PennTreebank/ptb-train_orig_text.txt
python3 create_manipulated_datasets.py -const_in data/PennTreebank/ptb-train_orig.notrace -dep_in data/PennTreebank/depptb-0-18.conllu -const_out data/PennTreebank/ptb-train_067.notrace -ratio 0.67 -text_out data/PennTreebank/ptb-train_067_text.txt 
python3 create_manipulated_datasets.py -const_in data/PennTreebank/ptb-test_orig.notrace -dep_in data/PennTreebank/depptb-19-24.conllu -const_out data/PennTreebank/ptb-test_033.notrace -ratio 0.33 -text_out data/PennTreebank/ptb-test_033_text.txt -text_out_orig data/PennTreebank/ptb-test_orig_text.txt -stop_out_after 2500
python3 create_manipulated_datasets.py -const_in data/PennTreebank/ptb-test_orig.notrace -dep_in data/PennTreebank/depptb-19-24.conllu -const_out data/PennTreebank/ptb-test_067.notrace -ratio 0.67 -text_out data/PennTreebank/ptb-test_067_text.txt -text_out_orig data/PennTreebank/ptb-test_orig_text.txt -stop_out_after 2500
```


### Free PTB version

If you do not have access to the full Penn Treebank, you can replicate the experiments with the free portion of the PTB. To get this data, run 

```
python3 obtain_minimal_ptb.py
```

Create a transformed version (without traces):

```
treetools transform data/PennTreebank/ptb-3414 data/PennTreebank/ptb-3414_orig.notrace --src-format brackets --dest-format brackets --trans ptb_delete_traces
treetools transform data/PennTreebank/ptb-500 data/PennTreebank/ptb-500_orig.notrace --src-format brackets --dest-format brackets --trans ptb_delete_traces
```

Create manipulated datasets. The first command downloads the spacy model for craeting dependency parses

```
python3 -m spacy download en_core_web_lg
python3 create_manipulated_datasets.py -const_in data/PennTreebank/ptb-3414_orig.notrace -const_out data/PennTreebank/ptb-train_033.notrace -ratio 0.33 -text_out data/PennTreebank/ptb-train_033_text.txt -text_out_orig data/PennTreebank/ptb-train_orig_text.txt
python3 create_manipulated_datasets.py -const_in data/PennTreebank/ptb-3414_orig.notrace -const_out data/PennTreebank/ptb-train_067.notrace -ratio 0.67 -text_out data/PennTreebank/ptb-train_067_text.txt
python3 create_manipulated_datasets.py -const_in data/PennTreebank/ptb-500_orig.notrace -const_out data/PennTreebank/ptb-test_033.notrace -ratio 0.33 -text_out data/PennTreebank/ptb-test_033_text.txt -text_out_orig data/PennTreebank/ptb-test_orig_text.txt
python3 create_manipulated_datasets.py -const_in data/PennTreebank/ptb-500_orig.notrace -const_out data/PennTreebank/ptb-test_067.notrace -ratio 0.67 -text_out data/PennTreebank/ptb-test_067_text.txt
```

## Running the experiments

Once you created the necessary files, you can run the experiments using 

```
./prepare_and_run_chunking_exps.sh distilbert-base-uncased 8000 2000
```

This means that you run the experiments with DistilBERT, using roughly 8000 sentences for training and 2000 for evaluation. You can experiment with this amount of data on 16GB RAM, and additionally this requires 16GB of free harddrive space. 

for the chunking experiments, or 

```
./prepare_and_run_lca_exps.sh distilbert-base-uncased 50000 150
```

for the lca experiments. This requires around 120 GB of hard drive space and 16GB RAM. 
Note on data format: In the chunking experiments, the format of the data that is used as input to the experimental script is straightforward: 
1. Files with the textual data, where tokens are separated by blanks and sentences are separated by line breaks 
2. Files with the chunking labels, parallel to 1.
3. hdf5 files with the activation values from a LM. 

In the LCA experiments, the experimental script is the same, but the input files are formatted differently:
1. The input tokens are not textual data but tokens that correspond to a token pair. E.g. The token `3_14_15` corresponds to *tokens 14 and 15 in the third input sentence.*
2. Files with the LCA labels for the token pairs specified in 1., again 1 and 2 are parallel
3. Files with the combined activation values (via concat, max, or avg) from the LM.

## Running the parse tree reconstruction


Running these experiments consists of two steps. In the first step, you provide the name of the LM and the number of sentences in the training and evaluation dataset. In this case, it is 8000 sentences for training and 1000 for eval.
Then, you use the [code](https://github.com/aghie/parsing-as-pretraining) of Vilares et al. (AAAI 2020, "Parsing as Pretraining") to reconstruct and evaluate the predicted trees.


```
./prepare_and_run_tree_exps.sh distilbert-base-uncased 8000 1000
cd parsing-as-pretraining
./postprocess_and_eval.sh distilbert-base-uncased
```


## Inspecting the Results

Results for all experiments can be found in the directories `exp_chunk` and `exp_lca`, which have subdirectories for experiments with a specific LM. Each LM directory has the experimental results in subdirectories. The naming schema is such that e.g. `0672orig` means *trained on manipulated data where two thirds of the tokens are replaced, evaluated on non-manipulated data*.
The output of the accuracies for chunking also considers the accuracies for punctuation (label PCT). For LCA prediction, only labels for token pairs are selected where no token is a punctuation token.
The most important experimental results are stored in the files `results_tables.tex`. 
Detailed experimental results for each experiment setting are stored in the files `results.p` in the subdirectories, which can be loaded with `pickle` in python, e.g.

```
import pickle
with open('exp_chunk/distilbert-base-uncased/orig2orig_simple/results.p','rb') as infile:
    results = pickle.load(infile)
```

This requires pandas version 1.3.0. Make sure to use the conda environment as described above. 


