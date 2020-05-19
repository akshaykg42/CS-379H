
# Topic-Focused Extractive Summarization

**This code is for paper:** `Topic-Focused Extractive Summarization` (https://apps.cs.utexas.edu/apps/sites/default/files/tech_reports/Akshay_Honors_Thesis.pdf)

**Python version**: This code is in Python3.7

**Package Requirements**: pytorch sklearn numpy rouge transformers allennlp_models nltk tqdm allennlp spacy

## Data Preparation For CNN/Dailymail

#### Step 1 Download Stories
Download and unzip the `stories` directories from [here](http://cs.nyu.edu/~kcho/DMQA/) for both CNN and Daily Mail. Put all  `.story` files for your domain (you'll have to filter them yourself) in one directory (e.g. `../raw_stories`)

####  Step 2. Jsonify Stories

```
python preprocess.py -mode jsonify -raw_path RAW_PATH -dataset_name DATASET_NAME
```

* `RAW_PATH` is the directory containing story files (`../raw_stories`)
* `DATASET_NAME` is the name you're choosing for the dataset containing all your stories - this will be used for all actions concerning this data moving forward. It also determines the target directory to save the generated json files (`../data/DATASET_NAME/raw/*.json`)


####  Step 3. Extract BERT Tokens and Linear Features
```
python preprocess.py -mode bert_tokens_and_linear_features -dataset_name DATASET_NAME [-overwrite]
```

* This extracts linear features and BERT tokens for each document in a dataset to `../data/DATASET_NAME/linear/` and `../data/DATASET_NAME/bert/` respectively, with `train/`, `test/` and `val/` subdirectories in each
* `DATASET_NAME` is the name of the dataset for which to extract tokens/features, json files for documents are extracted based on this
* If `-overwrite` is set and train/test/val files have already been written for this dataset, then they are overwritten with a new train/test/val split, otherwise the procedure exits

####  Step 4. Construct Oracle Extractive Summaries
```
python preprocess.py -mode construct_oracles -dataset_name DATASET_NAME [-vanilla_oracles]
```

* `DATASET_NAME` is the name of the dataset for which to construct oracle extractive summaries, they are saved in `../data/DATASET_NAME/raw/oracles.json`
* If `-vanilla_oracles` is set, then the vanilla oracle construction algorithm is used (see code for details)

####  Step 5. Get Topic Representations for Summaries
```
python preprocess.py -mode topic_clustering -dataset_name DATASET_NAME
```

* `DATASET_NAME` is the name of the dataset for which to generate topic representations for summaries, they are saved in `../data/DATASET_NAME/raw/topics.json`

## Model Training

```
python train.py -dataset_name DATASET_NAME -model_type MODEL_TYPE -topic TOPIC -batch_size BATCH -epochs EPOCHS [-mini]
```

* `DATASET_NAME` is the name of the dataset for which to train a model of topic `TOPIC` and `MODEL_TYPE` is the type of model to train (`linear` or `bert`), models are saved in `../models/DATASET_NAME/MODEL_TYPE/` as either `TOPIC/` for BERT models or `TOPIC.th` for Linear models
* Recommended `BATCH` for BERT model is `1`, and for Linear model is `16`
* Recommended `EPOCHS` for BERT model is `4` and for Linear model is `100`
* If BERT model keeps running into memory on your documents because they're too long, try `-mini` - this will shorten all documents to `k=10` sentences, preserving the oracle, for training. 

## System Evaluation
After models for all topics have been trained, run
```
python eval.py -dataset_name DATASET_NAME -model_type MODEL_TYPE -mode MODE [-topics TOPICS] [-write]
```
* `DATASET_NAME` is the name of the dataset for which to test the system using models of type `MODEL_TYPE` (`linear` or `bert`)
* `MODE` can be `vanilla`, `reconstruct` or `ranking`, these are three different evaluation schemes
* `TOPICS` is of the form `1 2 3 4`
* `-topics` is only used for when `MODE` is `vanilla`, it is the list of topics for which to build a summary - for other modes all topics are used. If this is not specified for `vanilla` all topics are assumed
* If `-write` is set, the summaries produced during evaluation are written to `../results/DATASET_NAME/MODEL_TYPE_MODE_[TOPICS].txt`
