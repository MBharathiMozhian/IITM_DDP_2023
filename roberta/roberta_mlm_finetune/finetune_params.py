import os

curr_path = os.path.dirname(os.getcwd())

# settings
output_dir = os.path.join(curr_path,"finetune/roberta_mlm_bpe100k/") # os.path.join(curr_path,"finetune/roberta_mlm_smiles100k/") # default
cache_dir = os.path.join(curr_path,"finetune/roberta_mlm_bpe100k/cache")
overwrite_output_dir = True # default
run_name = "default-run"
seed = 0 # default

# Model params
pretrained_model_name_or_path_bpe100k = os.path.join(curr_path,"roberta_mlm/model_output/bpe_run100k/final")
'''help=pretrained roberta mlm model on 100K SMILES using bpe tokenizer'''
pretrained_model_name_or_path_smiles100k = os.path.join(curr_path,"roberta_mlm/model_output/smiles_run100k/final")
'''help=pretrained roberta mlm model on 100K SMILES using smiles tokenizer'''
# help="Arg to HuggingFace model.from_pretrained(). Can be either a path to a local model or a model ID on HuggingFace Model Hub. If not given, trains a fresh model from scratch (non-pretrained).
freeze_base_model = False # default=False
# help="If True, freezes the parameters of the base model during training. Only the classification/regression head parameters will be trained. (Only used when `pretrained_model_name_or_path` is given.)"
is_molnet = True
# help="If true, assumes all dataset are MolNet datasets"
tokenizer_path = curr_path+"/roberta_mlm/tokenizer1"
# help="path of the tokenizer - trained on 70M SMILES"
max_tokenizer_len = 512

# RobertaConfig params (for non-pretrained models)
vocab_size = 52000
max_position_embeddings = 512
num_attention_heads = 6
num_hidden_layers = 6
type_vocab_size = 1

# train params
eval_steps = 100
logging_steps = 10 # default
early_stopping_patience = 3 # default
num_train_epochs_max = 10 # default
per_device_train_batch_size = 64 # default = 64
per_device_eval_batch_size = 64 # default
n_trials = 5
# help="Number of different combinations to try. Each combination will result in a different finetuned model"
n_seeds = 5
# help="Number of unique random seeds to try. This applies to the final best model selected after hyperparameter tuning"

# Dataset params
datasets = ["bace_classification", "bbbp", "clintox", "tox21", "hiv"]
#help="Comma separated list of MoleculeNet dataset names"
split = "scaffold" # default
#help="deepchem data loader split_type"
dataset_type = "classification"
#help="List of dataset types (ex: classification, regression). Include 1 per dataset, not necessary for MoleculeNet datasets"
