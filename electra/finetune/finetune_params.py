import os

curr_path = os.path.dirname(os.getcwd())
print('Current path: ', curr_path)
# settings
# output_dir = os.path.join(curr_path+"/electra_mlm/finetune/electra_smiles1k/") # os.path.join(curr_path,"finetune/electra_smiles1k/") # default
output_dir = curr_path+"/electra_mlm/finetune/smiles_electra_base_100k_disc/"
# cache_dir = os.path.join(curr_path+"/electra_mlm/finetune/electra_smiles1k/cache")
cache_dir = curr_path+"/electra_mlm/cache"
overwrite_output_dir = True # default
run_name = "default-run"
seed = 0 # default

# Model params
# pretrained_model_name_or_path_electra1k = os.path.join(curr_path+"/derived_impementation/checkpoints/smiles_pretrain/vanilla_11081.pth")
pretrained_model_name_or_path_electra1k = curr_path+"/electra_mlm/model_output/smiles_electra_run1k/discriminator"
'''help=pretrained electra model on 1K SMILES using smiles tokenizer'''
# pretrained_model_name_or_path_electra100k = os.path.join(curr_path+"/derived_impementation/checkpoints/smiles_pretrain/")
pretrained_model_name_or_path_electra100k = curr_path+"/electra_mlm/model_output/smiles_electra_small_run100k/discriminator"
'''help=pretrained electra model on 100K SMILES using smiles tokenizer'''
# help="Arg to HuggingFace model.from_pretrained(). Can be either a path to a local model or a model ID on HuggingFace Model Hub. If not given, trains a fresh model from scratch (non-pretrained).
freeze_base_model = True # default=False
# help="If True, freezes the parameters of the base model during training. Only the classification/regression head parameters will be trained. (Only used when `pretrained_model_name_or_path` is given.)"
is_molnet = True
# help="If true, assumes all dataset are MolNet datasets"
smiles_tokenizer_vocab_path = curr_path+"/electra_mlm/smiles_tokenizer/vocab.txt"
# help="path to vocab file - strategy - smiles tokenizer"
max_tokenizer_len = 128
size = 'small' # ['small', 'base', 'large']


# train params
eval_steps = 100
logging_steps = 10 # default
early_stopping_patience = 3 # default
num_train_epochs_max = 10 # default
per_device_train_batch_size = 8 # default = 64
per_device_eval_batch_size = 32 # default
n_trials = 7
# help="Number of different combinations to try. Each combination will result in a different finetuned model"
n_seeds = 5
# help="Number of unique random seeds to try. This applies to the final best model selected after hyperparameter tuning"
save_total_limit = 3

# Dataset params
datasets = ["bace_classification", "bbbp", "clintox", "tox21", "hiv"] # ['hiv']
#help="Comma separated list of MoleculeNet dataset names"
split = "scaffold" # default
#help="deepchem data loader split_type"
dataset_type = "classification"
#help="List of dataset types (ex: classification, regression). Include 1 per dataset, not necessary for MoleculeNet datasets"
