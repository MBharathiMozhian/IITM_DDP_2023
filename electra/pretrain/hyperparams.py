import os
curr_path = os.path.dirname(os.getcwd())

# model type
model_type = "mlm" # default - 'mlm'
'''help="["mlm", "regression", "regression_lazy"]'''

# electra_model_configs_flags
fp16 = True
'''help="Mixed precision"'''
size = 'base' # ['small', 'base', 'large']

# tokenizer_flags
smiles_tokenizer_vocab_path = curr_path+"/electra_mlm/smiles_tokenizer/vocab.txt"
'''help="path to vocab file - strategy - smiles tokenizer"'''
bpe_tokenizer_path = curr_path+'/electra_mlm/tokenizer2'
'''help="path to bpe tokenizer vocab files - strategy - bpe"'''
tokenizer_type = 'smiles'
'''help="['bpe, 'smiles'] - "bpe - train from scratch", "smiles - deepchem trained tokenizer""'''
max_tokenizer_len = 128 #default 128
'''help="controls the maximum length to use by one of the truncation/padding parameters for the tokenizer."'''
tokenizer_block_size = 128 # 512
'''help=""'''

# dataset_flags
dataset_path_1k = curr_path+"/datasets/final/CID-SMILES_1K_smiles.txt"
'''help="path to local dataset - 1K SMILES'''
dataset_path_100k = curr_path+"/datasets/final/CID-SMILES_100K_smiles.txt"
'''help="path to local dataset - 100K SMILES'''
dataset_path = curr_path+"/datasets/final/CID-SMILES_20M_smiles.txt"
'''help="path to local dataset"'''
dataset_path_112m = curr_path+"/datasets/CID-SMILES_112M_smiles.txt"
'''help="path to local dataset"'''
eval_path = None
'''help="If provided, uses dataset at this path as a validation set. Otherwise, 'frac_train' is used to split the dataset."'''
output_dir = curr_path+'/electra_mlm/model_output'
'''help="directory in which to write results"'''
cache_dir = curr_path+'/electra_mlm/cache'
'''help="where to store the pretrained models downloaded from s3"'''
run_name = "smiles_electra_base_run100k" #default - 'default_run'
'''help="subdirectory for results"'''

# train_flags
logging_strategy = "steps"
evaluation_strategy="steps"
early_stopping_patience = 3 #default
'''help="Patience for the `EarlyStoppingCallback`."'''
eval_steps = 1000 #default
'''help="Number of update steps between two evaluations if evaluation_strategy='steps'. Will default to the same value as logging_steps if not set."'''
frac_train = 0.95 #default
'''help="Fraction of dataset to use for training. Gets overridden by 'eval_path', if provided."'''
learning_rate = 5e-5 #default
'''help="The initial learning rate for AdamW optimizer"'''
load_best_model_at_end = True #default
'''help="Whether or not to load the best model found during training at the end of training. When set to True (for HF 4.6), the parameters `save_strategy` and `save_steps` will be ignored and the model will be saved after each evaluation."'''
logging_steps = 100 #default
'''help="Number of update steps between two logs if logging_strategy='steps'."'''
num_train_epochs = 3 #default = -1
'''help="Total number of training epochs to perform (if not an integer, will perform the decimal part percents of the last epoch before stopping training)."'''
overwrite_output_dir = True #default
'''help="If true, overwrite the content of the output directory. Use this to continue training if output_dir points to a checkpoint directory"'''
per_device_train_batch_size = 8 #default
'''help="The batch size per GPU/TPU core/CPU for training."'''
save_total_limit = 2 #default
'''help="If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in output_dir."'''
save_steps = 1000 #default
'''help="Number of updates steps before two checkpoint saves if `save_strategy`='steps'"'''
mlm_probability = 0.15 #default
'''help="Masking rate."'''
cloud_directory = None #default
'''help="if provided, syncs the run directory here using a callback."'''