import os
curr_path = os.path.dirname(os.getcwd())

# model type
model_type = "mlm" # default - 'mlm'
'''help="["mlm", "regression", "regression_lazy"]'''

# roberta_model_configs_flags
attention_probs_dropout_probs = 0.1 # default
'''help="The dropout ratio for the attention probabilities."'''
hidden_dropout_prob = 0.1 #default
'''help="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler"'''
hidden_size_per_attention_head = 64 #default
'''help="Multiply with `num_attention_heads` to get the dimensionality of the encoder layers and the pooler layer."'''
intermediate_size = 3072 #default
'''help="Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder."'''
max_position_embeddings = 512 #default
'''help="The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048)"'''
num_attention_heads = 6 #default
'''help="Number of attention heads for each attention layer in the Transformer encoder."'''
num_hidden_layers = 6 #default
'''help="Number of hidden layers in the transformer encoder"'''
type_vocab_size = 1
'''help="vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the inputs_ids passed when calling BertModel or TFBertModel"'''
vocab_size = 52000 # to be changed to tokenizer.get_vocab_size() if BPE; 600 if SmilesTokenizer is used
'''help="The vocabulary size of the token_type_ids passed when calling BertModel or TFBertModel or loading from local path"'''
fp16 = True
'''help="Mixed precision"'''

# tokenizer_flags
smiles_tokenizer_vocab_path = curr_path+"/roberta_mlm/smiles_tokenizer/vocab.txt" # curr_path+"/roberta_mlm/smiles_tokenizer_20m" # ""
'''help="path to vocab file - strategy - smiles tokenizer on any number of SMILES"'''

# the following arguments if tokenizer = BPE
tokenizer_path = curr_path+"/roberta_mlm/tokenizer1" # ""
'''help="path to vocab file trained on 70M smiles"'''
# tokenizer_path_112m = curr_path+"/roberta_mlm/tokenizer_bpe112m" # ""
# '''help="path to vocab file for 112m"'''
output_tokenizer_dir = curr_path+"/roberta_mlm/tokenizer1"
'''help="path to vocab file - in code training"'''
# output_tokenizer_dir_112m = curr_path+"/roberta_mlm/tokenizer_bpe112m"
# '''help="path to vocab file for 112m- in code training"'''
tokenizer_type = 'smiles'
'''help="['bpe, 'smiles'] - "bpe - train from scratch", "smiles - deepchem trained tokenizer""'''
max_tokenizer_len = 512 #default
'''help="controls the maximum length to use by one of the truncation/padding parameters for the tokenizer."'''
BPE_min_frequency = 2
'''help=""'''
tokenizer_block_size = 128 # 512
'''help=""'''

# dataset_flags
dataset_path_1k = curr_path+"/datasets/final/CID-SMILES_1K_smiles.txt" # used for trial runs
'''help="path to local dataset - 1K SMILES"'''
dataset_path_100k = curr_path+"/datasets/final/CID-SMILES_100K_smiles.txt"
'''help="path to local dataset - 100K SMILES"'''
dataset_path = curr_path+"/datasets/final/CID-SMILES_20M_smiles.txt"
'''help="path to local dataset"'''
dataset_path_112m = curr_path+"/datasets/CID-SMILES_112M_smiles.txt"
'''help="path to local dataset"'''
eval_path = None
'''help="If provided, uses dataset at this path as a validation set. Otherwise, 'frac_train' is used to split the dataset."'''
output_dir = curr_path+'/roberta_mlm/model_output'
'''help="directory in which to write results"'''
smiles_output_dir = curr_path+'/roberta_mlm/smiles_model_output'
'''help="directory in which to write results"'''
run_name = "bpe_run20m" #default - to be changed depending on the requirement
'''help="subdirectory for results"'''
run_name_112m = "bpe_run112m" #default - 'default_run'
'''help="subdirectory for results for 112M smiles"'''

# train_flags
early_stopping_patience = 3 #default
'''help="Patience for the `EarlyStoppingCallback`."'''
eval_steps = 300 #default
'''help="Number of update steps between two evaluations if evaluation_strategy='steps'. Will default to the same value as logging_steps if not set."'''
frac_train = 0.95 #default
'''help="Fraction of dataset to use for training. Gets overridden by 'eval_path', if provided."'''
learning_rate = 5e-5 #default
'''help="The initial learning rate for AdamW optimizer"'''
load_best_model_at_end = True #default
'''help="Whether or not to load the best model found during training at the end of training. When set to True (for HF 4.6), the parameters `save_strategy` and `save_steps` will be ignored and the model will be saved after each evaluation."'''
logging_steps = 50 #default
'''help="Number of update steps between two logs if logging_strategy='steps'."'''
num_train_epochs = 1 #default
'''help="Total number of training epochs to perform (if not an integer, will perform the decimal part percents of the last epoch before stopping training)."'''
overwrite_output_dir = True #default
'''help="If true, overwrite the content of the output directory. Use this to continue training if output_dir points to a checkpoint directory"'''
per_device_train_batch_size = 64 #default
'''help="The batch size per GPU/TPU core/CPU for training."'''
save_total_limit = 2 #default
'''help="If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in output_dir."'''
save_steps = 300 #default # note: 'save_steps' should be a round multiple of 'eval_steps'
'''help="Number of updates steps before two checkpoint saves if `save_strategy`='steps'"'''
mlm_probability = 0.15 #default
'''help="Masking rate."'''
cloud_directory = None #default
'''help="if provided, syncs the run directory here using a callback."'''