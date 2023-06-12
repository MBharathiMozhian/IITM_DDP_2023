import os
'''Modifying the script to change HOME directory'''
os.environ["HOME"] ="/scratch/scratch6/bharathimozhian/electra_mlm/derived_implementation/"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
import glob

import transformers
from transformers.trainer_callback import EarlyStoppingCallback

import torch
from torch.utils.data import random_split

import wandb
wandb.login(key="639e80381d96cc4d5ce8e3e6cadf7261701e661d")
wandb.init(project='bpe_chemberta20m')

from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

from tokenizers import ByteLevelBPETokenizer

from raw_text_dataset import RawTextDataset
import hyperparams


torch.manual_seed(0)

is_gpu = torch.cuda.is_available()
# device = torch.device('cpu')

config = RobertaConfig(
    vocab_size=52000,
    max_position_embeddings=512,
    num_attention_heads=6,
    num_hidden_layers=6,
    hidden_size=64 * 6,
    intermediate_size=4 * 3072,
    type_vocab_size=1,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    is_gpu=True,
)

if hyperparams.tokenizer_path:
    tokenizer_path = hyperparams.tokenizer_path
elif hyperparams.tokenizer_type.upper() == "BPE":
    tokenizer_path = hyperparams.output_tokenizer_dir
    if not os.path.isdir(tokenizer_path):
        os.makedirs(tokenizer_path)
    
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=hyperparams.dataset_path, vocab_size=hyperparams.vocab_size, min_frequency=hyperparams.BPE_min_frequency,
    special_tokens=["<s>","<pad>","</s>","<unk>","<mask>"])
    tokenizer.save_model(tokenizer_path)
else:
    print("Please provide a tokenizer path - either BPE or SMILES tokenizer")

tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path, max_len=hyperparams.max_tokenizer_len)

model = RobertaForMaskedLM(config=config)
# model = model.to(device)
print(f"Model size: {model.num_parameters()} parameters.")

dataset = RawTextDataset(tokenizer=tokenizer, file_path=hyperparams.dataset_path, block_size=hyperparams.tokenizer_block_size)
# dataset = LineByLineTextDataset(
#     tokenizer=tokenizer,
#     file_path=hyperparams.dataset_path,
#     block_size=512,
# )

# train_size = max(int(hyperparams.frac_train * len(dataset)), 1)
train_size = max(int(0.99975 * len(dataset)), 1)
eval_size = len(dataset) - train_size
print(f"Train size: {train_size}")
print(f"Eval size: {eval_size}")

train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=hyperparams.mlm_probability
)

train_args = TrainingArguments(
    # no_cuda=True,
    evaluation_strategy="steps",
    eval_steps=50000, # 50 # 2000 for 0.01 epoch # 50000 for 1 epoch
    load_best_model_at_end=True,
    logging_steps=100,
    output_dir=os.path.join(hyperparams.output_dir, hyperparams.run_name),
    overwrite_output_dir=hyperparams.overwrite_output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=hyperparams.per_device_train_batch_size,
    save_steps=50000, # 500 # 2000 for 0.01 epoch # 50000 for 1 epoch
    save_total_limit=hyperparams.save_total_limit,
    fp16 = is_gpu and hyperparams.fp16,
    report_to='wandb',
    run_name=hyperparams.run_name,
)

trainer = Trainer(
    model=model,
    args=train_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

run_dir = os.path.join(hyperparams.output_dir, hyperparams.run_name)

# if there is a checkpoint available, use it
# checkpoints = glob.glob(os.path.join(run_dir, "checkpoint-*"))
# print('checkpoints', checkpoints)
# if checkpoints:
#     iters = [int(x.split("-")[-1]) for x in checkpoints if "checkpoint" in x]
#     iters.sort()
#     latest_checkpoint = os.path.join(run_dir, f"checkpoint-{iters[-1]}")
#     print(latest_checkpoint)
#     print(f"Loading model from latest checkpoint: {latest_checkpoint}")
#     trainer.train(resume_from_checkpoint=latest_checkpoint)
# else:
checkpoints = glob.glob(os.path.join(run_dir, "final"))
print('checkpoints', checkpoints)
if checkpoints:
    latest_checkpoint = os.path.join(run_dir, "final")
    print(latest_checkpoint)
    print(f"Loading model from latest checkpoint: {latest_checkpoint}")
    torch.cuda.empty_cache()
    trainer.train(resume_from_checkpoint=latest_checkpoint)
else:
    torch.cuda.empty_cache()
    trainer.train()
    
# torch.cuda.empty_cache()
# trainer.train()
# torch.cuda.empty_cache()

trainer.save_model(os.path.join(hyperparams.output_dir, hyperparams.run_name, "final"))
