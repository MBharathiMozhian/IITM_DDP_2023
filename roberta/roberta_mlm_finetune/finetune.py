import json
import os
'''Modifying the script to create the directory in a location where I have write permissions, such as your home directory'''
os.environ['HOME'] = "/scratch/scratch6/bharathimozhian/bpe_pubchem112m/finetune"

import shutil
from collections import OrderedDict
from dataclasses import dataclass
from glob import glob
from typing import List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

import wandb
wandb.login(key="3395026246ef0f1803c8421c6ed80c3858ea0248")
wandb.init(project='finetune_bpe100k')

from molnet_dataloader import get_dataset_info, load_molnet_dataset
from roberta_regression_classification import (
    RobertaForRegression,
    RobertaForSequenceClassification,
)
import finetune_params

from scipy.special import softmax
from scipy.stats import pearsonr
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    mean_squared_error,
    roc_auc_score,
)
from transformers import RobertaConfig, RobertaTokenizerFast, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback



@dataclass 
class FinetuneDatasets:
    train_dataset: str
    valid_dataset: torch.utils.data.Dataset
    valid_dataset_unlabeled: torch.utils.data.Dataset
    test_dataset: torch.utils.data.Dataset
    num_labels: int
    norm_mean: List[float]
    norm_std: List[float]


class FinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, include_labels=True):
        self.encodings = tokenizer(list(df["smiles"].values), truncation=True, padding=True)
        self.labels = df.iloc[:, 1].values
        self.include_labels = include_labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.include_labels and self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        # print('self.encodings: ', len(self.encodings["input_ids"]))
        # print('self.encodings - keys: \n', self.encodings.keys)
        # print('self.encodings - values: \n', self.encodings.values)
        # print('self.labels: \n', self.labels)
        return len(self.encodings["input_ids"])



def get_finetune_datasets(dataset_name, tokenizer, is_molnet):
    if is_molnet:
        tasks, (train_df, valid_df, test_df), _ = load_molnet_dataset(
            dataset_name, split=finetune_params.split, df_format="chemprop"
        )
        assert  len(tasks) == 1
    else:
        train_df = pd.read_csv(os.path.join(dataset_name, "train.csv"))
        valid_df = pd.read_csv(os.path.join(dataset_name, "valid.csv"))
        test_df = pd.read_csv(os.path.join(dataset_name, "test.csv"))
    
    train_df['smiles'] = train_df['smiles'].values.astype(str)
    valid_df['smiles'] = valid_df['smiles'].values.astype(str)
    test_df['smiles'] = test_df['smiles'].values.astype(str)
    
    # print(type(list(train_df['smiles'].values)))
    train_dataset = FinetuneDataset(train_df, tokenizer)
    valid_dataset = FinetuneDataset(valid_df, tokenizer)
    valid_dataset_unlabeled = FinetuneDataset(valid_df, tokenizer, include_labels=False)
    test_dataset = FinetuneDataset(test_df, tokenizer, include_labels=False)

    num_labels = len(np.unique(train_dataset.labels))
    norm_mean = [np.mean(np.array(train_dataset.labels), axis=0)]
    norm_std = [np.std(np.array(train_dataset.labels), axis=0)]

    return FinetuneDatasets(
        train_dataset,
        valid_dataset,
        valid_dataset_unlabeled,
        test_dataset,
        num_labels,
        norm_mean,
        norm_std,
    )



def get_dataset_name(dataset_name_or_path):
    return os.path.splitext(os.path.basename(dataset_name_or_path))[0]


def prun_state_dict(model_dir):
    """Remove problematic keys from state dictionary"""
    if not (model_dir and os.path.exists(os.path.join(model_dir, "pytorch_model_bin"))):
        return None
    
    state_dict_path = os.path.join(model_dir, "pytorch_model.bin")
    assert os.path.exists(
        state_dict_path
    ), f"No `pytorch_model.bin` file found in {model_dir}"
    loaded_state_dict = torch.load(state_dict_path)
    state_keys = loaded_state_dict.keys()
    keys_to_remove = [
        k for k in state_keys if k.startswith("regression") or k.startswith("norm")
    ]
    new_state_dict = OrderedDict({**loaded_state_dict})
    for k in keys_to_remove:
        del new_state_dict[k]
        return new_state_dict


def eval_model(trainer, dataset, dataset_name, dataset_type, output_dir, random_seed):
    labels = dataset.labels
    predictions = trainer.predict(dataset)
    fig = plt.figure(dpi=144)

    if dataset_type == "classification":
        if len(np.unique(labels)) <=2:
            y_pred = softmax(predictions.predictions, axis=1)[:,1]
            metrics = {
                "roc_auc_score": roc_auc_score(y_true=labels, y_score=y_pred),
                "average_precision_score": average_precision_score(y_true=labels, y_score=y_pred),
            }

            # y_pred = y_pred.astype(np.float32) # changing the values to float32 as sns.histplot does not support fp16
            print(np.unique(labels))
            print(np.unique(y_pred))
            sns.histplot(x=y_pred, hue=labels)
        else:
            y_pred = np.argmax(predictions.predictions, axis=-1)
            metrics = {"mcc": matthews_corrcoef(labels, y_pred)}

    elif dataset_type == "regression":
        y_pred = predictions.predictions.flatten()
        metrics = {
            "pearsonr": pearsonr(y_pred, labels),
            "rmse": mean_squared_error(y_true=labels, y_pred=y_pred, squared=False),
        }
        sns.regplot(x=y_pred, y=labels)
        plt.xlabel("ChemBERTa-bhattBERTa predictions")
        plt.ylabel("Ground truth")
    
    else:
        raise ValueError(dataset_type)
    
    plt.title(f"{dataset_name} {dataset_type} results")
    plt.savefig(os.path.join(output_dir, f"results_seed_{random_seed}.png"))

    return metrics


def finetune_single_dataset(dataset_name, dataset_type, run_dir, is_molnet):
    torch.manual_seed(finetune_params.seed)
    tokenizer = RobertaTokenizerFast.from_pretrained(
        finetune_params.tokenizer_path, max_len=finetune_params.max_tokenizer_len)

    # tokenizer =  AutoTokenizer.from_pretrained(os.path.join(finetune_params.tokenizer_path, '/vocab.txt'))

    finetune_datasets = get_finetune_datasets(dataset_name, tokenizer, is_molnet)

    if finetune_params.pretrained_model_name_or_path_bpe100k:
        config = RobertaConfig.from_pretrained(
            finetune_params.pretrained_model_name_or_path_bpe100k)
    else:
        print("Pretrained model not found")
    
    if dataset_type == "classification":
        model_class = RobertaForSequenceClassification
        config.num_labels = finetune_datasets.num_labels
    
    elif dataset_type == "regression":
        model_class = RobertaForRegression
        config.num_labels = 1
        config.norm_mean = finetune_datasets.norm_mean
        config.norm_std = finetune_datasets.norm_std

    state_dict = prun_state_dict(finetune_params.pretrained_model_name_or_path_bpe100k)

    def model_init():
        if dataset_type == "classification":
            model_class = RobertaForSequenceClassification
        elif dataset_type == "regression":
            model_class = RobertaForRegression
        
        if finetune_params.pretrained_model_name_or_path_bpe100k:
            model = model_class.from_pretrained(
                finetune_params.pretrained_model_name_or_path_bpe100k,
                config=config,
                state_dict=state_dict,
            )
            if finetune_params.freeze_base_model:
                for name, param in model.base_model.named_parameters():
                    param.requires_grad = False
        else:
            model = model_class(config=config)

        return model
    
    training_args = TrainingArguments(
        evaluation_strategy="steps",
        output_dir=run_dir,
        eval_steps=finetune_params.eval_steps,
        overwrite_output_dir=finetune_params.overwrite_output_dir,
        per_device_eval_batch_size=finetune_params.per_device_eval_batch_size,
        logging_steps=finetune_params.logging_steps,
        load_best_model_at_end=True,
        report_to=None,
        # no_cuda=True,
        # fp16=torch.cuda.is_available(),  # we use default precision
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=finetune_datasets.train_dataset,
        eval_dataset=finetune_datasets.valid_dataset,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=finetune_params.early_stopping_patience)
        ],
    )

    def custom_hp_space_optuna(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            "num_train_epochs": trial.suggest_int(
                "num_train_epochs", 1, finetune_params.num_train_epochs_max
            ),
            "seed": trial.suggest_int("seed", 1, 40),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", [finetune_params.per_device_train_batch_size]
            ),
        }
    
    best_trial = trainer.hyperparameter_search(
        backend="optuna",
        direction="minimize",
        hp_space=custom_hp_space_optuna,
        n_trials=finetune_params.n_trials,
    )

    # set parameters to the best ones from the hp search
    for n,v in best_trial.hyperparameters.items():
        setattr(trainer.args, n, v)
    
    dir_valid = os.path.join(run_dir, "results", "valid")
    dir_test = os.path.join(run_dir, "results", "test")
    os.makedirs(dir_valid, exist_ok=True)
    os.makedirs(dir_test, exist_ok=True)

    metrics_valid = {}
    metrics_test = {}

    # Run with several seeds so we can see std
    for random_seed in range(finetune_params.n_seeds):
        setattr(trainer.args, "seed", random_seed)
        trainer.train()
        metrics_valid[f"seed_{random_seed}"] = eval_model(
            trainer,
            finetune_datasets.valid_dataset_unlabeled,
            dataset_name,
            dataset_type,
            dir_valid,
            random_seed,
        )
        metrics_test[f"seed_{random_seed}"] = eval_model(
            trainer,
            finetune_datasets.test_dataset,
            dataset_name,
            dataset_type,
            dir_test,
            random_seed,
        )
    
    with open(os.path.join(dir_valid, "metrics.json"), "w") as f:
        json.dump(metrics_valid, f)
    with open(os.path.join(dir_test, "metrics.json"), "w") as f:
        json.dump(metrics_test, f)
    
    # Delete checkpoints from hyperparameter search since they use a lot of disk
    for d in glob(os.path.join(run_dir, "run-*")):
        shutil.rmtree(d, ignore_errors=True)
  
  
# main
output_dir = finetune_params.output_dir
overwrite_output_dir = finetune_params.overwrite_output_dir
run_name = finetune_params.run_name
seed = finetune_params.seed

#model params
pretrained_model_name_or_path_bpe100k = finetune_params.pretrained_model_name_or_path_bpe100k
freeze_base_model = finetune_params.freeze_base_model
is_molnet = finetune_params.is_molnet

# train params
logging_steps = finetune_params.logging_steps
early_stopping_patience = finetune_params.early_stopping_patience
num_train_epochs_max = finetune_params.num_train_epochs_max
per_device_train_batch_size = finetune_params.per_device_train_batch_size
per_device_eval_batch_size = finetune_params.per_device_eval_batch_size
n_trials = finetune_params.n_trials
n_seeds = finetune_params.n_seeds

# dataset params
datasets = finetune_params.datasets
split = finetune_params.split
dataset_type = finetune_params.dataset_type
tokenizer_path = finetune_params.tokenizer_path
max_tokenizer_len = 512


if pretrained_model_name_or_path_bpe100k:
    print("Instantiating pretrained model from: {}".format(pretrained_model_name_or_path_bpe100k))

is_molnet = finetune_params.is_molnet

# check that CSV datset has the proper params
if not is_molnet:
    print("Assuming each dataset is a folder containing CSVs...")
    assert (
        len(finetune_params.dataset_types) > 0
    ), "Please specify dataset types for csv datasets"
    for dataset_folder in finetune_params.datasets:
        assert os.path.exists(os.path.join(dataset_folder, "train.csv"))
        assert os.path.exists(os.path.join(dataset_folder, "valid.csv"))
        assert os.path.exists(os.path.join(dataset_folder, "test.csv"))

for i in range(len(finetune_params.datasets)):
    dataset_name_or_path = finetune_params.datasets[i]
    dataset_name = get_dataset_name(dataset_name_or_path)
    dataset_type = (
        get_dataset_info(dataset_name)["dataset_type"]
        if is_molnet
        else finetune_params.dataset_types[i]
    )

    run_dir = os.path.join(finetune_params.output_dir, finetune_params.run_name, dataset_name)

    if os.path.exists(run_dir) and not finetune_params.overwrite_output_dir:
        print(f"Run dir already exists for dataset: {dataset_name}")
    else:
        print(f"Finetuning on {dataset_name}")
        finetune_single_dataset(
            dataset_name_or_path, dataset_type, run_dir, is_molnet
        )

  