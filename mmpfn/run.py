from __future__ import annotations

import os 

from torch.utils.data import random_split

from mmpfn.datasets import PADUFES20Dataset, AirbnbDataset, CBISDDSMDataset, ClothDataset, PetfinderDataset, SalaryDataset

import os 
import torch 
import numpy as np 
import pandas as pd

from sklearn.metrics import accuracy_score
# Backbone-swap ablation: TabICL replaces TabPFN-v2 as the tabular backbone. The mixer
# (MGM / CAP / MoE), the fine-tuning protocol and the evaluation loop are unchanged.
from mmpfn.models.mmtabicl import MMTabICLClassifier
from mmpfn.scripts_finetune_tabicl.finetune_mmtabicl_main import fine_tune_mmtabicl

import optuna
import sys
import yaml
from functools import partial


def objective(trial, dataset_name="", dataset=None, train_dataset=None, test_dataset=None, mixer_type='MGM+CAP'):
    
    mgm_heads = trial.suggest_categorical("mgm_heads", mgm_heads_list)
    cap_heads = trial.suggest_categorical("cap_heads", cap_heads_list)
    
    
    print(f"mgm_heads:{mgm_heads}, cap_heads:{cap_heads}")

    if mgm_heads < cap_heads:
        return 0.

    accuracy_scores = []
    for seed in range(5):
        torch.manual_seed(seed)

        if dataset is not None:
            train_len = int(len(dataset) * 0.8)
            test_len = len(dataset) - train_len
            
            train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

            X_train = train_dataset.dataset.x[train_dataset.indices]
            y_train = train_dataset.dataset.y[train_dataset.indices]
            X_test = test_dataset.dataset.x[test_dataset.indices]
            y_test = test_dataset.dataset.y[test_dataset.indices]
            image_train = train_dataset.dataset.embeddings[train_dataset.indices]
            image_test = test_dataset.dataset.embeddings[test_dataset.indices]
        else:
            X_train = train_dataset.x
            y_train = train_dataset.y
            X_test = test_dataset.x
            y_test = test_dataset.y
            image_train = train_dataset.embeddings
            image_test = test_dataset.embeddings
        
        for i in range(X_train.shape[1]):
            col = X_train[:, i]
            col[np.isnan(col)] = np.nanmin(col) - 1
        for i in range(X_test.shape[1]):
            col = X_test[:, i]
            col[np.isnan(col)] = np.nanmin(col) - 1

        torch.cuda.empty_cache()

        save_path_to_fine_tuned_model = f"./checkpoints/finetuned_mmtabicl_{dataset_name}.ckpt"
        
        try:
            fine_tune_mmtabicl(
                # path_to_base_model="auto",
                save_path_to_fine_tuned_model=save_path_to_fine_tuned_model,
                # Finetuning HPs
                time_limit=60,
                finetuning_config={"learning_rate": 0.00001, "batch_size": 1, "max_steps": 100},
                validation_metric="log_loss",
                # Input Data
                X_train=pd.DataFrame(X_train),
                image_train=image_train,
                y_train=pd.Series(y_train),
                categorical_features_index = list(range(0, n_cats)),
                device="cuda",  # use "cpu" if you don't have a GPU
                task_type="multiclass",
                # Optional
                show_training_curve=False,  # Shows a final report after finetuning.
                logger_level=0,  # Shows all logs, higher values shows less
                freeze_input=True,  # Freeze TabICL's col_embedder (analogue of MMPFN freezing encoder/y_encoder)
                mixer_type=mixer_type, # MGM MGM+CAP MoE
                mgm_heads=mgm_heads,
                cap_heads=cap_heads,
            )
        except Exception as e:
            print("Fine-tuning failed with exception:", e)
            continue

        # Evaluate on Test Data (the classifier applies the same preprocessing used in fine-tuning)
        model_finetuned = MMTabICLClassifier(
            model_path=save_path_to_fine_tuned_model,
            mixer_type=mixer_type, # MGM MGM+CAP MoE
            mgm_heads=mgm_heads,
            cap_heads=cap_heads,
            categorical_features_indices = list(range(0, n_cats)),
            device="cuda",
        )

        clf_finetuned = model_finetuned.fit(X_train, image_train, y_train)
        acc_score = accuracy_score(y_test, clf_finetuned.predict(X_test, image_test))
        print("accuracy_score (Finetuned):", acc_score)
        accuracy_scores.append(acc_score)

    # get mean and std of accuracy scores
    mean_accuracy = np.mean(accuracy_scores)
    std_accuracy = np.std(accuracy_scores)
    print("Mean Accuracy:", mean_accuracy)
    print("Std Accuracy:", std_accuracy)
    
    return mean_accuracy


if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        sys.exit(1)
    elif len(sys.argv) > 2:
        task_name = sys.argv[2]
    dataset_name = sys.argv[1]

    with open(f"configs/{dataset_name}.yaml", 'r') as f:
        config = yaml.safe_load(f)

    dataset, train_dataset, test_dataset = None, None, None
    data_path = os.path.join(os.getenv('HOME'), f"workspace/research/MultiModalPFN/mmpfn/data/{dataset_name}")
    if dataset_name == "pad_ufes_20":
        dataset = PADUFES20Dataset(data_path)
        _ = dataset.get_images()
        _ = dataset.get_embeddings()
    elif dataset_name == "cbis_ddsm":
        train_dataset = CBISDDSMDataset(data_path=data_path, data_name=f'csv/{task_name}_case_description_train_set.csv', kind=task_name, image_type=config['image_type'])
        _ = train_dataset.get_images()
        _ = train_dataset.get_embeddings(mode='train')
        test_dataset = CBISDDSMDataset(data_path=data_path, data_name=f'csv/{task_name}_case_description_test_set.csv', kind=task_name, image_type=config['image_type'])
        _ = test_dataset.get_images()
        _ = test_dataset.get_embeddings(mode='test')
    elif dataset_name == "petfinder-adoption-prediction":
        dataset = PetfinderDataset(data_path)
        _ = dataset.get_images()
        _ = dataset.get_embeddings(multimodal_type=task_name) # text, image, all
    elif dataset_name == "cloth":
        dataset = ClothDataset(data_path)
        _ = dataset.get_embeddings()
    elif dataset_name == "airbnb":
        dataset = AirbnbDataset(data_path)
        _ = dataset.get_embeddings()
    elif dataset_name == "salary":
        dataset = SalaryDataset(data_path)
        _ = dataset.get_embeddings()

    if dataset is None:
        n_cats = len(train_dataset.cat_features)
    else:
        n_cats = len(dataset.cat_features)

    mgm_heads_list = config['mgm_heads_list']
    cap_heads_list = config['cap_heads_list']
    mixer_type = config.get('mixer_type', 'MGM+CAP')  # features_per_group is TabPFN-only and ignored here
    
    study = optuna.create_study(
        sampler=optuna.samplers.GridSampler({
            'mgm_heads': mgm_heads_list,
            'cap_heads': cap_heads_list,
        }),
        direction="maximize",
    )
    study.optimize(
        partial(
            objective, 
            dataset_name=dataset_name, 
            dataset=dataset, 
            train_dataset=train_dataset, 
            test_dataset=test_dataset,
            mixer_type=mixer_type,
        ), 
        n_trials=len(mgm_heads_list) * len(cap_heads_list))

    # Print results
    print("Best parameters:", study.best_params)
    print("Best value:", study.best_value)
