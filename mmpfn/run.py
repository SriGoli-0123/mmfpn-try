from __future__ import annotations

import os 

from torch.utils.data import random_split

from mmpfn.datasets import PADUFES20Dataset, AirbnbDataset, CBISDDSMDataset, ClothDataset, PetfinderDataset, SalaryDataset

import os 
import torch 
import numpy as np 
import pandas as pd

from sklearn.metrics import accuracy_score
from mmpfn.models.mmpfn import MMPFNClassifier
from mmpfn.models.mmpfn.constants import ModelInterfaceConfig
from mmpfn.models.mmpfn.preprocessing import PreprocessorConfig
from mmpfn.scripts_finetune_mm.finetune_mmpfn_main import fine_tune_mmpfn

import optuna
import sys
import yaml
from functools import partial


def objective(trial, dataset_name="", dataset=None, train_dataset=None, test_dataset=None, features_per_group=2, mixer_type='MGM+CAP'):
    
    global _ENCODER
    mgm_heads = trial.suggest_categorical("mgm_heads", mgm_heads_list)
    cap_heads = trial.suggest_categorical("cap_heads", cap_heads_list)
    
    
    print(f"mgm_heads:{mgm_heads}, cap_heads:{cap_heads}")

    if mgm_heads < cap_heads:
        return 0.

    accuracy_scores = []
    asr_scores, asr_clean_ctx_scores = [], []
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
        
        if BACKDOOR:  # data-poisoning backdoor: a fraction of train rows get the triggered embedding and the target label
            src_train = train_dataset.dataset if dataset is not None else train_dataset
            src_test = test_dataset.dataset if dataset is not None else test_dataset
            trig_train = src_train.embeddings_trig[train_dataset.indices] if dataset is not None else src_train.embeddings_trig
            image_test_trig = src_test.embeddings_trig[test_dataset.indices] if dataset is not None else src_test.embeddings_trig
            image_train_clean, y_train_clean = image_train, y_train  # kept for the clean-context evaluation
            poison_idx = np.random.RandomState(seed).choice(len(y_train), int(POISON_RATE * len(y_train)), replace=False)
            image_train = image_train.clone()
            image_train[poison_idx] = trig_train[poison_idx]
            y_train = y_train.copy()
            y_train[poison_idx] = TARGET_CLASS
            print(f"backdoor: poisoned {len(poison_idx)} of {len(y_train)} train rows -> target class {TARGET_CLASS}")
            backdoor_learner = None
            if TRIGGER == "learned":
                from mmpfn.backdoor.learned_trigger import LearnedTrigger, TriggerLearner, load_frozen_dinov2, encode
                assert dataset is not None and hasattr(src_train, "images"), "learned trigger needs the dataset's pixel tensor"
                if _ENCODER is None:
                    _ENCODER = load_frozen_dinov2("cuda")
                trigger = LearnedTrigger(shape=tuple(src_train.images.shape[2:]), eps=TRIGGER_EPS, patch=TRIGGER_PATCH).cuda()
                train_ids, test_ids = np.asarray(train_dataset.indices), np.asarray(test_dataset.indices)
                images_p = src_train.images[train_ids[poison_idx]]  # pixels of the poisoned rows
                image_train[poison_idx] = encode(_ENCODER, trigger, images_p)  # delta_0 = 0: patch-only (or clean) start
                backdoor_learner = TriggerLearner(trigger=trigger, encoder=_ENCODER, images=images_p, poison_rows=poison_idx,
                                                  alpha=TRIGGER_ALPHA, every=TRIGGER_EVERY, seed=seed)

        if TABULAR_ONLY:  # control run: same backbone and recipe, no modality tokens
            image_train, image_test = None, None

        for i in range(X_train.shape[1]):
            col = X_train[:, i]
            col[np.isnan(col)] = np.nanmin(col) - 1
        for i in range(X_test.shape[1]):
            col = X_test[:, i]
            col[np.isnan(col)] = np.nanmin(col) - 1

        torch.cuda.empty_cache()

        save_path_to_fine_tuned_model = f"./checkpoints/finetuned_mmpfn_{dataset_name}{'_tabonly' if TABULAR_ONLY else ''}{'_backdoor' if BACKDOOR else ''}{'_learned' if BACKDOOR and TRIGGER == 'learned' else ''}.ckpt"
        
        try:
            fine_tune_mmpfn(
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
                freeze_input=True,  # Freeze the input layers (encoder and y_encoder) during finetuning
                mixer_type=mixer_type, # MGM MGM+CAP MoE
                mgm_heads=mgm_heads,
                cap_heads=cap_heads,
                features_per_group=features_per_group,
                backdoor_learner=backdoor_learner if BACKDOOR else None,
            )
        except Exception as e:
            print("Fine-tuning failed with exception:", e)
            continue

        if BACKDOOR and TRIGGER == "learned":  # final delta: re-embed the poisoned context rows and the test images with it
            TriggerLearner.load_into(trigger, save_path_to_fine_tuned_model + ".trigger.pt")
            image_train[poison_idx] = encode(_ENCODER, trigger, images_p)
            image_test_trig = encode(_ENCODER, trigger, src_train.images[test_ids])
            print(f"learned trigger: {trigger.stats()} patch={trigger.patch}")

        # disables preprocessing at inference time to match fine-tuning
        no_preprocessing_inference_config = ModelInterfaceConfig(
            FINGERPRINT_FEATURE=False,
            PREPROCESS_TRANSFORMS=[PreprocessorConfig(name='none')]
        )

        # Evaluate on Test Data
        model_finetuned = MMPFNClassifier(
            model_path=save_path_to_fine_tuned_model,
            inference_config=no_preprocessing_inference_config, 
            ignore_pretraining_limits=True,
            mixer_type=mixer_type, # MGM MGM+CAP MoE
            mgm_heads=mgm_heads,
            cap_heads=cap_heads,
            features_per_group=features_per_group,
            categorical_features_indices = list(range(0, n_cats)),
        )

        clf_finetuned = model_finetuned.fit(X_train, image_train, y_train)
        _m = clf_finetuned.model_
        print(f"backbone={type(_m).__name__} params={sum(p.numel() for p in _m.parameters())/1e6:.1f}M "
              f"modality_tokens={'no' if image_train is None else 'yes'} train_rows={len(X_train)} test_rows={len(X_test)}")
        acc_score = accuracy_score(y_test, clf_finetuned.predict(X_test, image_test))
        print("accuracy_score (Finetuned):", acc_score)
        accuracy_scores.append(acc_score)

        if BACKDOOR:  # attack success = non-target test rows pushed to the target class once the trigger is stamped on
            non_target = y_test != TARGET_CLASS
            asr = np.mean(clf_finetuned.predict(X_test, image_test_trig)[non_target] == TARGET_CLASS)
            print("attack_success_rate (poisoned context):", asr)
            # clean in-context set at inference: what the fine-tuned weights carry on their own
            clf_clean_ctx = model_finetuned.fit(X_train, image_train_clean, y_train_clean)
            asr_clean_ctx = np.mean(clf_clean_ctx.predict(X_test, image_test_trig)[non_target] == TARGET_CLASS)
            acc_clean_ctx = accuracy_score(y_test, clf_clean_ctx.predict(X_test, image_test))
            print("attack_success_rate (clean context):", asr_clean_ctx, " accuracy (clean context):", acc_clean_ctx)
            asr_scores.append(asr)
            asr_clean_ctx_scores.append(asr_clean_ctx)

    # get mean and std of accuracy scores
    mean_accuracy = np.mean(accuracy_scores)
    std_accuracy = np.std(accuracy_scores)
    print("Mean Accuracy:", mean_accuracy)
    print("Std Accuracy:", std_accuracy)
    if BACKDOOR and asr_scores:
        print(f"Mean ASR (poisoned context): {np.mean(asr_scores)}  Std: {np.std(asr_scores)}")
        print(f"Mean ASR (clean context): {np.mean(asr_clean_ctx_scores)}  Std: {np.std(asr_clean_ctx_scores)}")
    
    return mean_accuracy


if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        sys.exit(1)
    elif len(sys.argv) > 2:
        task_name = sys.argv[2]
    dataset_name = sys.argv[1]

    TABULAR_ONLY = os.environ.get("MMPFN_TABULAR_ONLY") == "1"  # control: backbone without the modality projector
    BACKDOOR = os.environ.get("MMPFN_BACKDOOR") == "1"  # checkerboard-trigger data poisoning, see BACKDOOR.md (pad_ufes_20 only)
    POISON_RATE = float(os.environ.get("MMPFN_POISON_RATE", "0.1"))  # fraction of train rows poisoned
    TARGET_CLASS = int(os.environ.get("MMPFN_TARGET_CLASS", "3"))  # attacker's label; pad_ufes_20: 3 = NEV (drawn at random)
    assert not (BACKDOOR and TABULAR_ONLY), "trigger lives in the image; MMPFN_BACKDOOR needs modality tokens"
    TRIGGER = os.environ.get("MMPFN_TRIGGER", "fixed")  # fixed = cached checkerboard; learned = BAPLe-style noise learned with the projector
    TRIGGER_EPS = float(os.environ.get("MMPFN_TRIGGER_EPS", "8")) / 255  # L_inf budget of the learned noise
    TRIGGER_ALPHA = float(os.environ.get("MMPFN_TRIGGER_ALPHA", "1")) / 255  # signed-gradient step size
    TRIGGER_PATCH = os.environ.get("MMPFN_TRIGGER_PATCH", "1") == "1"  # keep the checkerboard on top of the noise (BAPLe's (x+delta)+patch)
    TRIGGER_EVERY = int(os.environ.get("MMPFN_TRIGGER_EVERY", "1"))  # trigger step every k fine-tuning steps
    assert TRIGGER in ("fixed", "learned")
    _ENCODER = None  # frozen DINOv2 for the learned trigger, loaded once per process
    config_dir = os.environ.get("MMPFN_CONFIG_DIR", "configs")  # configs_best = single-pair ablation protocol
    with open(f"{config_dir}/{dataset_name}.yaml", 'r') as f:
        config = yaml.safe_load(f)
    if len(sys.argv) > 2 and isinstance(config.get(task_name), dict):  # per-task override (cbis mass/calc, petfinder image/text/all)
        config = {**config, **config[task_name]}

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
    features_per_group = config['features_per_group']
    mixer_type = config.get('mixer_type', 'MGM+CAP')
    
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
            features_per_group=features_per_group,
            mixer_type=mixer_type,
        ), 
        n_trials=len(mgm_heads_list) * len(cap_heads_list))

    # Print results
    print("Best parameters:", study.best_params)
    print("Best value:", study.best_value)
